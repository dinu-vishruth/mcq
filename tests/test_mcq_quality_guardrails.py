"""Phase 1 audit regressions: mocked evidence contracts, never live quality claims."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
from core.agents.context_validation import ContextValidationAgent
from core.agents.difficulty import DifficultyAgent
from core.agents.fact_verification import FactVerificationAgent
from core.agents.quality_assurance import QualityAssuranceAgent
from core.llm.base import LLMError, LLMProvider
from core.models.mcq_contract import question_key
from core.services import mcq_pipeline as pipeline
from core.services.mcq_quality import accept_candidates
from models.mcq_generator import MCQGenerationError, generate_mcqs

SOURCE = 'WHERE filters rows. HAVING filters groups.'


def mcq(stem='What does WHERE filter?', **changes):
    q = {'question': stem, 'options': [{'label': k, 'text': v} for k, v in
         zip('ABCD', ['rows', 'groups', 'tables', 'columns'])], 'answer_text': 'rows',
         'explanation': 'WHERE filters rows before grouping.', 'source_hint': 'WHERE filters rows'}
    q.update(changes)
    return q


class TestSharedQuality(unittest.TestCase):
    def setUp(self):
        for target, value in [('FACT_VERIFICATION_ENABLED', True), ('CONTEXT_VALIDATION_ENABLED', False),
                              ('AI_PIPELINE', 'rag'), ('IS_VERCEL', False)]:
            stub = patch.object(config, target, value)
            stub.start()
            self.addCleanup(stub.stop)
        self.facts = Mock()
        self.facts.complete_json.side_effect = lambda messages, **kwargs: {
            'results': [{'index': i, 'passed': True, "safe": True, 'confidence': 0.95} for i in range(30)]}
        # Individual tests set complete per-batch verdicts. Out-of-range indices
        # must not be treated as approval, even in a test double.
        self.grades = Mock()

    def checkers(self, count=1, passed=True, level='medium'):
        self.facts.complete_json.side_effect = None
        self.facts.complete_json.return_value = {'results': [
            {'index': i, 'passed': passed, 'safe': True, 'confidence': 0.95} for i in range(count)]}
        self.grades.complete_json.return_value = {'grades': [
            {'index': i, 'matches_requested': True, 'level': level} for i in range(count)]}
        facts = patch('core.agents.fact_verification.get_llm', return_value=self.facts)
        grades = patch('core.agents.difficulty.get_llm', return_value=self.grades)
        facts.start()
        grades.start()
        self.addCleanup(facts.stop)
        self.addCleanup(grades.stop)

    def test_structural_primary_vs_fallback(self):
        invalid = [mcq('Absent?', answer_text='absent'), mcq('Duplicate?'), mcq('Blank?')]
        invalid[1]['options'][1]['text'] = ' ROWS '
        invalid[2]['options'][0]['text'] = ''
        self.assertEqual(accept_candidates(invalid, SOURCE, 'medium').accepted, [])
        with patch('models.mcq_generator.generate_legacy_candidates', return_value=invalid):
            with self.assertRaises(MCQGenerationError):
                pipeline._legacy_fallback(SOURCE, 3, 'medium')

    def test_duplicate_policy(self):
        self.checkers()
        batch = [mcq(), mcq('What does WHERE filter ?'), mcq('WHAT does where FILTER!')]
        result = accept_candidates(batch, SOURCE, 'medium')
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(len(result.rejected), 2)
        with patch('models.mcq_generator.generate_legacy_candidates', return_value=batch):
            recovered = pipeline._legacy_fallback(SOURCE, 3, 'medium')
        self.assertEqual(len(recovered), 1)
        repeated = accept_candidates([mcq()], SOURCE, 'medium', excluded=result.accepted)
        self.assertEqual(repeated.accepted, [])
        self.assertNotEqual(question_key('Is x > y?'), question_key('Is x < y?'))

    def test_all_verifier_malformed_response_probes(self):
        malformed = [{}, {'results': []}, {'results': None}, {'results': [None]},
                     {'results': [{'index': 0}]}, {'results': 'bad'}, [], None,
                     {'results': [{'index': 0, 'passed': 'false', "safe": True, 'confidence': 0.99}]}]
        for confidence in ['bad', '0.9', True, float('nan'), float('inf'), -1, 1.1, None]:
            malformed.append({'results': [{'index': 0, 'passed': True, "safe": True, 'confidence': confidence}]})
        for data in malformed:
            with self.subTest(data=data), patch('core.agents.fact_verification.get_llm',
                                               return_value=Mock(complete_json=Mock(return_value=data))):
                result = FactVerificationAgent().run([mcq()], SOURCE)
                self.assertEqual(result.verified, [])
                self.assertEqual(len(result.unverified), 1)
        with patch('core.agents.fact_verification.get_llm', side_effect=LLMError('down')):
            result = FactVerificationAgent().run([mcq()], SOURCE)
            self.assertEqual(len(result.unverified), 1)
            self.assertFalse(result.verified)
        result = FactVerificationAgent().run([mcq()], '')
        self.assertEqual(len(result.unverified), 1)

    def test_explicit_false_is_rejected_and_duplicate_indices_are_unverified(self):
        self.checkers(passed=False)
        result = FactVerificationAgent().run([mcq()], SOURCE)
        self.assertEqual(len(result.rejected), 1)
        self.assertEqual(result.unverified, [])
        self.facts.complete_json.return_value['results'] += [
            {'index': 0, 'passed': True, "safe": True, 'confidence': 1}]
        result = FactVerificationAgent().run([mcq()], SOURCE)
        self.assertEqual(result.verified, [])
        self.assertEqual(len(result.unverified), 1)

    def test_string_boolean_other_agents(self):
        with patch('core.agents.context_validation.get_llm', return_value=Mock(
                complete_json=Mock(return_value={'sufficient': 'false', 'confidence': 0.99}))):
            verdict = ContextValidationAgent().run(SOURCE, 1, 'medium')
            self.assertFalse(verdict['sufficient'])
            self.assertEqual(verdict['status'], 'unverified')
        for payload in [{'grades': None}, {'grades': [None]}, {'grades': []},
                        {'grades': [{'index': 0, 'matches_requested': 'false', 'level': 'medium'}]}]:
            with self.subTest(payload=payload), patch('core.agents.difficulty.get_llm', return_value=Mock(
                    complete_json=Mock(return_value=payload))):
                result = DifficultyAgent().run([mcq()], 'medium')
                self.assertEqual(result['matched'], [])
                self.assertEqual(result['unverified_indices'], [0])

    def test_fact_rejection_bypassed_by_fallback_is_fixed(self):
        self.checkers(passed=False)
        wrong = mcq(answer_text='groups')
        with patch.object(pipeline.QuestionAgent, 'run', return_value=[wrong]), \
                patch('models.mcq_generator.generate_legacy_candidates', return_value=[wrong]):
            with self.assertRaises(MCQGenerationError):
                pipeline._generate_checked(SOURCE, 1, 'medium')
        self.assertGreaterEqual(self.facts.complete_json.call_count, 2)

    def test_five_of_ten(self):
        self.checkers(count=5)
        five = [mcq(f'Question {i}?') for i in range(5)]
        with patch.object(pipeline.QuestionAgent, 'run', return_value=five) as primary, \
                patch('models.mcq_generator.generate_legacy_candidates', return_value=five) as fallback:
            result = pipeline._generate_checked(SOURCE, 10, 'medium', topic='WHERE')
        self.assertEqual(len(result), 5)
        self.assertEqual([c.args[1] for c in primary.call_args_list], [10, 5, 5])
        self.assertTrue(all(c.args[1] == 5 for c in fallback.call_args_list))
        self.assertEqual(len(fallback.call_args.kwargs['excluded']), 5)
        self.assertEqual(fallback.call_args.kwargs['topic'], 'WHERE')

    def test_later_failure_discards_verified_partial_is_fixed(self):
        self.checkers()
        with patch.object(pipeline.QuestionAgent, 'run', side_effect=[[mcq()], LLMError('late failure')]), \
                patch('models.mcq_generator.generate_legacy_candidates', side_effect=LLMError('fallback down')) as fallback:
            result = pipeline._generate_checked(SOURCE, 2, 'medium')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['verification_status'], 'verified')
        self.assertEqual(fallback.call_args.args[1], 1)
        self.assertEqual(fallback.call_args.args[0], SOURCE)

    def test_six_source_controlled_comparison(self):
        cases = [json.loads(line) for line in (Path(__file__).parents[1] / 'evals/dataset.jsonl').read_text().splitlines()]
        fallback_accepted = 0
        for case in cases:
            with self.subTest(source=case['id']):
                quote = ' '.join(case['document'].split()[:4])
                invalid = [mcq('Unsupported?', answer_text='groups', source_hint=quote),
                           mcq('Absent?', answer_text='absent', source_hint=quote),
                           mcq('Duplicate?', source_hint=quote)]
                invalid[2]['options'][1]['text'] = 'rows'
                self.checkers(passed=False, level=case['difficulty'])
                primary = accept_candidates(copy.deepcopy(invalid), case['document'], case['difficulty'])
                self.assertEqual(primary.accepted, [])
                with patch('models.mcq_generator.generate_legacy_candidates', return_value=invalid):
                    try:
                        output = pipeline._legacy_fallback(case['document'], case['num_questions'],
                                                           case['difficulty'], topic=case['topic'])
                    except MCQGenerationError:
                        output = []
                fallback_accepted += len(output)
        self.assertEqual(fallback_accepted, 0, 'None of the 18 controlled bad candidates may be published')

    def test_qa_not_typed_schema_is_fixed(self):
        cases = [None, True, 3, [], 'bad', {}, {'question': 17, 'options': [], 'answer_text': 'x'},
                 mcq(options=None), mcq(explanation=1), mcq(source_hint=[]), mcq(answer_text=True)]
        valid, rejected = QualityAssuranceAgent().run(cases, require_grounding=True, context=SOURCE)
        self.assertEqual(valid, [])
        self.assertEqual(len(rejected), len(cases))
        self.assertEqual(QualityAssuranceAgent().run(None)[0], [])

    def test_option_normalization_and_retry_date(self):
        for duplicate in [' ROWS ', 'Rows', 'ｒｏｗｓ']:
            candidate = mcq()
            candidate['options'][1]['text'] = duplicate
            self.assertEqual(QualityAssuranceAgent().run([candidate])[0], [])
        # Phase 2 parses HTTP-date Retry-After values.
        response = Mock(headers={'Retry-After': 'Wed, 25 Nov 2099 00:00:00 GMT'})
        self.assertGreater(LLMProvider._parse_retry_after(response), 0)

    def test_metadata_and_quote_required_for_fresh_generation(self):
        for changes in [{'explanation': None}, {'source_hint': None}, {'source_hint': 'invented quote'}]:
            result = accept_candidates([mcq(**changes)], SOURCE, 'medium')
            self.assertEqual(result.accepted, [])
            self.assertEqual(len(result.rejected), 1)

    def test_disabled_verifier_is_not_a_legacy_escape(self):
        with patch.object(config, 'FACT_VERIFICATION_ENABLED', False):
            result = accept_candidates([mcq()], SOURCE, 'medium')
            self.assertEqual(len(result.unverified), 1)
            with self.assertRaises(MCQGenerationError):
                pipeline._legacy_fallback(SOURCE, 1, 'medium')

    def test_public_legacy_generator_uses_shared_policy_and_exclusions(self):
        self.checkers()
        with patch.object(config, 'AI_PIPELINE', 'legacy'), patch('config.llm_key_present', return_value=True), \
                patch('models.mcq_generator.generate_legacy_candidates', side_effect=[[mcq()], [mcq('Second?')]]) as generate:
            result = generate_mcqs(SOURCE, 2, 'medium')
        self.assertEqual(len(result), 2)
        self.assertEqual(len(generate.call_args.kwargs['excluded']), 1)
        self.assertEqual(config.AI_PIPELINE, 'rag')

    def test_rejected_question_requires_successful_revalidation(self):
        self.checkers()
        self.facts.complete_json.side_effect = [
            {'results': [{'index': 0, 'passed': False, "safe": True, 'confidence': 0.95}]},
            {'results': [{'index': 0, 'passed': True, "safe": True, 'confidence': 0.95}]}]
        with patch('models.mcq_generator.generate_legacy_candidates', return_value=[mcq()]):
            result = pipeline._legacy_fallback(SOURCE, 1, 'medium')
        self.assertEqual(len(result), 1)
        self.assertEqual(self.facts.complete_json.call_count, 2)

    def test_later_verifier_failure_preserves_accepted_questions(self):
        self.checkers()
        for stage in ['fact', 'difficulty']:
            with self.subTest(stage=stage):
                self.facts.complete_json.side_effect = None
                self.grades.complete_json.side_effect = None
                provider = self.facts if stage == 'fact' else self.grades
                successful = copy.deepcopy(provider.complete_json.return_value)
                provider.complete_json.side_effect = [successful, LLMError('later failure'),
                                                      LLMError('still down')]
                with patch.object(pipeline.QuestionAgent, 'run',
                                  side_effect=[[mcq()], [mcq('Second?')], [mcq('Third?')]]):
                    result = pipeline._collect_mcqs(SOURCE, 2, 'medium')
                self.assertEqual([q['question'] for q in result], [mcq()['question']])

    def test_missing_verdict_does_not_discard_an_independently_verified_item(self):
        self.checkers()
        result = FactVerificationAgent().run([mcq(), mcq('Second?')], SOURCE)
        self.assertEqual(len(result.verified), 1)
        self.assertEqual(len(result.unverified), 1)

    def test_prompt_constraints_are_carried_to_every_candidate_strategy(self):
        from core.prompts.mcq import legacy_prompt, rag_prompt
        for prompt in [legacy_prompt, rag_prompt]:
            text = prompt(SOURCE, 5, 'hard', topic='SQL row filters', excluded=[mcq()])
            self.assertIn('SQL row filters', text)
            self.assertIn('What does WHERE filter?', text)
            self.assertIn('HARD', text)
            self.assertIn(SOURCE, text)

    def test_verification_status_cannot_be_forged_by_candidate(self):
        self.checkers(passed=False)
        result = accept_candidates([mcq(verification_status='verified')], SOURCE, 'medium')
        self.assertEqual(result.accepted, [])

    def test_context_verdict_contract_fails_closed_for_all_invalid_shapes(self):
        for payload in [None, [], {}, {'sufficient': True},
                        {'sufficient': True, 'confidence': '0.99'},
                        {'sufficient': True, 'confidence': float('nan')},
                        {'sufficient': False, 'confidence': 0.95}]:
            with self.subTest(payload=payload), patch('core.agents.context_validation.get_llm',
                    return_value=Mock(complete_json=Mock(return_value=payload))):
                self.assertFalse(ContextValidationAgent().run(SOURCE, 1, 'medium')['sufficient'])

    def test_fallback_does_not_mutate_process_pipeline_policy(self):
        self.checkers()

        def candidates(*args, **kwargs):
            self.assertEqual(config.AI_PIPELINE, 'rag')
            return [mcq()]

        with patch('models.mcq_generator.generate_legacy_candidates', side_effect=candidates):
            self.assertEqual(len(pipeline._legacy_fallback(SOURCE, 1, 'medium')), 1)
        self.assertEqual(config.AI_PIPELINE, 'rag')
