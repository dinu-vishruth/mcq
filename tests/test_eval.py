"""Eval harness: scoring contracts + an end-to-end dry run.

Scoring is the part that must be trustworthy: a harness that silently reports
1.00 for broken output is worse than no harness. These tests pin each metric and
the pass/fail gates, then run one stub case through main() to catch plumbing
regressions (dataset parsing, report writing).
"""
import json
import tempfile
import unittest
from pathlib import Path

from evals.run_eval import DEFAULT_DATASET, generate_stub, load_cases, main, score_case


def _mcq(question: str = "What does the source say about ATP?",
         answer: str = "mitochondria produce ATP",
         options: list[str] | None = None) -> dict:
    texts = options or [answer, "the nucleus stores DNA", "ribosomes build lipids",
                        "oxygen is not used"]
    return {
        "question": question,
        "options": [{"label": label, "text": text}
                    for label, text in zip("ABCD", texts)],
        "answer_text": answer,
        "source_hint": "Mitochondria produce ATP through respiration.",
        "explanation": "The source places ATP production in mitochondria.",
    }


CASE = {
    "id": "unit",
    "num_questions": 2,
    "difficulty": "medium",
    "document": "Mitochondria produce ATP through respiration. The nucleus stores DNA.",
    "expected_keywords": ["ATP", "nucleus"],
}


class TestScoring(unittest.TestCase):
    def test_perfect_batch_passes(self):
        nucleus = _mcq(question="What does the nucleus store?", answer="DNA",
                       options=["DNA", "starch", "glucose", "oxygen"])
        nucleus.update(source_hint="The nucleus stores DNA.", explanation="The source states that the nucleus stores DNA.")
        result = score_case(CASE, [_mcq(), nucleus], 1.2)
        self.assertEqual(result["validity"], 1.0)
        self.assertEqual(result["grounded"], 1.0)
        self.assertEqual(result["source_quote_rate"], 1.0)
        self.assertEqual(result["keyword_coverage"], 1.0)
        self.assertEqual(result["duplicate_rate"], 0.0)
        self.assertTrue(result["passed"])

    def test_invalid_contract_fails_validity(self):
        broken = _mcq(options=["only one", "two"])
        result = score_case(CASE, [broken], 0.1)
        self.assertEqual(result["validity"], 0.0)
        self.assertFalse(result["passed"])

    def test_answer_match_is_only_a_lexical_metric(self):
        fabricated = _mcq(answer="chloroplasts generate ATP")
        result = score_case(CASE, [fabricated], 0.1)
        self.assertEqual(result["answer_text_match_rate"], 0.0)
        self.assertEqual(result["source_quote_rate"], 1.0)
        self.assertFalse(result["passed"])

    def test_duplicate_questions_are_flagged(self):
        result = score_case(CASE, [_mcq(), _mcq(question="WHAT does the source say about ATP?!")], 0.1)
        self.assertEqual(result["duplicate_rate"], 0.5)
        self.assertFalse(result["passed"])

    def test_missing_keywords_lower_coverage(self):
        case = dict(CASE, expected_keywords=["photosynthesis", "chlorophyll"])
        result = score_case(case, [_mcq(), _mcq(question="Where is DNA stored?")], 0.1)
        self.assertEqual(result["keyword_coverage"], 0.0)
        self.assertFalse(result["passed"])

    def test_shortfall_is_reported(self):
        result = score_case(CASE, [_mcq()], 0.1)
        self.assertEqual(result["count_ratio"], 0.5)
        self.assertFalse(result["passed"])

    def test_malformed_types_and_normalized_option_duplicates_fail(self):
        broken = _mcq(options=["mitochondria produce ATP", "ｏｘｙｇｅｎ", "OXYGEN", "other"])
        result = score_case(CASE, [broken, "not an object"], 0.1)
        self.assertEqual(result["validity"], 0.0)
        self.assertFalse(result["passed"])

    def test_quote_must_appear_in_the_evidence_actually_supplied(self):
        result = score_case(CASE, [_mcq(), _mcq(question="What does the nucleus store?")], 0.1,
                            evidence="The nucleus stores DNA.")
        self.assertEqual(result["source_quote_rate"], 0.0)
        self.assertEqual(result["grounding_contract_rate"], 1.0)
        self.assertEqual(result["validity"], 1.0)
        self.assertFalse(result["passed"])

    def test_withholding_requires_positive_confirmation(self):
        case = {**CASE, "expected_outcome": "withhold"}
        empty = score_case(case, [], 0.1)
        withheld = score_case(case, [], 0.1, withheld=True)
        unexpected = score_case(case, [_mcq()], 0.1, withheld=True)
        self.assertEqual((empty["status"], empty["passed"]), ("failed", False))
        self.assertEqual((withheld["status"], withheld["passed"]), ("withheld", True))
        self.assertFalse(unexpected["passed"])


class TestDatasetAndStub(unittest.TestCase):
    def test_dataset_parses_and_has_required_fields(self):
        cases = load_cases(DEFAULT_DATASET)
        self.assertGreaterEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["document"])
                self.assertGreaterEqual(case["num_questions"], 1)
                self.assertIn(case["difficulty"], ("easy", "medium", "hard"))

    def test_stub_generates_requested_count_of_valid_grounded_mcqs(self):
        case = next(c for c in load_cases(DEFAULT_DATASET) if c["id"] == "biology-cell-energy")
        mcqs = generate_stub(case)
        self.assertEqual(len(mcqs), case["num_questions"])
        result = score_case(case, mcqs, 0.0)
        self.assertEqual(result["validity"], 1.0)
        self.assertEqual(result["grounded"], 1.0)
        self.assertTrue(result["passed"])

    def test_dry_run_writes_json_and_markdown(self):
        directory = Path(tempfile.mkdtemp(prefix="mcq_eval_test_"))
        output, markdown = directory / "latest.json", directory / "latest.md"
        code = main(["--dry-run", "--limit", "1", "--output", str(output),
                     "--markdown", str(markdown)])
        self.assertEqual(code, 0)

        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["mode"], "stub")
        self.assertEqual(report["summary"]["cases"], 1)
        self.assertEqual(len(report["cases"]), 1)
        self.assertTrue(markdown.read_text(encoding="utf-8").startswith("# MCQ evaluation"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
