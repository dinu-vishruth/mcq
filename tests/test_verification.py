"""
Tests for the new intelligence/verification agents: ContextValidation,
FactVerification, and AnswerEvaluation. Fully offline — every LLM call is
stubbed, so no key/network is required.

Run:  python -m unittest tests.test_verification -v
"""
import unittest

import config
from core.agents.context_validation import ContextValidationAgent
from core.agents.fact_verification import FactVerificationAgent
from core.agents.answer_evaluation import AnswerEvaluationAgent
from core.agents.retriever import RetrieverAgent


class _Stub:
    """Fake LLM returning a scripted JSON payload for complete_json."""
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, messages, **kw):
        self.calls += 1
        # Support a per-call sequence (list of payloads) or a fixed payload.
        if isinstance(self.payload, list):
            return self.payload[min(self.calls - 1, len(self.payload) - 1)]
        return self.payload


def _mcq(q="Q?", answer="a", extra=None):
    m = {"question": q,
         "options": [{"label": "A", "text": "a"}, {"label": "B", "text": "b"},
                     {"label": "C", "text": "c"}, {"label": "D", "text": "d"}],
         "answer_text": answer}
    if extra:
        m.update(extra)
    return m


class TestRetrieverAssess(unittest.TestCase):
    def test_empty_is_low_confidence(self):
        s = RetrieverAgent.assess([])
        self.assertEqual(s["count"], 0)
        self.assertTrue(s["low_confidence"])

    def test_scores_summarised(self):
        class H:
            def __init__(self, sc): self.score = sc
        s = RetrieverAgent.assess([H(0.9), H(0.7), H(0.8)])
        self.assertEqual(s["count"], 3)
        self.assertAlmostEqual(s["avg_score"], 0.8, places=3)
        self.assertFalse(s["low_confidence"])


class TestContextValidation(unittest.TestCase):
    def setUp(self):
        import core.agents.context_validation as mod
        self.mod = mod

    def test_empty_context_is_insufficient(self):
        v = ContextValidationAgent().run("", 5, "medium")
        self.assertFalse(v["sufficient"])

    def test_confident_insufficient_blocks(self):
        self.mod.get_llm = lambda: _Stub({"sufficient": False, "confidence": 0.95, "reason": "just headings"})
        v = ContextValidationAgent().run("Table of contents...", 5, "medium")
        self.assertFalse(v["sufficient"])

    def test_low_confidence_insufficient_fails_open(self):
        # Not confident enough to block -> treated as sufficient.
        self.mod.get_llm = lambda: _Stub({"sufficient": False, "confidence": 0.2, "reason": "unsure"})
        v = ContextValidationAgent().run("Some real content about photosynthesis.", 5, "medium")
        self.assertTrue(v["sufficient"])

    def test_llm_error_fails_open(self):
        def boom():
            raise RuntimeError("no key")
        self.mod.get_llm = boom
        v = ContextValidationAgent().run("content", 5, "medium")
        self.assertTrue(v["sufficient"])


class TestFactVerification(unittest.TestCase):
    def setUp(self):
        import core.agents.fact_verification as mod
        self.mod = mod

    def test_rejects_failed_question(self):
        self.mod.get_llm = lambda: _Stub({"results": [
            {"index": 0, "passed": True, "confidence": 0.95, "issue": ""},
            {"index": 1, "passed": False, "confidence": 0.9, "issue": "two correct answers"},
        ]})
        verified, rejected = FactVerificationAgent().run([_mcq("Q1?"), _mcq("Q2?")], "some context")
        self.assertEqual(len(verified), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("two correct", rejected[0]["reason"])

    def test_low_confidence_pass_is_rejected(self):
        self.mod.get_llm = lambda: _Stub({"results": [
            {"index": 0, "passed": True, "confidence": 0.1, "issue": ""},
        ]})
        verified, rejected = FactVerificationAgent().run([_mcq()], "ctx")
        self.assertEqual(len(verified), 0)
        self.assertEqual(len(rejected), 1)

    def test_ungraded_question_kept_fail_open(self):
        # Model returns no verdict for index 0 -> keep it rather than drop it.
        self.mod.get_llm = lambda: _Stub({"results": []})
        verified, rejected = FactVerificationAgent().run([_mcq()], "ctx")
        self.assertEqual(len(verified), 1)

    def test_llm_error_accepts_all(self):
        def boom():
            raise RuntimeError("down")
        self.mod.get_llm = boom
        verified, rejected = FactVerificationAgent().run([_mcq(), _mcq("Q2?")], "ctx")
        self.assertEqual(len(verified), 2)
        self.assertEqual(len(rejected), 0)

    def test_no_context_skips(self):
        verified, rejected = FactVerificationAgent().run([_mcq()], "")
        self.assertEqual(len(verified), 1)
        self.assertEqual(len(rejected), 0)


class TestAnswerEvaluation(unittest.TestCase):
    def setUp(self):
        import core.agents.answer_evaluation as mod
        self.mod = mod

    def _payload(self, **over):
        p = {"is_correct": True, "score": 90, "confidence": 0.95, "concept_match": True,
             "strengths": ["clear"], "missing_points": [], "incorrect_points": [],
             "feedback": "Nicely done.", "model_answer": "WHERE filters rows before grouping."}
        p.update(over)
        return p

    def test_contract_shape(self):
        self.mod.get_llm = lambda: _Stub(self._payload())
        out = AnswerEvaluationAgent().run("Q", "ref", "student answer")
        for k in ("is_correct", "score", "confidence", "concept_match", "strengths",
                  "missing_points", "incorrect_points", "feedback", "model_answer"):
            self.assertIn(k, out)
        self.assertIsInstance(out["strengths"], list)
        self.assertTrue(0 <= out["score"] <= 100)
        self.assertTrue(0.0 <= out["confidence"] <= 1.0)

    def test_blank_answer_short_circuits(self):
        out = AnswerEvaluationAgent().run("Q", "ref", "   ")
        self.assertFalse(out["is_correct"])
        self.assertEqual(out["score"], 0)

    def test_low_confidence_triggers_reevaluation(self):
        stub = _Stub([self._payload(confidence=0.3),
                      self._payload(confidence=0.9, score=88)])
        self.mod.get_llm = lambda: stub
        out = AnswerEvaluationAgent().run("Q", "ref", "answer")
        self.assertEqual(stub.calls, 2)  # re-evaluated once
        self.assertEqual(out["confidence"], 0.9)

    def test_score_clamped_and_coerced(self):
        self.mod.get_llm = lambda: _Stub(self._payload(score=150, confidence="oops"))
        out = AnswerEvaluationAgent().run("Q", "ref", "answer")
        self.assertEqual(out["score"], 100)
        self.assertEqual(out["confidence"], 0.0)

    def test_llm_error_returns_safe_blank(self):
        def boom():
            raise RuntimeError("no key")
        self.mod.get_llm = boom
        out = AnswerEvaluationAgent().run("Q", "reference text", "answer")
        self.assertFalse(out["is_correct"])
        self.assertEqual(out["model_answer"], "reference text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
