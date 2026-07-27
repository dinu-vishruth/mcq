"""
Phase 5 learning-intelligence tests. Offline: stubs the LLM, forces a temp DB.

Run:  python -m unittest tests.test_learning -v
"""
import os
import sqlite3
import tempfile
import unittest

import config

_TMP = tempfile.mkdtemp(prefix="mcq_learn_")
config.DB_PATH = os.path.join(_TMP, "learn.db")
config.LLM_API_KEY = "test-key"  # force the AI path (stubbed below)

from core.models.migrations import run_migrations
from core.agents.explanation import ExplanationAgent
from core.agents.evaluation import EvaluationAgent
from core.repositories import learning_repo


def _init_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript("CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY);")
    run_migrations(conn)
    conn.close()


class TestExplanationContract(unittest.TestCase):
    """The list[str], positionally-aligned contract result.html depends on."""

    def setUp(self):
        config.DB_PATH = os.path.join(_TMP, "learn.db")
        self.details = [
            {"question": "Q1", "selected": "4", "correct": "4", "is_correct": True},
            {"question": "Q2", "selected": "Berlin", "correct": "Paris", "is_correct": False},
            {"question": "Q3", "selected": "", "correct": "Blue", "is_correct": False},
        ]

    def test_no_key_static_fallback(self):
        old = config.LLM_API_KEY
        config.LLM_API_KEY = ""
        try:
            out = ExplanationAgent().run(self.details)
            self.assertEqual(len(out), 3)
            self.assertTrue(all(isinstance(x, str) for x in out))
        finally:
            config.LLM_API_KEY = old

    def test_ai_path_alignment(self):
        import core.agents.explanation as em

        class Stub:
            def complete_json(self, messages, **kw):
                return {"explanations": ["Berlin is in Germany; Paris is the capital of France."]}

        em.get_llm = lambda: Stub()
        out = ExplanationAgent().run(self.details)
        self.assertEqual(len(out), 3)                 # one per question
        self.assertIn("Correct", out[0])              # right answer
        self.assertIn("Paris", out[1])                # AI explanation merged in
        self.assertIn("Not answered", out[2])         # unanswered handled


class TestEvaluationAndWeakTopics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.DB_PATH = os.path.join(_TMP, "learn.db")
        _init_db()

    def _stub_topics(self, mapping):
        import core.agents.evaluation as ev

        class Stub:
            def complete_json(self, messages, **kw):
                content = messages[-1]["content"]
                if "assign a short" in content:
                    return {"topics": [{"index": i, "topic": t} for i, t in mapping.items()]}
                return {"recommendations": ["Review photosynthesis basics."]}

        ev.get_llm = lambda: Stub()

    def test_weak_topic_detection_and_persistence(self):
        details = [
            {"question": "About photosynthesis?", "selected": "x", "correct": "y", "is_correct": False},
            {"question": "About photosynthesis again?", "selected": "x", "correct": "y", "is_correct": False},
            {"question": "About mitosis?", "selected": "y", "correct": "y", "is_correct": True},
        ]
        self._stub_topics({0: "Photosynthesis", 1: "Photosynthesis", 2: "Mitosis"})

        result = EvaluationAgent().run(details, user_id=42, session_key="abc", difficulty="medium")
        stats = result["topic_stats"]
        self.assertEqual(stats["Photosynthesis"]["wrong"], 2)
        self.assertEqual(stats["Photosynthesis"]["total"], 2)
        self.assertEqual(stats["Mitosis"]["wrong"], 0)

        weak = learning_repo.top_weak_topics(42, limit=5)
        self.assertTrue(weak)
        self.assertEqual(weak[0]["topic"], "Photosynthesis")
        self.assertEqual(weak[0]["pct"], 100)

    def test_recommendations(self):
        self._stub_topics({})
        recs = EvaluationAgent().recommend([{"topic": "Photosynthesis", "wrong": 2, "total": 2}])
        self.assertTrue(recs)
        self.assertIsInstance(recs[0], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
