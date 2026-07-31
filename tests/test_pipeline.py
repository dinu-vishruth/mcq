"""
Phase 4 pipeline tests. Deterministic + offline: forces hashing embedder +
sqlite store and stubs the LLM so no network/key is needed.

Run:  python -m unittest tests.test_pipeline -v
"""
import os
import sqlite3
import tempfile
import unittest

import config

_TMP = tempfile.mkdtemp(prefix="mcq_pipe_")
config.DB_PATH = os.path.join(_TMP, "pipe.db")
config.EMBEDDING_BACKEND = "hashing"
config.VECTOR_STORE = "sqlite"

from core.models.migrations import run_migrations
from core.agents.quality_assurance import QualityAssuranceAgent
from core.embeddings import reset_embedder
from core.vectorstore import reset_vector_store


def _init_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript("CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, document_id INTEGER);")
    run_migrations(conn)
    conn.close()


class TestQualityAssurance(unittest.TestCase):
    """QA agent is pure logic -- the contract guardian. Test it thoroughly."""

    def setUp(self):
        self.qa = QualityAssuranceAgent()

    def _good(self):
        return {"question": "What is 2+2?",
                "options": [{"label": "A", "text": "3"}, {"label": "B", "text": "4"},
                            {"label": "C", "text": "5"}, {"label": "D", "text": "6"}],
                "answer_text": "4"}

    def test_accepts_valid(self):
        valid, rejected = self.qa.run([self._good()])
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 0)

    def test_rejects_wrong_option_count(self):
        q = self._good()
        q["options"] = q["options"][:3]
        valid, rejected = self.qa.run([q])
        self.assertEqual(len(valid), 0)
        self.assertIn("4 options", rejected[0]["reason"])

    def test_rejects_duplicate_options(self):
        q = self._good()
        q["options"][2]["text"] = "4"  # duplicate of B
        valid, rejected = self.qa.run([q])
        self.assertEqual(len(valid), 0)

    def test_repairs_label_answer(self):
        q = self._good()
        q["answer_text"] = "B"  # label instead of text
        valid, _ = self.qa.run([q])
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["answer_text"], "4")

    def test_rejects_answer_not_in_options(self):
        q = self._good()
        q["answer_text"] = "42"
        valid, rejected = self.qa.run([q])
        self.assertEqual(len(valid), 0)

    def test_normalises_labels(self):
        q = self._good()
        for o in q["options"]:
            o["label"] = "X"
        valid, _ = self.qa.run([q])
        self.assertEqual([o["label"] for o in valid[0]["options"]], ["A", "B", "C", "D"])

    def test_preserves_extra_metadata(self):
        q = self._good()
        q["bloom"] = "Apply"
        q["source_hint"] = "two plus two"
        valid, _ = self.qa.run([q])
        self.assertEqual(valid[0]["bloom"], "Apply")
        self.assertEqual(valid[0]["source_hint"], "two plus two")


class TestPipelineWithStubbedLLM(unittest.TestCase):
    """End-to-end pipeline with a fake LLM so it's deterministic and offline."""

    @classmethod
    def setUpClass(cls):
        config.DB_PATH = os.path.join(_TMP, "pipe.db")
        config.EMBEDDING_BACKEND = "hashing"
        config.VECTOR_STORE = "sqlite"
        _init_db()
        reset_embedder()
        reset_vector_store()

    def _install_stub(self, questions):
        """Patch get_llm() used by every LLM-calling agent in the pipeline."""
        import core.agents.question as qmod
        import core.agents.difficulty as dmod
        import core.agents.context_validation as cvmod
        import core.agents.fact_verification as fvmod

        class StubLLM:
            def complete_json(self, messages, **kw):
                content = messages[-1]["content"]
                if "Grade the cognitive difficulty" in content:
                    # Mark everything as matching the requested level.
                    import re
                    idxs = [int(n) for n in re.findall(r"^(\d+)\.", content, re.M)]
                    return {"grades": [{"index": i, "level": "medium", "matches_requested": True}
                                       for i in idxs]}
                if "context-sufficiency" in content or "enough substantive" in content:
                    return {"sufficient": True, "confidence": 0.9, "reason": "ok"}
                if "Verify each multiple-choice question" in content:
                    import re
                    idxs = [int(n) for n in re.findall(r"^#(\d+)", content, re.M)]
                    return {"results": [{"index": i, "passed": True, "confidence": 0.95, "issue": ""}
                                        for i in idxs]}
                # Question generation.
                return {"questions": questions}

        stub = StubLLM()
        qmod.get_llm = lambda: stub
        dmod.get_llm = lambda: stub
        cvmod.get_llm = lambda: stub
        fvmod.get_llm = lambda: stub

    def test_pipeline_returns_legacy_shape(self):
        questions = [{
            "question": f"Question {i} about ATP?",
            "options": [{"label": "A", "text": f"a{i}"}, {"label": "B", "text": f"b{i}"},
                        {"label": "C", "text": f"c{i}"}, {"label": "D", "text": f"d{i}"}],
            "answer_text": f"a{i}", "bloom": "Apply", "source_hint": "mitochondria"
        } for i in range(3)]
        self._install_stub(questions)

        from core.services.mcq_pipeline import generate_mcqs_rag
        text = ("Mitochondria produce ATP via cellular respiration. "
                "Chloroplasts perform photosynthesis. DNA is in the nucleus. ") * 20
        mcqs = generate_mcqs_rag(text, num_questions=3, difficulty="medium", owner="t")

        self.assertEqual(len(mcqs), 3)
        for q in mcqs:
            self.assertEqual(set(q.keys()) >= {"question", "options", "answer_text"}, True)
            self.assertEqual(len(q["options"]), 4)
            self.assertIn(q["answer_text"], [o["text"] for o in q["options"]])
            self.assertEqual([o["label"] for o in q["options"]], ["A", "B", "C", "D"])

    def test_generate_from_document_legacy_is_single_shot(self):
        """In legacy mode, making a quiz from a saved resource must NOT run the
        multi-call RAG verification loop (which made the page hang). It should do
        one single-shot generation, so the ContextValidation/FactVerification
        agents are never called."""
        questions = [{
            "question": f"Saved Q{i}?",
            "options": [{"label": "A", "text": f"a{i}"}, {"label": "B", "text": f"b{i}"},
                        {"label": "C", "text": f"c{i}"}, {"label": "D", "text": f"d{i}"}],
            "answer_text": f"a{i}",
        } for i in range(3)]
        self._install_stub(questions)

        # Trip-wire: the legacy path must not touch these agents at all.
        import core.agents.context_validation as cvmod
        import core.agents.fact_verification as fvmod
        def _boom():
            raise AssertionError("verification agent called in legacy single-shot path")
        cvmod.get_llm = _boom
        fvmod.get_llm = _boom

        # The legacy single-shot path delegates to models.mcq_generator.generate_mcqs.
        # Stub it directly (other suites also monkeypatch this module attr, so we
        # set + restore our own to stay isolated) and record it was called.
        import models.mcq_generator as legacy_mod
        calls = {"n": 0}
        def _fake_generate(text, num_questions=5, difficulty="medium"):
            calls["n"] += 1
            return questions[:num_questions]
        prev_gen = legacy_mod.generate_mcqs
        legacy_mod.generate_mcqs = _fake_generate

        prev_mode = config.AI_PIPELINE
        config.AI_PIPELINE = "legacy"
        try:
            # Ingest a document to get a real document_id to generate from.
            from core.services.ingestion_service import ingest_document
            ing = ingest_document(
                ("Normalization reduces redundancy. Primary keys identify rows. "
                 "Indexes speed reads but slow writes. ") * 15,
                owner="t2", title="DBMS", source_type="paste")
            from core.services.mcq_pipeline import generate_from_document
            mcqs = generate_from_document(ing["document_id"], num_questions=3, difficulty="medium")
        finally:
            config.AI_PIPELINE = prev_mode
            legacy_mod.generate_mcqs = prev_gen

        self.assertEqual(calls["n"], 1)          # exactly one single-shot call
        self.assertEqual(len(mcqs), 3)
        for q in mcqs:
            self.assertEqual(len(q["options"]), 4)
            self.assertIn(q["answer_text"], [o["text"] for o in q["options"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
