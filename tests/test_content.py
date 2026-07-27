"""
Phase 7 extensible-content tests. Offline: forces hashing embedder + sqlite
store and stubs the LLM. Proves the generator registry produces each content
type through the shared retrieve->context->generate spine.

Run:  python -m unittest tests.test_content -v
"""
import os
import sqlite3
import tempfile
import unittest

import config

_TMP = tempfile.mkdtemp(prefix="mcq_content_")
config.DB_PATH = os.path.join(_TMP, "content.db")
config.EMBEDDING_BACKEND = "hashing"
config.VECTOR_STORE = "sqlite"

from core.models.migrations import run_migrations
from core.embeddings import reset_embedder
from core.vectorstore import reset_vector_store


def _init_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript("CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, document_id INTEGER);")
    run_migrations(conn)
    conn.close()


class TestContentRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.DB_PATH = os.path.join(_TMP, "content.db")
        config.EMBEDDING_BACKEND = "hashing"
        config.VECTOR_STORE = "sqlite"
        _init_db()
        reset_embedder()
        reset_vector_store()

    def _stub(self, payload):
        import core.services.content_service as cs

        class Stub:
            def complete_json(self, messages, **kw):
                return payload

        cs.get_llm = lambda: Stub()

    def test_supported_intents(self):
        from core.services.content_service import supported_intents
        intents = supported_intents()
        for expected in ("generate_flashcards", "summarize", "interview_questions",
                         "coding_questions", "explain_topic"):
            self.assertIn(expected, intents)

    def test_flashcards(self):
        self._stub({"flashcards": [{"front": "ATP", "back": "energy currency"}]})
        from core.services.content_service import generate_content
        out = generate_content("generate_flashcards", "Mitochondria make ATP. " * 30,
                               num_items=3, owner="t")
        self.assertEqual(out["intent"], "generate_flashcards")
        self.assertTrue(out["items"])
        self.assertIn("front", out["items"][0])

    def test_summary_no_count(self):
        self._stub({"summary": {"overview": "Cells...", "key_points": ["ATP", "DNA"]}})
        from core.services.content_service import generate_content
        out = generate_content("summarize", "Cells contain organelles. " * 30, owner="t")
        self.assertIn("overview", out["items"])

    def test_unknown_intent_raises(self):
        from core.services.content_service import generate_content
        with self.assertRaises(ValueError):
            generate_content("nonsense_intent", "text " * 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
