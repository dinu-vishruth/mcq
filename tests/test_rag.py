"""
Phase 3 RAG unit tests. Deterministic and offline: forces the hashing embedder
and sqlite vector store so no torch load or network is needed.

Run:  python -m unittest tests.test_rag -v
"""
import os
import sqlite3
import tempfile
import unittest

import config

# Point everything at a temp DB and deterministic backends BEFORE importing
# the RAG modules that read config at build time.
_TMP = tempfile.mkdtemp(prefix="mcq_rag_")
config.DB_PATH = os.path.join(_TMP, "rag.db")
config.EMBEDDING_BACKEND = "hashing"
config.VECTOR_STORE = "sqlite"

from core.models.migrations import run_migrations
from core.rag.chunker import chunk_text
from core.embeddings import get_embedder, reset_embedder
from core.vectorstore import get_vector_store, reset_vector_store
from core.services.ingestion_service import ingest_document


def _init_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript("CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, document_id INTEGER);")
    run_migrations(conn)
    conn.close()


class TestChunker(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text_single_chunk(self):
        chunks = chunk_text("A short sentence about biology.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].index, 0)

    def test_long_text_multiple_chunks_with_overlap(self):
        text = "Sentence about topic number %d. " % 0
        text = " ".join(f"Sentence number {i} about a distinct subtopic of science." for i in range(400))
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        self.assertGreater(len(chunks), 1)
        # Indices are contiguous from 0.
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        # Each chunk respects the size bound (allowing the overlap tail).
        for c in chunks:
            self.assertLessEqual(len(c.content), 500 + 100 + 50)

    def test_offsets_are_ordered(self):
        text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        starts = [c.char_start for c in chunks]
        self.assertEqual(starts, sorted(starts))


class TestHashingEmbedder(unittest.TestCase):
    def setUp(self):
        config.EMBEDDING_BACKEND = "hashing"
        reset_embedder()

    def test_deterministic(self):
        e = get_embedder()
        v1 = e.embed("cellular respiration produces ATP")
        v2 = e.embed("cellular respiration produces ATP")
        self.assertEqual(v1, v2)

    def test_normalized(self):
        e = get_embedder()
        v = e.embed("some text about photosynthesis and chloroplasts")
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_similar_texts_score_higher(self):
        e = get_embedder()
        import numpy as np
        q = np.array(e.embed("ATP is produced in mitochondria"))
        near = np.array(e.embed("mitochondria produce ATP energy"))
        far = np.array(e.embed("the capital of France is Paris"))
        self.assertGreater(float(q @ near), float(q @ far))


class TestIngestionAndRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Re-pin in case another test module changed these at import time.
        config.DB_PATH = os.path.join(_TMP, "rag.db")
        config.EMBEDDING_BACKEND = "hashing"
        config.VECTOR_STORE = "sqlite"
        _init_db()
        reset_embedder()
        reset_vector_store()

    def test_ingest_chunk_embed_retrieve(self):
        text = ("Photosynthesis converts sunlight into glucose in chloroplasts. "
                "Mitochondria produce ATP through cellular respiration. "
                "The nucleus stores genetic material as DNA. "
                "Ribosomes are responsible for protein synthesis. ") * 15
        r = ingest_document(text, owner="tester", title="Cell Biology", source_type="paste")
        self.assertFalse(r["reused"])
        self.assertGreater(r["chunk_count"], 0)

        emb = get_embedder()
        store = get_vector_store()
        hits = store.query(r["document_id"], emb.embed("How is ATP generated?"), top_k=3)
        self.assertTrue(hits)
        # The mitochondria/ATP chunk should rank at or near the top.
        joined = " ".join(h.content for h in hits[:2]).lower()
        self.assertTrue("atp" in joined or "mitochondria" in joined)

    def test_dedup_reuses_document(self):
        text = "Unique content for the dedup test. " * 30
        a = ingest_document(text, owner="tester", title="Dup", source_type="paste")
        b = ingest_document(text, owner="tester", title="Dup", source_type="paste")
        self.assertEqual(a["document_id"], b["document_id"])
        self.assertTrue(b["reused"])

    def test_embedding_cache_avoids_recompute(self):
        # Second ingest of identical text must report reuse (0 new embeddings).
        text = "Cacheable sentence about vectors and retrieval. " * 25
        ingest_document(text, owner="t2", title="Cache", source_type="paste")
        again = ingest_document(text, owner="t2", title="Cache", source_type="paste")
        self.assertTrue(again["reused"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
