"""Retrieval policy tests: MMR selection and document spread.

Pure-unit and offline: MMR is exercised with hand-written vectors, and the
Retriever is driven through fake embedder/store doubles, so no model load and no
database are involved.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.rag.retrieval.retriever import Retriever, mmr_select


class TestMMR(unittest.TestCase):
    def test_empty_candidates(self):
        self.assertEqual(mmr_select(np.array([1.0, 0.0]), [], 3), [])

    def test_returns_at_most_k_unique_indices(self):
        candidates = np.eye(5, dtype="float32")
        picked = mmr_select(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), candidates, 3)
        self.assertEqual(len(picked), 3)
        self.assertEqual(len(set(picked)), 3)
        self.assertTrue(all(0 <= i < 5 for i in picked))

    def test_k_larger_than_candidates_returns_all(self):
        candidates = np.eye(2, dtype="float32")
        picked = mmr_select(np.array([1.0, 0.0]), candidates, 10)
        self.assertEqual(sorted(picked), [0, 1])

    def test_lambda_one_is_pure_relevance_order(self):
        candidates = np.array([
            [0.2, 0.979796],  # relevance to [1, 0] is 0.2
            [0.9, 0.435890],  # highest relevance
            [-0.5, 0.866025],
        ], dtype="float32")
        picked = mmr_select(np.array([1.0, 0.0]), candidates, 2, lambda_mult=1.0)
        self.assertEqual(picked, [1, 0])


class _FakeEmbedder:
    def embed(self, text):
        return np.zeros(3, dtype="float32")


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def query(self, document_id, query_vec, top_k):
        self.calls.append((document_id, top_k))
        return self._hits


def _hit(chunk_index: int):
    return SimpleNamespace(content=f"chunk {chunk_index}", chunk_index=chunk_index, score=0.5)


class TestRetrieverSpread(unittest.TestCase):
    def test_spread_sorts_selected_hits_by_chunk_index(self):
        hits = [_hit(5), _hit(1), _hit(3)]
        store = _FakeStore(hits)
        with patch("core.rag.retrieval.retriever.get_embedder", return_value=_FakeEmbedder()), \
                patch("core.rag.retrieval.retriever.get_vector_store", return_value=store):
            result = Retriever().retrieve(7, "anything", top_k=3, use_mmr=False, spread=True)

        self.assertEqual([h.chunk_index for h in result], [1, 3, 5])
        # Over-fetch gives MMR/spread candidates beyond top_k.
        self.assertEqual(store.calls[0][0], 7)
        self.assertGreaterEqual(store.calls[0][1], 3)

    def test_empty_store_returns_empty(self):
        with patch("core.rag.retrieval.retriever.get_embedder", return_value=_FakeEmbedder()), \
                patch("core.rag.retrieval.retriever.get_vector_store",
                      return_value=_FakeStore([])):
            self.assertEqual(Retriever().retrieve(1, "q", top_k=5), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
