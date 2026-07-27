"""
Retrieval with MMR (Maximal Marginal Relevance) + document spread.

Plain similarity top-k tends to cluster: for "generate 20 MCQs" it would pull
20 near-duplicate chunks from one section. Two mechanisms fix that:

  - MMR: greedily pick chunks that are relevant to the query BUT dissimilar to
    already-picked chunks, balancing relevance vs. novelty (lambda).
  - Spread: when the caller wants broad coverage (many questions), we also make
    sure picks are distributed across the document's chunk_index range so a
    20-question set draws from the whole document, not just chapter 1.
"""
from __future__ import annotations

import numpy as np

from core.embeddings import get_embedder
from core.vectorstore import get_vector_store


def _cosine_matrix(vectors: np.ndarray, v: np.ndarray) -> np.ndarray:
    return vectors @ v


def mmr_select(query_vec, cand_vecs, k, lambda_mult=0.6):
    """Return indices of selected candidates using MMR."""
    if len(cand_vecs) == 0:
        return []
    cand = np.asarray(cand_vecs, dtype="float32")
    q = np.asarray(query_vec, dtype="float32")
    rel = cand @ q  # relevance to query
    selected: list[int] = []
    remaining = set(range(len(cand)))

    while remaining and len(selected) < k:
        if not selected:
            best = int(np.argmax(rel))
        else:
            sel_mat = cand[selected]
            best, best_score = None, -1e9
            for i in remaining:
                # Max similarity to anything already picked.
                redundancy = float(np.max(sel_mat @ cand[i]))
                score = lambda_mult * float(rel[i]) - (1 - lambda_mult) * redundancy
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.discard(best)
    return selected


class Retriever:
    """Retriever Agent core: similarity search + MMR + optional spread."""

    def retrieve(self, document_id: int, query: str, top_k: int,
                 *, use_mmr: bool = True, spread: bool = False):
        embedder = get_embedder()
        store = get_vector_store()

        # Over-fetch so MMR/spread have candidates to choose from.
        fetch_k = min(max(top_k * 4, top_k + 8), 200)
        qv = embedder.embed(query)
        hits = store.query(document_id, qv, fetch_k)
        if not hits:
            return []

        if use_mmr and len(hits) > top_k:
            cand_vecs = [embedder.embed(h.content) for h in hits]
            idxs = mmr_select(qv, cand_vecs, top_k)
            hits = [hits[i] for i in idxs]

        if spread and len(hits) > 1:
            hits = sorted(hits, key=lambda h: h.chunk_index)

        return hits[:top_k]
