"""
Embedding Agent: generate embeddings for chunks and store them in the vector
store, with a content-hash cache so unchanged chunks are never re-embedded.

Performance requirements addressed:
  - Cache: chunks whose content_hash already has a vector (embeddings table)
    are skipped -> re-uploading an unchanged doc costs zero embedding calls.
  - Batch: new chunks are embedded via embed_batch in one shot.
  - Incremental: only the missing chunks are embedded and upserted.
"""
from __future__ import annotations

import hashlib

from core.agents.base import Agent
from core.embeddings import get_embedder
from core.vectorstore import get_vector_store
from core.models.db import get_db


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingAgent(Agent):
    name = "embedding"

    def run(self, document_id: int, chunks: list) -> dict:
        """chunks: list of core.rag.chunker.Chunk. Returns a small stats dict."""
        if not chunks:
            return {"embedded": 0, "cached": 0, "total": 0}

        embedder = get_embedder()
        store = get_vector_store()

        # Which chunk hashes already have a cached vector?
        hashes = [_hash(c.content) for c in chunks]
        cached = self._cached_hashes(hashes)

        to_embed_idx = [i for i, h in enumerate(hashes) if h not in cached]
        vectors: dict[int, list[float]] = {}

        if to_embed_idx:
            texts = [chunks[i].content for i in to_embed_idx]
            embedded = embedder.embed_batch(texts)
            for pos, i in enumerate(to_embed_idx):
                vectors[i] = embedded[pos]

        # For cached ones, reuse stored vectors.
        cached_vectors = self._load_cached(hashes, cached)

        items = []
        for i, c in enumerate(chunks):
            vec = vectors.get(i) or cached_vectors.get(hashes[i])
            if vec is None:
                # Defensive: embed individually if a cache miss slipped through.
                vec = embedder.embed(c.content)
            items.append({
                "chunk_index": c.index,
                "content": c.content,
                "vector": vec,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "token_estimate": c.token_estimate,
            })

        store.upsert(document_id, items)
        stats = {"embedded": len(to_embed_idx), "cached": len(chunks) - len(to_embed_idx),
                 "total": len(chunks)}
        self._log(f"doc {document_id}: {stats}")
        return stats

    def _cached_hashes(self, hashes: list[str]) -> set[str]:
        if not hashes:
            return set()
        conn = get_db()
        try:
            q = ",".join("?" * len(hashes))
            rows = conn.execute(
                f"SELECT content_hash FROM embeddings WHERE content_hash IN ({q})", hashes
            ).fetchall()
            return {r["content_hash"] for r in rows}
        finally:
            conn.close()

    def _load_cached(self, hashes: list[str], cached: set[str]) -> dict[str, list[float]]:
        import numpy as np
        wanted = [h for h in hashes if h in cached]
        if not wanted:
            return {}
        conn = get_db()
        try:
            q = ",".join("?" * len(wanted))
            rows = conn.execute(
                f"SELECT content_hash, vector FROM embeddings WHERE content_hash IN ({q})", wanted
            ).fetchall()
            return {r["content_hash"]: np.frombuffer(r["vector"], dtype="float32").tolist()
                    for r in rows}
        finally:
            conn.close()
