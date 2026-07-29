"""
FAISS vector store (opt-in via VECTOR_STORE=faiss).

Fastest similarity search at scale. Requires a faiss wheel compatible with the
runtime Python (no cp314 wheel exists at time of writing, so this is off by
default). Index files persist under CHROMA_PATH/faiss; chunk content is read
back from the SQLite `chunks` table so we don't duplicate text on disk.
"""
from __future__ import annotations

import os

import numpy as np

import config
from core.models.db import get_db
from core.vectorstore.base import VectorStore, SearchHit


class FaissStore(VectorStore):
    def __init__(self, model_id: str, dim: int):
        import faiss  # lazy import
        self._faiss = faiss
        self.model_id = model_id
        self.dim = dim
        self._dir = os.path.join(config.CHROMA_PATH, "faiss")
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, document_id: int) -> str:
        return os.path.join(self._dir, f"doc_{document_id}.index")

    def upsert(self, document_id: int, items: list[dict]) -> None:
        if not items:
            return
        # Persist chunk text via the SQLite store so content is recoverable.
        from core.vectorstore.sqlite_numpy import SqliteNumpyStore
        SqliteNumpyStore(self.model_id, self.dim).upsert(document_id, items)

        ordered = sorted(items, key=lambda x: x["chunk_index"])
        mat = np.asarray([it["vector"] for it in ordered], dtype="float32")
        index = self._faiss.IndexFlatIP(self.dim)  # inner product = cosine (normalised)
        index.add(mat)
        self._faiss.write_index(index, self._path(document_id))

    def query(self, document_id: int, query_vector, top_k: int) -> list[SearchHit]:
        path = self._path(document_id)
        if not os.path.exists(path):
            return []
        index = self._faiss.read_index(path)
        q = np.asarray([query_vector], dtype="float32")
        k = min(top_k, index.ntotal)
        if k == 0:
            return []
        scores, idxs = index.search(q, k)

        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT chunk_index, content FROM chunks WHERE document_id=? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
        finally:
            conn.close()

        hits = []
        for rank, ci in enumerate(idxs[0]):
            if ci < 0 or ci >= len(rows):
                continue
            hits.append(SearchHit(
                chunk_id=f"{document_id}:{rows[ci]['chunk_index']}",
                document_id=document_id,
                chunk_index=int(rows[ci]["chunk_index"]),
                content=rows[ci]["content"],
                score=float(scores[0][rank]),
            ))
        return hits

    def has_document(self, document_id: int) -> bool:
        return os.path.exists(self._path(document_id))

    def delete_document(self, document_id: int) -> None:
        path = self._path(document_id)
        if os.path.exists(path):
            os.remove(path)
        from core.vectorstore.sqlite_numpy import SqliteNumpyStore
        SqliteNumpyStore(self.model_id, self.dim).delete_document(document_id)
