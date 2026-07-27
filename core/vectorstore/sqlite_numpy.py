"""
SQLite + numpy vector store. The portable default and the Vercel fallback.

Vectors live as float32 BLOBs in the `embeddings` table (Phase 2 schema),
joined to `chunks` for content. Search loads a document's vectors into a numpy
matrix and does a single dot product (vectors are L2-normalised, so this is
cosine similarity). No extra services, one DB file to back up, and it works on
Vercel's read-only FS because mcq.db lives in /tmp there.

Scales comfortably to well past 100k chunks at this app's usage.
"""
from __future__ import annotations

import numpy as np

from core.models.db import get_db
from core.vectorstore.base import VectorStore, SearchHit


def _to_blob(vector) -> bytes:
    return np.asarray(vector, dtype="float32").tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="float32")


class SqliteNumpyStore(VectorStore):
    def __init__(self, model_id: str, dim: int):
        self.model_id = model_id
        self.dim = dim

    def upsert(self, document_id: int, items: list[dict]) -> None:
        conn = get_db()
        try:
            for it in items:
                content = it["content"]
                idx = it["chunk_index"]
                vector = it["vector"]
                # content_hash links chunk <-> embedding and dedups identical text.
                import hashlib
                chash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                # Upsert chunk row (idempotent per document_id+chunk_index).
                existing = conn.execute(
                    "SELECT id FROM chunks WHERE document_id=? AND chunk_index=?",
                    (document_id, idx),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO chunks (document_id, chunk_index, content, content_hash, "
                        "char_start, char_end, token_estimate) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (document_id, idx, content, chash, it.get("char_start", 0),
                         it.get("char_end", 0), it.get("token_estimate", 0)),
                    )
                else:
                    conn.execute(
                        "UPDATE chunks SET content=?, content_hash=?, char_start=?, char_end=?, "
                        "token_estimate=? WHERE id=?",
                        (content, chash, it.get("char_start", 0), it.get("char_end", 0),
                         it.get("token_estimate", 0), existing["id"]),
                    )

                # Upsert embedding keyed by content_hash (shared cache across docs).
                row = conn.execute("SELECT id FROM embeddings WHERE content_hash=?", (chash,)).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO embeddings (content_hash, model, dim, vector) VALUES (?, ?, ?, ?)",
                        (chash, self.model_id, self.dim, _to_blob(vector)),
                    )
            conn.commit()
        finally:
            conn.close()

    def query(self, document_id: int, query_vector, top_k: int) -> list[SearchHit]:
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT c.chunk_index, c.content, e.vector "
                "FROM chunks c JOIN embeddings e ON c.content_hash = e.content_hash "
                "WHERE c.document_id=? ORDER BY c.chunk_index",
                (document_id,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        mat = np.vstack([_from_blob(r["vector"]) for r in rows])
        q = np.asarray(query_vector, dtype="float32")
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        scores = mat @ q  # cosine (both normalised)

        k = min(top_k, len(rows))
        top_idx = np.argsort(-scores)[:k]
        return [
            SearchHit(
                chunk_id=f"{document_id}:{rows[i]['chunk_index']}",
                document_id=document_id,
                chunk_index=int(rows[i]["chunk_index"]),
                content=rows[i]["content"],
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    def has_document(self, document_id: int) -> bool:
        conn = get_db()
        try:
            r = conn.execute(
                "SELECT 1 FROM chunks c JOIN embeddings e ON c.content_hash=e.content_hash "
                "WHERE c.document_id=? LIMIT 1",
                (document_id,),
            ).fetchone()
            return r is not None
        finally:
            conn.close()

    def delete_document(self, document_id: int) -> None:
        conn = get_db()
        try:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.commit()
        finally:
            conn.close()
