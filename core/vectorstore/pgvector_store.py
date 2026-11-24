"""pgvector-backed vector store (production option).

Stores one ``vector`` column per chunk in the Postgres database behind
DATABASE_URL, so vectors survive restarts and share the DB's durability with
the rest of the app's state. Requires a server with the pgvector extension;
``core.vectorstore.registry`` degrades to the sqlite store with a warning when
that isn't available.

Interface matches core.vectorstore.base.VectorStore, i.e. documents are
namespaced by document_id and vectors are assumed L2-normalised (dot product =
cosine). The table is created against the embedder's current dim; if the dim
changes later the table must be dropped (additive-migration policy: never alter
types in place).
"""
from __future__ import annotations

import numpy as np

from core.models import pg_adapter
from core.models.db import is_postgres
from core.vectorstore.base import SearchHit, VectorStore


class PgVectorStore(VectorStore):
    def __init__(self, model_id: str, dim: int) -> None:
        if pg_adapter.psycopg is None:  # pragma: no cover - guarded by registry
            raise RuntimeError("pgvector store requires the psycopg dependency")
        if not is_postgres():
            raise RuntimeError("pgvector store requires DATABASE_URL (Postgres)")
        self.model_id = model_id
        self.dim = dim
        self._ensure_table()

    # -- connection ---------------------------------------------------------
    def _one(self):
        """A fresh connection per operation.

        Mirrors the "open, use, close" model the SQLite repositories use so the
        adapter is exercised the same way in both backends, with no cursor state
        shared across threads.
        """
        import config

        return pg_adapter.open_postgres(config.DATABASE_URL)

    def _ensure_table(self) -> None:
        conn = self._one()
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.executescript(
                "CREATE TABLE IF NOT EXISTS chunk_vectors ("
                " id SERIAL PRIMARY KEY,"
                " document_id INTEGER NOT NULL,"
                " chunk_index INTEGER NOT NULL,"
                " content TEXT NOT NULL,"
                f" embedding vector({self.dim}) NOT NULL,"
                " UNIQUE(document_id, chunk_index))"
            )
            conn.executescript(
                "CREATE INDEX IF NOT EXISTS idx_chunk_vectors_doc ON chunk_vectors(document_id)"
            )
            conn.commit()
        finally:
            conn.close()

    # -- VectorStore interface ---------------------------------------------
    def upsert(self, document_id: int, items: list[dict]) -> None:
        if not items:
            return
        conn = self._one()
        try:
            for it in items:
                vec = np.asarray(it["vector"], dtype="float32")
                conn.execute(
                    "INSERT INTO chunk_vectors (document_id, chunk_index, content, embedding) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (document_id, chunk_index) "
                    "DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding",
                    (document_id, it["chunk_index"], it["content"], vec.tolist()),
                )
            conn.commit()
        finally:
            conn.close()

    def query(self, document_id: int, query_vector: list[float], top_k: int) -> list[SearchHit]:
        conn = self._one()
        try:
            cur = conn.execute(
                "SELECT chunk_index, content, embedding::text AS stored_vector, 1 - (embedding <=> ?::vector) AS score "
                "FROM chunk_vectors WHERE document_id = ? "
                "ORDER BY embedding <=> ?::vector LIMIT ?",
                (
                    "[" + ",".join(str(float(x)) for x in query_vector) + "]",
                    document_id,
                    "[" + ",".join(str(float(x)) for x in query_vector) + "]",
                    top_k,
                ),
            )
            rows = cur.fetchall()
            return [
                SearchHit(
                    chunk_id=f"{document_id}:{row['chunk_index']}",
                    document_id=document_id,
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    score=float(row["score"]),
                    vector=__import__("json").loads(row["stored_vector"]),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def has_document(self, document_id: int) -> bool:
        conn = self._one()
        try:
            row = conn.execute(
                "SELECT 1 FROM chunk_vectors WHERE document_id = ? LIMIT 1", (document_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def delete_document(self, document_id: int) -> None:
        conn = self._one()
        try:
            conn.execute("DELETE FROM chunk_vectors WHERE document_id = ?", (document_id,))
            conn.commit()
        finally:
            conn.close()
