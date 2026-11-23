"""
Document + chunk data access over the Phase 2 schema.

Dedup is by doc_hash (sha256 of cleaned full text): re-uploading the same
document returns the existing row instead of re-ingesting, which satisfies the
"avoid regenerating embeddings" performance requirement.
"""
from __future__ import annotations

import hashlib
import json

from core.models.db import get_db


def doc_hash(cleaned_text: str) -> str:
    return hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()


def find_by_hash(dh: str, owner: str | None = None, *, owner_id: int | None = None):
    """Find a document by content hash, scoped to ``owner`` when given.

    Scoping matters: dedup is per user. A global lookup handed the second person
    to upload a given document the FIRST person's row, so no row was created for
    them and their library came back empty. Pass the owner for any dedup decision.

    ``owner=None`` keeps the old global behaviour for callers that legitimately
    want any copy of the content (embedding reuse, admin lookups).
    """
    conn = get_db()
    try:
        if owner_id is not None:
            return conn.execute("SELECT * FROM documents WHERE doc_hash=? AND owner_user_id=?", (dh, owner_id)).fetchone()
        if owner is None:
            return conn.execute(
                "SELECT * FROM documents WHERE doc_hash=?", (dh,)
            ).fetchone()
        return conn.execute(
            "SELECT * FROM documents WHERE doc_hash=? AND owner=?", (dh, owner)
        ).fetchone()
    finally:
        conn.close()


def create(dh, owner, title, source_type, char_count, meta=None, *, owner_id=None) -> int:
    conn = get_db()
    try:
        if owner_id is None and owner:
            # Compatibility for trusted internal callers creating NEW records.
            # Authenticated routes always supply the immutable ID explicitly.
            user = conn.execute("SELECT id FROM users WHERE username=?", (owner,)).fetchone()
            owner_id = user["id"] if user else None
        cur = conn.execute(
            "INSERT INTO documents (doc_hash, owner, title, source_type, char_count, status, meta_json, owner_user_id) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?) RETURNING id",
            (dh, owner, title, source_type, char_count, json.dumps(meta or {}), owner_id),
        )
        document_id = cur.fetchone()["id"]
        conn.commit()
        return document_id
    finally:
        conn.close()


def set_status(document_id, status, chunk_count=None) -> None:
    conn = get_db()
    try:
        if chunk_count is None:
            conn.execute("UPDATE documents SET status=? WHERE id=?", (status, document_id))
        else:
            conn.execute("UPDATE documents SET status=?, chunk_count=? WHERE id=?",
                         (status, chunk_count, document_id))
        conn.commit()
    finally:
        conn.close()


def get(document_id):
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    finally:
        conn.close()


def get_chunks(document_id) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT chunk_index, content, char_start, char_end FROM chunks "
            "WHERE document_id=? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_by_owner(owner: str = "", *, owner_id: int | None = None) -> list[dict]:
    """All documents uploaded by this user, newest first."""
    if not owner and owner_id is None:
        return []
    conn = get_db()
    try:
        clause, value = ("owner_user_id", owner_id) if owner_id is not None else ("owner", owner)
        rows = conn.execute(
            "SELECT id, title, source_type, char_count, chunk_count, status, created_at, meta_json "
            f"FROM documents WHERE {clause}=? ORDER BY created_at DESC",
            (value,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete(document_id, owner: str = "", *, owner_id: int | None = None) -> bool:
    """Delete a document and its chunks, scoped to the owner. Returns True if a
    row was removed. Embeddings are content-hash keyed and shared/cached, so we
    leave them (harmless orphans, reused on re-ingest)."""
    conn = get_db()
    try:
        row = conn.execute("SELECT owner, owner_user_id FROM documents WHERE id=?", (document_id,)).fetchone()
        allowed = row is not None and (row["owner_user_id"] == owner_id if owner_id is not None else row["owner"] == owner)
        if not allowed:
            return False
        conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        conn.commit()
        return True
    finally:
        conn.close()
