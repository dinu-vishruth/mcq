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


def find_by_hash(dh: str):
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM documents WHERE doc_hash=?", (dh,)).fetchone()
    finally:
        conn.close()


def create(dh, owner, title, source_type, char_count, meta=None) -> int:
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO documents (doc_hash, owner, title, source_type, char_count, status, meta_json) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (dh, owner, title, source_type, char_count, json.dumps(meta or {})),
        )
        conn.commit()
        return cur.lastrowid
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


def list_by_owner(owner: str) -> list[dict]:
    """All documents uploaded by this user, newest first."""
    if not owner:
        return []
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, source_type, char_count, chunk_count, status, created_at, meta_json "
            "FROM documents WHERE owner=? ORDER BY created_at DESC",
            (owner,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete(document_id, owner: str) -> bool:
    """Delete a document and its chunks, scoped to the owner. Returns True if a
    row was removed. Embeddings are content-hash keyed and shared/cached, so we
    leave them (harmless orphans, reused on re-ingest)."""
    conn = get_db()
    try:
        row = conn.execute("SELECT owner FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None or row["owner"] != owner:
            return False
        conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        conn.commit()
        return True
    finally:
        conn.close()
