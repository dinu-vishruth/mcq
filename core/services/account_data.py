"""Account-data removal with atomic SQLite rows/vectors and active-store cleanup."""
from core.vectorstore.sqlite_numpy import SqliteNumpyStore


def delete_documents(conn, document_ids):
    if not document_ids:
        return
    from core.vectorstore import get_vector_store
    store = get_vector_store()
    hashes = set()
    for doc_id in document_ids:
        hashes.update(row[0] for row in conn.execute(
            "SELECT DISTINCT content_hash FROM chunks WHERE document_id=?", (doc_id,)
        ).fetchall())
    # External stores cannot join SQLite's transaction. Fail before any account
    # row is deleted if their cleanup is unavailable. No distributed atomicity
    # is promised for a later SQL failure after an external deletion succeeds.
    if not isinstance(store, SqliteNumpyStore):
        for doc_id in document_ids:
            store.delete_document(doc_id)
    for doc_id in document_ids:
        if isinstance(store, SqliteNumpyStore):
            store.delete_document(doc_id, connection=conn)
        else:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    for content_hash in hashes:
        conn.execute("DELETE FROM embeddings WHERE content_hash=? AND NOT EXISTS "
                     "(SELECT 1 FROM chunks WHERE content_hash=?)", (content_hash, content_hash))


def delete_account_data(conn, user_id, username):
    document_ids = [row[0] for row in conn.execute(
        "SELECT id FROM documents WHERE owner=?", (username,)
    ).fetchall()]
    try:
        delete_documents(conn, document_ids)
        # Other learners retain their own results, including attempts at a quiz
        # whose creator is leaving. The results table has no session foreign key.
        for table in ("results", "learning_history", "weak_topics", "user_prefs",
                      "chat_history", "generation_jobs"):
            conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM sessions WHERE teacher=?", (username,))
        conn.execute("DELETE FROM login_attempts WHERE username=?", (username,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
