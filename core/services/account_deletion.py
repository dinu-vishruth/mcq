"""Transactional account erasure with retryable external-object cleanup."""
import json
import uuid

from core.models.db import get_db
from core.services.account_identity import identity_write_lock


def delete_account_data(conn, user_id):
    identity_write_lock(conn)
    row = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None:
        conn.rollback()
        return None
    username = row["username"]
    documents = [r["id"] for r in conn.execute("SELECT id FROM documents WHERE owner=?", (username,)).fetchall()]
    hashes = [r["content_hash"] for r in conn.execute("SELECT c.content_hash FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.owner=?", (username,)).fetchall()]
    keys = [r["storage_key"] for r in conn.execute("SELECT storage_key FROM upload_objects WHERE user_id=?", (user_id,)).fetchall()]
    task_id = uuid.uuid4().hex
    prefixes = [r["prefix"] for r in conn.execute("SELECT prefix FROM upload_prefixes WHERE user_id=?", (user_id,)).fetchall()]
    prefixes = set(prefixes + [username + "/"])
    # Legacy usernames could contain slashes. Never enumerate a parent prefix
    # that also contains another existing account's originals.
    prefixes = [prefix for prefix in prefixes if not conn.execute(
        "SELECT 1 FROM users WHERE id<>? AND substr(username,1,?)=?", (user_id, len(prefix), prefix)).fetchone()]
    conn.execute("INSERT INTO deletion_tasks VALUES (?,?,?)", (task_id, json.dumps(documents),
                 json.dumps({"keys": keys, "prefixes": prefixes})))
    conn.execute("INSERT INTO retired_usernames(name) VALUES(lower(?)) ON CONFLICT(name) DO NOTHING", (username,))
    conn.execute("DELETE FROM results WHERE session_key IN (SELECT session_key FROM sessions WHERE teacher=?)", (username,))
    conn.execute("DELETE FROM quiz_attempts WHERE user_id=? OR session_key IN (SELECT session_key FROM sessions WHERE teacher=?)", (user_id, username))
    conn.execute("DELETE FROM sessions WHERE teacher=?", (username,))
    for document in documents:
        conn.execute("DELETE FROM chunks WHERE document_id=?", (document,))
        conn.execute("DELETE FROM chat_history WHERE document_id=?", (document,))
        conn.execute("DELETE FROM documents WHERE id=?", (document,))
    # Cached vectors are shared by content hash. Remove only unreferenced ones.
    for content_hash in set(hashes):
        conn.execute("DELETE FROM embeddings WHERE content_hash=? AND NOT EXISTS "
                     "(SELECT 1 FROM chunks WHERE chunks.content_hash=embeddings.content_hash)", (content_hash,))
    for table in ("results", "learning_history", "weak_topics", "user_prefs", "quiz_uploads", "identity_email_conflicts",
                  "account_tokens", "upload_objects", "upload_prefixes", "chat_history", "generation_jobs"):
        conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM generation_admissions WHERE username=?", (username,))
    conn.execute("DELETE FROM generation_leases WHERE scope=?", ("user:" + username,))
    conn.execute("DELETE FROM login_attempts WHERE username=?", (username,))
    conn.execute("DELETE FROM access_events WHERE user_id=? OR username=?", (user_id, username))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    return task_id


def process_deletions(task_id=None):
    from core.storage import get_upload_store
    from core.vectorstore import get_vector_store
    conn = get_db()
    try:
        tasks = conn.execute("SELECT * FROM deletion_tasks" + (" WHERE id=?" if task_id else " LIMIT 100"),
                             (task_id,) if task_id else ()).fetchall()
        for task in tasks:
            try:
                for document in json.loads(task["document_ids"]):
                    get_vector_store().delete_document(document)
                uploads = json.loads(task["storage_keys"])
                store = get_upload_store()
                keys = set(uploads["keys"])
                for prefix in uploads["prefixes"]:
                    keys.update(store.keys(prefix))
                for key in keys:
                    store.delete(key)
            except Exception:
                continue  # Durable task remains until a subsequent maintenance pass.
            conn.execute("DELETE FROM deletion_tasks WHERE id=?", (task["id"],))
            conn.commit()
    finally:
        conn.close()
