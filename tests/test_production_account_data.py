"""Deletion/export isolate account data and preserve shared vector references."""
import json
import sqlite3
from unittest.mock import Mock

import pytest

from core.models.db import get_db
from core.repositories import document_repo, job_repo
from core.services.ingestion_service import ingest_document
from core.vectorstore import get_vector_store


def _seed(production_app):
    alice, alice_id = production_app.client()
    bob, bob_id = production_app.client("bob")
    docs = []
    for owner, title, text in (
        ("alice", "Alice private", "raw-source-unique: Atomic writes preserve consistency. " * 10),
        ("alice", "Alice shared", "Shared source about transactions. " * 10),
        ("bob", "Bob shared", "Shared source about transactions. " * 10),
        ("bob", "Bob private", "Bob source about indexes. " * 10),
    ):
        docs.append(ingest_document(text, owner=owner, title=title, source_type="txt")["document_id"])
    production_app.session("alice", "alicequiz")
    production_app.session("bob", "bobquiz")
    conn = get_db()
    try:
        conn.execute("UPDATE sessions SET document_id=? WHERE teacher='alice'", (docs[0],))
        conn.execute("UPDATE sessions SET document_id=? WHERE teacher='bob'", (docs[3],))
        for user_id, doc_id, key in ((alice_id, docs[0], "alicequiz"), (bob_id, docs[3], "bobquiz")):
            conn.execute("INSERT INTO user_prefs (user_id, goal) VALUES (?, 'college')", (user_id,))
            conn.execute("INSERT INTO results (user_id, session_key, score, total) VALUES (?, ?, 1, 1)", (user_id, key))
            conn.execute("INSERT INTO learning_history (user_id, session_key, topic) VALUES (?, ?, 'Transactions')", (user_id, key))
            conn.execute("INSERT INTO weak_topics (user_id, topic) VALUES (?, 'Transactions')", (user_id,))
            conn.execute("INSERT INTO chat_history (user_id, document_id, content) VALUES (?, ?, 'study notes')", (user_id, doc_id))
        # Another learner's result remains theirs even if its quiz owner leaves.
        conn.execute("INSERT INTO results (user_id, session_key, score, total) VALUES (?, 'alicequiz', 1, 1)", (bob_id,))
        conn.commit()
    finally:
        conn.close()
    for user_id, username, doc_id in ((alice_id, "alice", docs[0]), (bob_id, "bob", docs[3])):
        job_repo.create(user_id=user_id, username=username, document_id=doc_id, params={}, total=1)
    return alice, bob, alice_id, bob_id, docs


def test_account_deletion_cleans_all_owned_rows_and_only_unshared_vectors(production_app):
    alice, bob, alice_id, bob_id, docs = _seed(production_app)
    own_hash = document_repo.get_chunks(docs[0])[0]["content"]
    own_hash = document_repo.doc_hash(own_hash)
    shared_hash = document_repo.doc_hash(document_repo.get_chunks(docs[1])[0]["content"])
    response = alice.post("/delete_account")
    assert response.status_code == 302
    conn = get_db()
    try:
        assert conn.execute("SELECT 1 FROM users WHERE id=?", (alice_id,)).fetchone() is None
        for table in ("results", "learning_history", "weak_topics", "user_prefs", "chat_history", "generation_jobs"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id=?", (alice_id,)).fetchone()[0] == 0, table
        assert conn.execute("SELECT COUNT(*) FROM documents WHERE owner='alice'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE teacher='alice'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id IN (?, ?)", docs[:2]).fetchone()[0] == 0
        assert conn.execute("SELECT 1 FROM embeddings WHERE content_hash=?", (own_hash,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM embeddings WHERE content_hash=?", (shared_hash,)).fetchone() is not None
        assert conn.execute("SELECT COUNT(*) FROM results WHERE user_id=?", (bob_id,)).fetchone()[0] == 2
        for table in ("learning_history", "weak_topics", "user_prefs", "chat_history", "generation_jobs"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id=?", (bob_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM login_attempts WHERE username='alice'").fetchone()[0] == 0
    finally:
        conn.close()
    assert get_vector_store().has_document(docs[2])
    assert get_vector_store().has_document(docs[3])
    assert bob.get("/api/knowledge").status_code == 200


def test_export_adds_owned_sessions_and_document_metadata_only(production_app):
    alice, _, _, _, _ = _seed(production_app)
    data = alice.get("/api/export").json()
    assert {"account", "prefs", "results", "weak_topics", "exported_at", "sessions", "documents"} <= data.keys()
    assert len(data["sessions"]) == 1
    session = data["sessions"][0]
    assert {"title", "timer", "difficulty", "questions"} <= session.keys()
    assert session["title"] == "Alice private"
    assert session["questions"][0]["question"] == "Which option is correct?"
    assert {doc["title"] for doc in data["documents"]} == {"Alice private", "Alice shared"}
    assert "raw-source-unique" not in json.dumps(data)
    assert "Bob private" not in json.dumps(data)


def test_sqlite_deletion_failure_rolls_back_rows_and_vectors(production_app):
    alice, _, alice_id, _, docs = _seed(production_app)
    conn = get_db()
    try:
        conn.execute("CREATE TRIGGER fail_delete BEFORE DELETE ON documents BEGIN SELECT RAISE(ABORT, 'injected delete failure'); END")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        alice.post("/delete_account")
    assert document_repo.get(docs[0]) is not None
    assert get_vector_store().has_document(docs[0])
    conn = get_db()
    try:
        assert conn.execute("SELECT 1 FROM users WHERE id=?", (alice_id,)).fetchone() is not None
        assert conn.execute("SELECT COUNT(*) FROM results WHERE user_id=?", (alice_id,)).fetchone()[0] == 1
    finally:
        conn.close()


def test_external_vector_failure_does_not_delete_account_rows(production_app, monkeypatch):
    alice, _, alice_id, _, docs = _seed(production_app)
    store = Mock()
    store.delete_document.side_effect = RuntimeError("vector backend unavailable")
    monkeypatch.setattr("core.vectorstore.get_vector_store", lambda: store)
    with pytest.raises(RuntimeError, match="vector backend unavailable"):
        alice.post("/delete_account")
    assert document_repo.get(docs[0]) is not None
    conn = get_db()
    try:
        assert conn.execute("SELECT 1 FROM users WHERE id=?", (alice_id,)).fetchone() is not None
    finally:
        conn.close()
