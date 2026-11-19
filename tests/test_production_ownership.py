"""Renaming an account preserves ownership and invalidates stale name caches."""
import sqlite3

import pytest

import config
from core.models.db import get_db
from core.repositories import document_repo, job_repo
from tests._client import FlaskLikeClient
from tests.production_support import TEST_PASSWORD


def test_rename_preserves_session_document_and_job_access(production_app):
    client, user_id = production_app.client()
    key = production_app.session()
    doc_id = document_repo.create("rename-source", "alice", "My source", "txt", 100)
    job_id = job_repo.create(user_id=user_id, username="alice", document_id=doc_id, params={}, total=1)
    stale = FlaskLikeClient(production_app.app, config.SECRET_KEY)
    stale.post("/login", data={"username": "alice", "password": TEST_PASSWORD})
    other, _ = production_app.client("bob")
    response = client.post("/profile", data={"username": "renamed", "email": ""})
    assert response.status_code == 302
    assert client.get("/api/knowledge").json()["items"][0]["id"] == doc_id
    assert client.get(f"/session_report/{key}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 200
    assert stale.get(f"/session_report/{key}").status_code == 200
    assert stale.get("/api/knowledge").json()["items"][0]["id"] == doc_id
    assert other.get(f"/session_report/{key}").status_code == 403
    assert other.get(f"/api/jobs/{job_id}").status_code == 404
    assert other.delete(f"/api/knowledge/{doc_id}").status_code == 404
    assert other.post("/api/practice/generate", json={"document_id": doc_id}).status_code == 404
    assert other.get("/api/knowledge").json()["items"] == []
    assert client.delete(f"/api/knowledge/{doc_id}").status_code == 200
    assert client.post(f"/delete_session/{key}").status_code == 302


def test_rename_failure_rolls_back_all_ownership_updates(production_app):
    client, user_id = production_app.client()
    key = production_app.session()
    document_repo.create("rollback-source", "alice", "My source", "txt", 100)
    conn = get_db()
    try:
        conn.execute("CREATE TRIGGER fail_rename BEFORE UPDATE OF owner ON documents "
                     "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        client.post("/profile", data={"username": "renamed"})
    conn = get_db()
    try:
        assert conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()[0] == "alice"
        assert conn.execute("SELECT teacher FROM sessions WHERE session_key=?", (key,)).fetchone()[0] == "alice"
        assert conn.execute("SELECT owner FROM documents").fetchone()[0] == "alice"
    finally:
        conn.close()
