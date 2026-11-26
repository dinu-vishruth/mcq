"""Regression tests for recovery, privacy, evidence coverage and worker fencing."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from werkzeug.security import check_password_hash

import config
from core.models.db import get_db
from core.services.account_recovery import consume_token, issue_token


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "isolated.db"))
    monkeypatch.setattr(config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    from fastapi_app.database import init_db
    init_db()
    conn = get_db()
    conn.execute("INSERT INTO users(username,password_hash,role,email,email_verified) VALUES ('owner','old','user','owner@example.com',1)")
    conn.commit()
    yield conn
    conn.close()


def user(conn):
    return conn.execute("SELECT * FROM users WHERE username='owner'").fetchone()


def test_reset_is_single_use_and_revokes_existing_sessions(database):
    row = user(database)
    token = issue_token(database, row, "reset")
    assert consume_token(database, token, "Cedar mountain lantern 42!")
    updated = user(database)
    assert updated["session_epoch"] == 1
    assert check_password_hash(updated["password_hash"], "Cedar mountain lantern 42!")
    assert not consume_token(database, token, "another-password")


def test_reset_cannot_use_unverified_email_or_a_changed_contact(database):
    row = user(database)
    token = issue_token(database, row, "reset")
    database.execute("UPDATE users SET email_verified=0 WHERE id=?", (row["id"],))
    database.commit()
    assert not consume_token(database, token, "some-password")
    database.execute("UPDATE users SET email_verified=1,email='new@example.com' WHERE id=?", (row["id"],))
    database.commit()
    assert not consume_token(database, token, "some-password")


def test_deletion_cancels_jobs_preserves_other_owner_and_retries_objects(database, monkeypatch):
    from core.services.account_deletion import delete_account_data, process_deletions
    row = user(database)
    database.execute("INSERT INTO documents(id,owner,title) VALUES (100,'owner','secret'),(101,'other','keep')")
    database.execute("INSERT INTO chunks(document_id,chunk_index,content,content_hash) VALUES "
                     "(100,0,'secret','private-hash'),(100,1,'shared','shared-hash'),(101,0,'shared','shared-hash')")
    database.execute("INSERT INTO embeddings(content_hash,model,dim,vector) VALUES "
                     "('private-hash','mock',1,?),('shared-hash','mock',1,?)", (b'1234', b'1234'))
    database.execute("INSERT INTO generation_jobs(id,user_id,username,status) VALUES ('private-job',?,'owner','running')", (row["id"],))
    database.execute("INSERT INTO upload_objects VALUES ('owner/file.pdf',?)", (row["id"],))
    database.commit()
    task = delete_account_data(database, row["id"])
    assert database.execute("SELECT 1 FROM generation_jobs WHERE id='private-job'").fetchone() is None
    assert database.execute("SELECT title FROM documents WHERE id=101").fetchone()[0] == "keep"
    assert database.execute("SELECT 1 FROM embeddings WHERE content_hash='private-hash'").fetchone() is None
    assert database.execute("SELECT 1 FROM embeddings WHERE content_hash='shared-hash'").fetchone()
    store = Mock(keys=Mock(return_value=[]), delete=Mock(side_effect=RuntimeError("outage")))
    vectors = Mock()
    monkeypatch.setattr("core.storage.get_upload_store", lambda: store)
    monkeypatch.setattr("core.vectorstore.get_vector_store", lambda: vectors)
    process_deletions(task)
    assert database.execute("SELECT 1 FROM deletion_tasks WHERE id=?", (task,)).fetchone()
    store.delete.side_effect = None
    process_deletions(task)
    assert database.execute("SELECT 1 FROM deletion_tasks WHERE id=?", (task,)).fetchone() is None
    vectors.delete_document.assert_called_with(100)


def test_context_keeps_the_last_section_and_obeys_exact_budget():
    from core.agents.context_builder import ContextBuilderAgent
    hits = [SimpleNamespace(chunk_index=i, content=f"section-{i} " + "evidence " * 200) for i in range(4)]
    packed = ContextBuilderAgent().run(hits, max_chars=800)
    assert "section-0" in packed and "section-3" in packed
    assert len(packed) <= 800


def test_fenced_job_cannot_publish_and_session_commit_is_atomic(database):
    from core.repositories import job_repo
    from core.services.generation_metrics import GenerationMetrics
    row = user(database)
    database.execute("INSERT INTO documents(id,owner,title) VALUES (200,'owner','source')")
    database.commit()
    jid = job_repo.create(user_id=row["id"], username="owner", document_id=200, params={}, total=1)
    assert job_repo.claim(jid, "worker-1")
    job = job_repo.get(jid)
    with pytest.raises(ValueError):
        job_repo.publish(job, "wrong-worker", [], "easy", 60, GenerationMetrics())
    assert database.execute("SELECT count(*) FROM sessions WHERE document_id=200").fetchone()[0] == 0
    key = job_repo.publish(job, "worker-1", [{"question": "stored"}], "easy", 60, GenerationMetrics())
    stored = job_repo.get(jid)
    assert stored["status"] == "done"
    assert json.loads(stored["result_json"])["session_key"] == key
    assert database.execute("SELECT 1 FROM sessions WHERE session_key=?", (key,)).fetchone()
    with pytest.raises(ValueError):
        job_repo.publish(job, "worker-1", [], "easy", 60, GenerationMetrics())


def test_dead_worker_is_requeued_at_most_once(database):
    from core.repositories import job_repo
    row = user(database)
    jid = job_repo.create(user_id=row["id"], username="owner", document_id=1, params={}, total=1)
    assert job_repo.claim(jid, "dead-1")
    old = (datetime.utcnow() - timedelta(seconds=100)).isoformat()
    database.execute("UPDATE generation_jobs SET heartbeat_at=? WHERE id=?", (old, jid))
    database.commit()
    assert job_repo.get(jid)["status"] == "queued"
    assert job_repo.claim(jid, "dead-2")
    database.execute("UPDATE generation_jobs SET heartbeat_at=? WHERE id=?", (old, jid))
    database.commit()
    assert job_repo.get(jid)["status"] == "failed"


def test_provenance_receipt_binds_document_location():
    from core.services.source_evidence import authentic, issue, source_locations
    from tests.test_mcq_quality_guardrails import SOURCE, mcq
    q = mcq()
    with source_locations([SimpleNamespace(document_id=17,chunk_index=2,content=SOURCE)]):
        q.update(source_evidence=issue(q, SOURCE), verification_status="verified", safety_status="verified")
    assert authentic(q)
    q["source_evidence"]["provenance"]["document_id"] = 18
    assert not authentic(q)


def test_local_storage_cannot_escape_upload_directory(tmp_path):
    from core.storage import LocalUploadStore
    store = LocalUploadStore(str(tmp_path / "uploads"))
    with pytest.raises(ValueError):
        store.delete("../outside.txt")


def test_semantic_duplicate_verdict_is_required_and_applies_across_batches(monkeypatch):
    from core.agents.fact_verification import FactVerificationAgent
    from tests.test_mcq_quality_guardrails import SOURCE, mcq
    verdict = {"index": 0, "passed": True, "safe": True, "confidence": .99, "semantic_duplicate": True}
    llm = Mock(complete_json=Mock(return_value={"results": [verdict]}))
    monkeypatch.setattr("core.agents.fact_verification.get_llm", lambda: llm)
    result = FactVerificationAgent().run([mcq("What is filtered by WHERE?")], SOURCE, excluded=[mcq()])
    assert not result.verified
    assert result.rejected[0]["reason"] == "semantic duplicate"
    assert "What does WHERE filter?" in llm.complete_json.call_args.args[0][1]["content"]
    for bad in ["false", None]:
        if bad is None:
            del verdict["semantic_duplicate"]
        else:
            verdict["semantic_duplicate"] = bad
        result = FactVerificationAgent().run([mcq()], SOURCE)
        assert not result.verified and len(result.unverified) == 1


def test_explicit_link_never_merges_an_existing_provider_account(database):
    import time

    from core.services.account_identity import link_provider
    row = user(database)
    database.execute("INSERT INTO users(username,password_hash,role,google_id) VALUES('other','hash','user','existing-sub')")
    database.commit()
    request = SimpleNamespace(session={"user_id": row["id"], "_google_link_target": {"user_id": row["id"], "expires": time.time()+60}})
    with pytest.raises(ValueError, match="already belongs"):
        link_provider(database, request, "google", "existing-sub", row["email"])
    database.rollback()
    assert user(database)["google_id"] is None
    request.session["_google_link_target"] = {"user_id": row["id"], "expires": time.time()+60}
    linked = link_provider(database, request, "google", "new-sub", row["email"])
    assert linked["id"] == row["id"] and linked["google_id"] == "new-sub" and linked["password_hash"] == "!"


def test_source_endpoint_checks_owner_and_document_revision(database):
    import hashlib

    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware

    from core.services.account_identity import establish_session
    from fastapi_app.routers import student
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="isolated-sources")
    app.include_router(student.router)
    login_user = dict(user(database))

    @app.get("/_login")
    def login(request: Request):
        establish_session(request, login_user)
        return {}

    database.execute("INSERT INTO documents(id,owner,title) VALUES (1,'owner','Owned'),(2,'other','Private')")
    database.execute("INSERT INTO chunks(document_id,chunk_index,content,content_hash) VALUES (1,0,'Evidence','one'),(2,0,'Private','two')")
    database.commit()
    with TestClient(app) as client:
        client.get("/_login")
        version = hashlib.sha256(b"Evidence").hexdigest()
        assert client.get(f"/sources/1/chunks/0?version={version}").status_code == 200
        assert client.get("/sources/1/chunks/0?version=old-version").status_code == 409
        assert client.get("/sources/2/chunks/0").status_code == 404


def test_readiness_distinguishes_database_outage(database, monkeypatch):
    from fastapi.testclient import TestClient

    from fastapi_app.main import create_app
    app = create_app()
    with TestClient(app) as client:
        with monkeypatch.context() as failed:
            failed.setattr("core.models.db.get_db", Mock(side_effect=RuntimeError("private-database-details")))
            response = client.get("/readyz")
            assert response.status_code == 503
            assert "private-database-details" not in response.text
            assert client.get("/health").status_code == 200
