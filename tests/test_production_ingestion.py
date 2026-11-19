"""Serverless ingestion completes before responding and stale work is visible."""
from datetime import datetime, timedelta, timezone

import pytest

from core.models.db import get_db
from core.repositories import document_repo


@pytest.mark.parametrize("status", ["pending", "chunked", "embedded"])
def test_stale_unfinished_document_is_reported_failed(production_app, status):
    client, _ = production_app.client()
    doc_id = document_repo.create("stuck-source", "alice", "Stuck source", "txt", 100)
    old = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
    conn = get_db()
    try:
        conn.execute("UPDATE documents SET status=?, created_at=? WHERE id=?", (status, old, doc_id))
        conn.commit()
    finally:
        conn.close()
    item = client.get("/api/knowledge").json()["items"][0]
    assert item["status"] == "failed"
    assert item["indexed"] is False


@pytest.mark.parametrize("path", ["/upload", "/ingest_resource"])
def test_upload_is_ready_before_response_returns(production_app, path):
    client, _ = production_app.client()
    response = client.post(path, data={"title": "Completed source",
        "extracted_text": "Transactions preserve consistency. Indexes speed lookups. " * 20})
    assert response.status_code == 302
    docs = document_repo.list_by_owner("alice")
    assert len(docs) == 1 and docs[0]["status"] == "ready"
    assert docs[0]["chunk_count"] > 0


@pytest.mark.parametrize("path", ["/upload", "/ingest_resource"])
def test_indexing_failure_marks_document_failed_and_stays_on_form(production_app, monkeypatch, path):
    client, _ = production_app.client()
    def fail(*args, **kwargs):
        raise RuntimeError("test embedding failure")
    monkeypatch.setattr("core.agents.embedding.EmbeddingAgent.run", fail)
    response = client.post(path, data={"title": "Failed source", "extracted_text": "Indexing fails. " * 20})
    assert response.status_code == 200
    assert "Could not save this resource" in response.text
    assert document_repo.list_by_owner("alice")[0]["status"] == "failed"


def test_status_updates_refresh_the_stale_work_clock(production_app):
    client, _ = production_app.client()
    doc_id = document_repo.create("old-source", "alice", "Retried source", "txt", 100)
    conn = get_db()
    try:
        conn.execute("UPDATE documents SET created_at='2000-01-01' WHERE id=?", (doc_id,))
        conn.commit()
    finally:
        conn.close()
    document_repo.set_status(doc_id, "chunked")
    row = document_repo.get(doc_id)
    assert datetime.fromisoformat(row["status_updated_at"]).tzinfo is not None
    assert client.get("/api/knowledge").json()["items"][0]["status"] == "chunked"
