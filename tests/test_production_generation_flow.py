"""Current upload -> configure -> generate contract, without Flask-era assumptions."""
import json
from unittest.mock import Mock

import pytest

import config
from core.models.db import get_db
from tests.production_support import QUESTION


def _upload(client, **extra):
    return client.post("/upload", data={"title": "Quiz source", "extracted_text":
        "Rows are filtered by WHERE. Groups are filtered by HAVING. " * 20, **extra})


def test_upload_defers_generation_and_ignores_stale_timer_fields(production_app):
    client, _ = production_app.client()
    response = _upload(client, timer="999999", num_questions="2", difficulty="hard")
    assert response.status_code == 302
    assert response.headers["Location"].startswith("/knowledge?doc=")
    assert client.get("/api/knowledge").json()["items"][0]["indexed"] is True
    conn = get_db()
    try:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        conn.close()


def test_practice_generation_returns_playable_session_with_requested_timer(production_app, monkeypatch):
    client, _ = production_app.client()
    document_id = int(_upload(client).headers["Location"].split("doc=")[1])
    monkeypatch.setattr(config, "GENERATION_BACKEND", "inline")
    pipeline = Mock(return_value=[QUESTION])
    monkeypatch.setattr("core.services.mcq_pipeline.generate_from_document", pipeline)
    response = client.post("/api/practice/generate", json={
        "document_id": document_id, "num_questions": 1, "difficulty": "hard", "timer": 300,
    })
    assert response.status_code == 200
    assert set(response.json()) == {"ok", "session_key", "count"}
    assert response.json()["count"] == 1
    pipeline.assert_called_once_with(document_id, num_questions=1, difficulty="hard", topic=None)
    key = response.json()["session_key"]
    conn = get_db()
    try:
        row = conn.execute("SELECT timer, mcqs_json FROM sessions WHERE session_key=?", (key,)).fetchone()
        assert row["timer"] == 300
        assert json.loads(row["mcqs_json"]) == [QUESTION]
    finally:
        conn.close()
    assert client.post("/student_login", data={"session_key": key}).status_code == 302
    assert client.get("/mcq_test").status_code == 200


@pytest.mark.parametrize("timer", [0, 10801])
def test_actual_generation_rejects_out_of_range_timer(production_app, timer):
    client, _ = production_app.client()
    response = client.post("/api/practice/generate", json={"document_id": 1, "timer": timer})
    assert response.status_code == 422
    assert any("timer" in entry["loc"] for entry in response.json()["detail"])
