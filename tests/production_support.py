"""Isolated offline app setup shared by production regression tests."""
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from werkzeug.security import generate_password_hash

import config
from core.models.db import get_db

TEST_PASSWORD = "Example-Only-482!"
QUESTION = {
    "question": "Which option is correct?",
    "options": [{"label": label, "text": value} for label, value in
                zip("ABCD", ["option-one", "option-two", "option-three", "option-four"])],
    "answer_text": "option-two",
}


@dataclass
class ProductionApp:
    app: object

    def client(self, username="alice"):
        from tests._client import FlaskLikeClient
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')",
                (username, generate_password_hash(TEST_PASSWORD)),
            )
            user_id = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()[0]
            conn.commit()
        finally:
            conn.close()
        client = FlaskLikeClient(self.app, config.SECRET_KEY)
        client.post("/login", data={"username": username, "password": TEST_PASSWORD})
        return client, user_id

    def session(self, username="alice", key="prodquiz"):
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, mcqs_json) "
                "VALUES (?, ?, ?, 'easy', 300, ?)",
                (key, username, datetime.now(timezone.utc).isoformat(), json.dumps([QUESTION])),
            )
            conn.commit()
            return key
        finally:
            conn.close()


@pytest.fixture
def production_app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "production.db"))
    monkeypatch.setattr(config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")
    monkeypatch.setattr(config, "VECTOR_STORE", "sqlite")
    monkeypatch.setattr(config, "SEED_DEMO_USERS", False)
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    from core.vectorstore.registry import reset_vector_store
    from fastapi_app.main import create_app
    reset_vector_store()
    app = create_app()
    monkeypatch.setattr("models.explanation_engine.explain_answers", lambda details: [])
    monkeypatch.setattr("core.services.learning_service.analyse_submission", lambda *args, **kwargs: {})
    yield ProductionApp(app)
    reset_vector_store()
