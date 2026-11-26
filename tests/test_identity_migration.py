"""Identity upgrades preserve ownership and retire unsafe authentication paths."""
import base64
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner
from starlette.middleware.sessions import SessionMiddleware

import config
from core.services.account_identity import SESSION_VERSION
from fastapi_app.deps import CurrentApiUser


def test_legacy_duplicates_are_preserved_for_recovery_without_claiming_email(tmp_path):
    from fastapi_app.database import init_db

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, "
                     "role TEXT, email TEXT, google_id TEXT, github_id TEXT)")
        conn.executemany("INSERT INTO users VALUES (?, ?, ?, 'user', ?, ?, NULL)", [
            (1, "first", "linked-local-password", " Learner@example.com ", "google-test-id"),
            (2, "second", "their-own-password", "learner@EXAMPLE.com", None),
            (3, "third", "unchanged-password", "unrelated@example.com", None),
        ])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "DB_PATH", str(db_path))
        patch.setattr(config, "DATABASE_URL", "")
        init_db()
        init_db()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id, username, email, email_verified, password_hash FROM users ORDER BY id").fetchall() == [
            (1, "first", None, 0, "!"),
            (2, "second", None, 0, "their-own-password"),
            (3, "third", "unrelated@example.com", 0, "unchanged-password"),
        ]
        assert conn.execute("SELECT user_id FROM identity_email_conflicts ORDER BY user_id").fetchall() == [(1,), (2,)]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE users SET email=? WHERE id=2", (" UNRELATED@example.com ",))


def test_old_cookies_are_rejected_and_current_cookies_use_the_database_name(tmp_path, monkeypatch):
    filename = tmp_path / "sessions.db"
    with sqlite3.connect(filename) as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, session_epoch INTEGER NOT NULL DEFAULT 0)")
        conn.execute("INSERT INTO users(id,username) VALUES (1, 'new-name'), (2, 'old-name')")
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "DB_PATH", str(Path(filename)))

    secret = "isolated-session-test"
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=secret)

    @app.get("/api/identity")
    def identity(user: CurrentApiUser):
        return {"id": user.id, "username": user.username}

    def install_cookie(client, payload):
        cookie = TimestampSigner(secret).sign(base64.b64encode(json.dumps(payload).encode())).decode()
        client.cookies.set("session", cookie, domain="testserver.local", path="/")

    with TestClient(app) as client:
        old_session = {"user_id": 1, "username": "old-name"}
        install_cookie(client, old_session)
        assert client.get("/api/identity").status_code == 401
        install_cookie(client, {**old_session, "identity_version": SESSION_VERSION})
        assert client.get("/api/identity").json() == {"id": 1, "username": "new-name"}
