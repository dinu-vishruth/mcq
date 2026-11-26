"""Owner authorization, revocation, application controls, and activity privacy."""
import time
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from werkzeug.security import generate_password_hash

import config
from core.models.db import get_db
from core.services.admin_control import update_control
from tests._client import make_client_factory

PASSWORD = "Cedar river orbit 42!"
OWNER_EMAIL = "owner@example.com"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "admin.db"))
    monkeypatch.setattr(config, "ADMIN_OWNER_EMAIL", OWNER_EMAIL)
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")
    monkeypatch.setattr(config, "VECTOR_STORE", "sqlite")
    from fastapi_app.main import create_app
    app = create_app()
    conn = get_db()
    conn.execute("INSERT INTO users(username,password_hash,role,email,email_verified) VALUES ('owner',?,'user',?,1),('learner',?,'user','learner@example.com',0)",
                 (generate_password_hash(PASSWORD), OWNER_EMAIL, generate_password_hash(PASSWORD)))
    conn.commit()
    factory = make_client_factory(app, config.SECRET_KEY)
    try:
        yield app, conn, factory
    finally:
        conn.close()


def sign_in(factory, username):
    client = factory()
    assert client.post("/login", data={"username": username, "password": PASSWORD}).status_code == 302
    return client


def test_only_the_verified_owner_can_view_or_change_the_admin_dashboard(workspace):
    _, conn, factory = workspace
    assert factory().get("/admin").status_code == 302
    learner = sign_in(factory, "learner")
    assert learner.get("/admin").status_code == 403
    assert learner.post("/admin/controls/demo", data={"value": 0}).status_code == 403
    conn.execute("UPDATE users SET email_verified=0 WHERE username='owner'")
    conn.commit()
    owner = sign_in(factory, "owner")
    assert owner.get("/admin").status_code == 403
    conn.execute("UPDATE users SET email_verified=1 WHERE username='owner'")
    conn.commit()
    response = owner.get("/admin")
    assert response.status_code == 200
    assert b"Your application, at a glance" in response.data
    assert b"learner@example.com" in response.data
    assert PASSWORD.encode() not in response.data and b"scrypt:" not in response.data
    assert owner.post("/admin/controls/demo", data={"value": 0}).status_code == 303
    assert conn.execute("SELECT enabled FROM app_controls WHERE name='demo'").fetchone()[0] == 0


def test_admin_mutations_require_csrf_and_a_recent_sign_in(workspace, monkeypatch):
    _, conn, factory = workspace
    owner = sign_in(factory, "owner")
    assert owner._client.post("/admin/controls/demo", data={"value": 0}).status_code == 400
    from types import SimpleNamespace

    from fastapi_app.routers import admin
    advanced = time.time() + 901
    monkeypatch.setattr(admin, "time", SimpleNamespace(time=lambda: advanced))
    assert owner.get("/admin").status_code == 200
    assert owner.post("/admin/controls/demo", data={"value": 0}).status_code == 403
    assert conn.execute("SELECT enabled FROM app_controls WHERE name='demo'").fetchone()[0] == 1


def test_disable_revokes_open_sessions_blocks_password_login_and_preserves_data(workspace):
    _, conn, factory = workspace
    owner = sign_in(factory, "owner")
    learner = sign_in(factory, "learner")
    uid = conn.execute("SELECT id FROM users WHERE username='learner'").fetchone()[0]
    conn.execute("INSERT INTO documents(owner,title) VALUES ('learner','Keep this material')")
    conn.execute("INSERT INTO generation_jobs(id,user_id,username,status,run_token) VALUES ('queued',?,'learner','running','active-worker')", (uid,))
    conn.commit()
    assert owner.post(f"/admin/accounts/{uid}/disable").status_code == 303
    assert learner.get("/api/prefs").status_code == 401
    assert b"Invalid username or password" in factory().post("/login", data={"username": "learner", "password": PASSWORD}).data
    assert conn.execute("SELECT title FROM documents WHERE owner='learner'").fetchone()[0] == "Keep this material"
    assert tuple(conn.execute("SELECT status,run_token FROM generation_jobs WHERE id='queued'").fetchone()) == ('failed', None)
    assert owner.post(f"/admin/accounts/{uid}/enable").status_code == 303
    assert learner.get("/api/prefs").status_code == 401
    assert sign_in(factory, "learner").get("/api/prefs").status_code == 200


def test_owner_is_protected_and_can_revoke_sessions_without_disabling(workspace):
    _, conn, factory = workspace
    owner = sign_in(factory, "owner")
    uid = conn.execute("SELECT id FROM users WHERE username='learner'").fetchone()[0]
    owner_id = conn.execute("SELECT id FROM users WHERE username='owner'").fetchone()[0]
    learner = sign_in(factory, "learner")
    assert owner.post(f"/admin/accounts/{owner_id}/disable").status_code == 400
    assert owner.post(f"/admin/accounts/{uid}/revoke").status_code == 303
    assert learner.get("/api/prefs").status_code == 401
    assert conn.execute("SELECT is_disabled FROM users WHERE id=?", (uid,)).fetchone()[0] == 0
    assert conn.execute("SELECT action,target FROM admin_audit").fetchone()["action"] == "revoke"


def test_controls_are_enforced_by_signup_demo_and_generation(workspace):
    _, conn, factory = workspace
    owner = sign_in(factory, "owner")
    learner = sign_in(factory, "learner")
    for control in ("registration", "demo", "generation"):
        assert owner.post(f"/admin/controls/{control}", data={"value": 0}).status_code == 303
    assert b"signup is currently paused" in factory().post("/signup", data={"username": "new", "email": "new@example.com", "password": PASSWORD}).data
    assert b"demo is currently paused" in factory().post("/demo").data
    from fastapi import HTTPException

    from core.services.generation_status import generation_available, require_generation
    assert not generation_available()
    with pytest.raises(HTTPException, match="paused"):
        require_generation()
    from core.services.generation_admission import GenerationBusy, admission
    with pytest.raises(GenerationBusy, match="paused"):
        with admission("learner", seconds=60):
            pytest.fail("Paused generation must not be admitted")
    assert learner.get("/api/prefs").status_code == 200
    assert len(conn.execute("SELECT * FROM admin_audit").fetchall()) == 3


def test_owner_search_is_paginated_and_escapes_user_content(workspace):
    _, conn, factory = workspace
    conn.execute("INSERT INTO users(username,password_hash,role,email) VALUES ('<script>alert(1)</script>','x','user','x@example.com')")
    conn.commit()
    owner = sign_in(factory, "owner")
    response = owner.get("/admin?q=%3Cscript%3E")
    assert response.status_code == 200
    assert b"&lt;script&gt;alert(1)&lt;/script&gt;" in response.data
    assert b"<script>alert(1)</script>" not in response.data
    assert b"No accounts match" in owner.get("/admin?q=%25").data
    assert owner.get("/admin?page=0").status_code == 422


def test_access_activity_is_recorded_and_expired_activity_is_removed(workspace):
    _, conn, factory = workspace
    conn.execute("INSERT INTO access_events(username,event,occurred_at) VALUES ('expired','sign_in',?)", (time.time() - 31 * 86400,))
    conn.commit()
    learner = sign_in(factory, "learner")
    row = conn.execute("SELECT * FROM access_events WHERE username='learner'").fetchone()
    assert row["event"] == "sign_in" and row["ip"]
    assert not conn.execute("SELECT 1 FROM access_events WHERE username='expired'").fetchone()
    assert learner.get("/api/prefs").status_code == 200
    assert conn.execute("SELECT last_seen_at FROM users WHERE username='learner'").fetchone()[0] > 0


def test_disabled_oauth_account_cannot_obtain_a_session(workspace, monkeypatch):
    app, conn, _ = workspace
    from fastapi_app.routers import google_auth
    monkeypatch.setattr(config, "GOOGLE_OAUTH_ENABLED", True)
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "test-client")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(config, "GOOGLE_REDIRECT_URI", "")
    conn.execute("UPDATE users SET google_id='disabled-subject',auth_provider='google',is_disabled=1 WHERE username='learner'")
    conn.commit()
    monkeypatch.setattr(google_auth, "_fetch_google_profile", AsyncMock(return_value={"sub": "disabled-subject", "email": "learner@example.com", "email_verified": True}))
    with TestClient(app, follow_redirects=False) as client:
        response = client.get("/auth/google")
        state = parse_qs(urlsplit(response.headers["location"]).query)["state"][0]
        assert client.get("/auth/google/callback", params={"state": state, "code": "test"}).status_code == 403
        assert client.get("/api/prefs").status_code == 401


def test_paused_registration_does_not_allow_new_oauth_accounts(workspace):
    _, conn, _ = workspace
    owner_id = conn.execute("SELECT id FROM users WHERE username='owner'").fetchone()[0]
    update_control(conn, owner_id, "registration", False)
    from fastapi_app.routers.google_auth import _upsert_google_user
    with pytest.raises(ValueError, match="paused"):
        _upsert_google_user(conn, {"sub": "new-subject", "email": "new@example.com", "email_verified": True})
    conn.rollback()
    assert not conn.execute("SELECT 1 FROM users WHERE google_id='new-subject'").fetchone()
