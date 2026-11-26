"""OAuth browser flows with signed cookies, a temporary DB and mocked providers."""
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

import config
from fastapi_app.deps import db
from fastapi_app.routers import auth, github_auth, google_auth
from fastapi_app.templating import templates


@pytest.fixture
def browser(tmp_path, monkeypatch):
    connection = sqlite3.connect(tmp_path / "oauth.db", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE users (
        id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT,
        password_hash TEXT NOT NULL, role TEXT, google_id TEXT UNIQUE,
        github_id TEXT UNIQUE, picture TEXT, auth_provider TEXT,
        email_verified INTEGER NOT NULL DEFAULT 0, session_epoch INTEGER NOT NULL DEFAULT 0)""")
    connection.execute("""CREATE TABLE identity_email_conflicts (
        user_id INTEGER NOT NULL, former_email TEXT NOT NULL,
        PRIMARY KEY(user_id, former_email))""")
    connection.execute("CREATE TABLE retired_usernames (name TEXT PRIMARY KEY)")
    connection.executescript((Path(__file__).resolve().parents[1] / "database/migrations/009_sqlite_admin.sql").read_text())
    for provider in ("GOOGLE", "GITHUB"):
        monkeypatch.setattr(config, provider + "_OAUTH_ENABLED", True)
        monkeypatch.setattr(config, provider + "_CLIENT_ID", "test-client")
        monkeypatch.setattr(config, provider + "_CLIENT_SECRET", "test-secret")
        monkeypatch.setattr(config, provider + "_REDIRECT_URI", "")
        monkeypatch.setitem(templates.env.globals, provider.lower() + "_oauth_enabled", True)
    monkeypatch.setattr(config, "IS_DEPLOYED", True)
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-stable-secret", https_only=True)
    app.include_router(auth.router)
    app.include_router(google_auth.router)
    app.include_router(github_auth.router)
    app.dependency_overrides[db] = lambda: connection

    @app.get("/test-session")
    def session(request: Request):
        return request.session

    with TestClient(app, base_url="https://examly.example", follow_redirects=False) as client:
        yield client, connection
    connection.close()


@pytest.mark.parametrize("provider", ["google", "github"])
def test_new_and_returning_user(browser, monkeypatch, provider):
    client, connection = browser
    module = google_auth if provider == "google" else github_auth
    profile = {"sub": "google-123", "id": 123, "email": "learner@example.com",
               "email_verified": True, "name": "Learner"}
    fetch = AsyncMock(return_value=profile)
    monkeypatch.setattr(module, f"_fetch_{provider}_profile", fetch)
    for _ in range(2):
        response = client.get(f"/auth/{provider}")
        assert response.status_code == 302
        query = parse_qs(urlsplit(response.headers["location"]).query)
        callback = f"https://examly.example/auth/{provider}/callback"
        assert query["redirect_uri"] == [callback]
        # Preserve the exact authorization URI even if configuration changes.
        monkeypatch.setattr(config, provider.upper() + "_REDIRECT_URI", callback + "?changed=1")
        response = client.get(f"/auth/{provider}/callback",
                              params={"code": "test-code", "state": query["state"][0]})
        assert response.status_code == 302
        assert response.headers["location"] == "/dashboard"
        fetch.assert_awaited_with("test-code", callback)
        session = client.get("/test-session").json()
        assert session["user_id"] == 1
        assert f"_{provider}_oauth_state" not in session
        assert f"_{provider}_oauth_state_redirect_uri" not in session
        monkeypatch.setattr(config, provider.upper() + "_REDIRECT_URI", "")
    assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 1


@pytest.mark.parametrize("provider", ["google", "github"])
def test_invalid_state_never_exchanges_code(browser, monkeypatch, provider):
    client, connection = browser
    module = google_auth if provider == "google" else github_auth
    fetch = AsyncMock()
    monkeypatch.setattr(module, f"_fetch_{provider}_profile", fetch)
    client.get(f"/auth/{provider}")
    response = client.get(f"/auth/{provider}/callback?code=test&state=wrong")
    assert ("Start Google sign-in again" if provider == "google" else "invalid state") in response.text
    fetch.assert_not_awaited()
    assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 0


@pytest.mark.parametrize("provider", ["google", "github"])
def test_oauth_email_collision_does_not_sign_in_or_link_an_existing_account(browser, monkeypatch, provider):
    client, connection = browser
    connection.execute("INSERT INTO users (username, email, password_hash, email_verified) VALUES (?, ?, ?, 1)",
                       ("existing", "owner@example.com", "hash"))
    connection.commit()
    module = google_auth if provider == "google" else github_auth
    profile = {"sub": "new-google-subject", "id": 456, "email": "owner@example.com",
               "email_verified": True, "name": "Learner"}
    monkeypatch.setattr(module, f"_fetch_{provider}_profile", AsyncMock(return_value=profile))
    start = client.get(f"/auth/{provider}")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.get(f"/auth/{provider}/callback", params={"code": "case", "state": state})
    assert response.status_code == 200
    assert "another sign-in method" in response.text
    assert not client.get("/test-session").json().get("user_id")
    users = connection.execute("SELECT username, google_id, github_id FROM users").fetchall()
    assert [tuple(user) for user in users] == [("existing", None, None)]


@pytest.mark.parametrize("provider", ["google", "github"])
def test_verified_provider_claims_contact_without_linking_unverified_account(browser, monkeypatch, provider):
    client, connection = browser
    connection.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                       ("unverified", "  OWNER@EXAMPLE.COM  ", "local-password-hash"))
    connection.commit()
    module = google_auth if provider == "google" else github_auth
    profile = {"sub": "new-google-subject", "id": 456, "email": "owner@example.com",
               "email_verified": True, "name": "Learner"}
    monkeypatch.setattr(module, f"_fetch_{provider}_profile", AsyncMock(return_value=profile))
    start = client.get(f"/auth/{provider}")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.get(f"/auth/{provider}/callback", params={"code": "case", "state": state})
    assert response.headers["location"] == "/dashboard"
    assert client.get("/test-session").json()["user_id"] == 2
    old_account = connection.execute("SELECT email, password_hash, google_id, github_id FROM users WHERE id=1").fetchone()
    assert not old_account["email"]
    assert old_account["password_hash"] == "local-password-hash"
    assert old_account["google_id"] is None and old_account["github_id"] is None
    owner = connection.execute("SELECT email, email_verified FROM users WHERE id=2").fetchone()
    assert tuple(owner) == ("owner@example.com", 1)
    assert connection.execute("SELECT user_id FROM identity_email_conflicts").fetchone()[0] == 1


@pytest.mark.parametrize("provider", ["google", "github"])
def test_returning_subject_keeps_own_account_if_provider_email_changes(browser, monkeypatch, provider):
    client, connection = browser
    connection.execute("INSERT INTO users (username, email, password_hash, email_verified) VALUES (?, ?, ?, 1)",
                       ("different-user", "different@example.com", "hash"))
    connection.execute("INSERT INTO users (username, email, password_hash, google_id, github_id) "
                       "VALUES (?, ?, ?, ?, ?)", ("returning", "original@example.com", "hash", "sub-123", "123"))
    connection.commit()
    module = google_auth if provider == "google" else github_auth
    profile = {"sub": "sub-123", "id": 123, "email": "different@example.com",
               "email_verified": True, "name": "Learner"}
    monkeypatch.setattr(module, f"_fetch_{provider}_profile", AsyncMock(return_value=profile))
    start = client.get(f"/auth/{provider}")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.get(f"/auth/{provider}/callback", params={"code": "case", "state": state})
    assert response.headers["location"] == "/dashboard"
    assert client.get("/test-session").json()["user_id"] == 2
    assert connection.execute("SELECT email FROM users WHERE id=2").fetchone()[0] == "original@example.com"


@pytest.mark.parametrize("provider", ["google", "github"])
def test_oauth_does_not_use_an_email_as_username(browser, monkeypatch, provider):
    client, connection = browser
    module = google_auth if provider == "google" else github_auth
    profile = {"sub": "sub-123", "id": 123, "email": "MixedCase@example.com", "email_verified": True,
               "name": "MixedCase@example.com", "login": "MixedCase@example.com"}
    monkeypatch.setattr(module, f"_fetch_{provider}_profile", AsyncMock(return_value=profile))
    start = client.get(f"/auth/{provider}")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.get(f"/auth/{provider}/callback", params={"code": "case", "state": state})
    assert response.headers["location"] == "/dashboard"
    user = connection.execute("SELECT username, email FROM users").fetchone()
    assert "@" not in user["username"]
    assert user["username"].casefold() != "mixedcase"
    assert user["email"] == "mixedcase@example.com"


@pytest.mark.parametrize("provider", ["google", "github"])
def test_start_on_registered_callback_host(browser, monkeypatch, provider):
    client, _ = browser
    monkeypatch.setattr(config, provider.upper() + "_REDIRECT_URI",
                        f"https://production.example/auth/{provider}/callback")
    response = client.get(f"/auth/{provider}")
    assert response.headers["location"] == f"https://production.example/auth/{provider}"
    assert "set-cookie" not in response.headers


def test_signup_buttons_start_both_providers(browser):
    client, _ = browser
    for path in ("/", "/signup"):
        html = client.get(path).text
        assert 'href="/auth/google"' in html
        assert 'href="/auth/github"' in html


@pytest.mark.parametrize(("status", "payload", "reason", "message"), [
    (401, {"error": "invalid_client", "error_description": "private provider details"},
     "invalid_client", "Google sign-in is temporarily unavailable"),
    (400, {"error": "invalid_grant"}, "invalid_grant", "Start Google sign-in again"),
    (503, "private provider details", "unknown", "Google sign-in is temporarily unavailable"),
    (200, ["unexpected JSON"], "invalid_response", "Google sign-in is temporarily unavailable"),
])
def test_google_token_failures_are_diagnosable_without_disclosing_details(
        browser, monkeypatch, caplog, status, payload, reason, message):
    client, connection = browser

    def respond(request):
        assert request.method == "POST"
        assert request.url == google_auth.GOOGLE_TOKEN_ENDPOINT
        form = parse_qs(request.content.decode())
        assert form["redirect_uri"] == ["https://examly.example/auth/google/callback"]
        assert form["grant_type"] == ["authorization_code"]
        if isinstance(payload, str):
            return httpx.Response(status, text=payload)
        return httpx.Response(status, json=payload)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(google_auth.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    start = client.get("/auth/google")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.get(google_auth.CALLBACK_PATH, params={"code": "test-code", "state": state})
    assert message in response.text
    assert "private provider details" not in response.text
    assert "test-secret" not in response.text
    events = [record.fields for record in caplog.records if record.getMessage() == "oauth.google.failed"]
    assert events[-1] == {"stage": "token", "reason": reason, "provider_status": status}
    assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 0


def test_google_denied_consent_shows_feedback_without_exchanging_code(browser, monkeypatch):
    client, _ = browser
    fetch = AsyncMock()
    monkeypatch.setattr(google_auth, "_fetch_google_profile", fetch)
    start = client.get("/auth/google")
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.get(google_auth.CALLBACK_PATH, params={"error": "access_denied", "state": state})
    assert "Google did not grant access" in response.text
    fetch.assert_not_awaited()


@pytest.mark.parametrize("verified", [True, False, "true"])
def test_github_only_stores_verified_email(browser, monkeypatch, verified):
    _, connection = browser

    def respond(request):
        if request.url == github_auth.GITHUB_TOKEN_ENDPOINT:
            return httpx.Response(200, json={"access_token": "test-token"})
        if request.url == github_auth.GITHUB_USER_ENDPOINT:
            return httpx.Response(200, json={"id": 123, "login": "learner",
                                            "email": "existing@example.com"})
        assert request.url == github_auth.GITHUB_EMAILS_ENDPOINT
        return httpx.Response(200, json=[{"email": "existing@example.com",
                                          "primary": True, "verified": verified}])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(github_auth.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    import asyncio

    profile = asyncio.run(github_auth._fetch_github_profile("code", "https://examly.example"))
    user = github_auth._upsert_github_user(connection, profile)
    email = connection.execute("SELECT email FROM users WHERE id=?", (user["id"],)).fetchone()[0]
    assert email == ("existing@example.com" if verified is True else None)
