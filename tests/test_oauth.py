"""OAuth browser flows with signed cookies, a temporary DB and mocked providers."""
import sqlite3
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
        github_id TEXT UNIQUE, picture TEXT, auth_provider TEXT)""")
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
    assert "invalid state" in response.text
    fetch.assert_not_awaited()
    assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 0


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


@pytest.mark.parametrize("verified", [True, False])
def test_github_links_only_verified_email(browser, monkeypatch, verified):
    _, connection = browser
    connection.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                       ("existing", "existing@example.com", "hash"))
    connection.commit()

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
    assert user["id"] == (1 if verified else 2)
    assert profile["email"] == ("existing@example.com" if verified else "")
