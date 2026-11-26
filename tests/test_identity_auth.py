"""Password sign-in and profile identity checks in a temporary SQLite app."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import generate_password_hash

import config
from core.models.db import open_connection
from core.services.account_identity import establish_session
from fastapi_app.database import init_db
from fastapi_app.deps import Db
from fastapi_app.routers import auth

PASSWORD = "identity-test-password"


@pytest.fixture
def browser(tmp_path, monkeypatch):
    db_path = str(tmp_path / "identity.db")
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "DB_PATH", db_path)
    init_db()

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="isolated-identity-test")
    app.include_router(auth.router)

    @app.post("/_test/provider-session/{user_id}")
    async def provider_session(request: Request, user_id: int, conn: Db):
        user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        establish_session(request, user)
        return {"ok": True}

    conn = open_connection(db_path)
    try:
        with TestClient(app, follow_redirects=False) as client:
            yield client, conn
    finally:
        conn.close()


def add_user(conn, username, email=None, *, google_id=None, github_id=None):
    user_id = conn.execute(
        "INSERT INTO users (username, email, password_hash, role, google_id, github_id) "
        "VALUES (?, ?, ?, 'user', ?, ?) RETURNING id",
        (username, email, generate_password_hash(PASSWORD), google_id, github_id),
    ).fetchone()["id"]
    conn.commit()
    return user_id


def login(client, username):
    return client.post("/login", data={"username": username, "password": PASSWORD})


def test_signup_requires_a_non_email_username_and_rejects_case_variants(browser):
    client, conn = browser
    for username in ("friend@example.com", "friend＠example.com"):
        response = client.post("/signup", data={"username": username, "password": PASSWORD, "email": "friend@gmail.com"})
        assert response.status_code == 200
        assert conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone() is None

    assert client.post("/signup", data={"username": "Learner", "password": PASSWORD, "email": "learner@gmail.com"}).status_code == 302
    response = client.post("/signup", data={"username": "LEARNER", "password": PASSWORD, "email": "another@gmail.com"})
    assert "Username already exists" in response.text
    assert conn.execute("SELECT COUNT(*) FROM users WHERE LOWER(username)='learner'").fetchone()[0] == 1


@pytest.mark.parametrize("column", ["google_id", "github_id"])
def test_password_cannot_open_a_previously_linked_account(browser, column):
    client, conn = browser
    add_user(conn, "linked", "owner@example.com", **{column: "provider-identity"})
    response = login(client, "linked")
    assert "Invalid username or password" in response.text
    assert client.get("/").status_code == 200


def test_profile_cannot_claim_an_existing_email_or_rename_to_an_email(browser):
    client, conn = browser
    user_id = add_user(conn, "local")
    add_user(conn, "provider", "owner@example.com", google_id="verified-subject")
    assert login(client, "local").status_code == 302

    response = client.post("/profile", data={"username": "rename", "email": "  OWNER@EXAMPLE.COM "})
    assert "email is already in use" in response.text
    response = client.post("/profile", data={"username": "local@example.com", "email": "valid@example.com"})
    assert response.status_code == 200
    row = conn.execute("SELECT username, email FROM users WHERE id=?", (user_id,)).fetchone()
    assert tuple(row) == ("local", None)


def test_provider_profile_cannot_change_its_email_or_set_a_local_password(browser):
    client, conn = browser
    user_id = add_user(conn, "provider", "owner@example.com", google_id="verified-subject")
    old = conn.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()[0]
    assert client.post(f"/_test/provider-session/{user_id}").status_code == 200

    profile = client.get("/profile")
    assert '"email_locked": true' in profile.text
    assert '"password_enabled": false' in profile.text
    response = client.post("/profile", data={"username": "provider", "email": "elsewhere@example.com"})
    assert "sign-in provider manages this email" in response.text
    response = client.post("/profile", data={"username": "provider", "password": "new-password"})
    assert "Use your sign-in provider to manage your password" in response.text
    row = conn.execute("SELECT email, password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    assert tuple(row) == ("owner@example.com", old)


def test_legacy_email_username_can_sign_in_and_rename_with_its_library_and_quizzes(browser):
    client, conn = browser
    old = "legacy@example.com"
    user_id = add_user(conn, old)
    conn.execute("INSERT INTO documents (owner, title) VALUES (?, 'Study notes')", (old,))
    conn.execute("INSERT INTO sessions (session_key, teacher) VALUES ('test-quiz', ?)", (old,))
    conn.commit()

    signed_in = login(client, old)
    assert signed_in.status_code == 302
    assert signed_in.headers["location"] == "/profile"
    response = client.post("/profile", data={"username": "learner", "email": " Contact@Example.com "})
    assert response.status_code == 302
    row = conn.execute("SELECT username, email, email_verified FROM users WHERE id=?", (user_id,)).fetchone()
    assert tuple(row) == ("learner", "contact@example.com", 0)
    assert conn.execute("SELECT owner FROM documents WHERE title='Study notes'").fetchone()[0] == "learner"
    assert conn.execute("SELECT teacher FROM sessions WHERE session_key='test-quiz'").fetchone()[0] == "learner"
    assert client.get("/profile").status_code == 200

    assert client.post("/profile", data={"username": "learner", "email": ""}).status_code == 302
    assert conn.execute("SELECT email FROM users WHERE id=?", (user_id,)).fetchone()[0] is None


def test_deleted_username_cannot_be_reclaimed(browser):
    client, conn = browser
    user_id = add_user(conn, "DeletedLearner")
    assert login(client, "DeletedLearner").status_code == 302
    assert client.post("/delete_account").status_code == 302
    assert conn.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is None

    response = client.post("/signup", data={"username": "deletedlearner", "password": PASSWORD, "email": "deleted@gmail.com"})
    assert "Username already exists" in response.text


@pytest.mark.parametrize("email", [None, "", "  ", "not-an-email", "learner@gmail", "learner @gmail.com"])
def test_signup_rejects_missing_or_invalid_email(browser, email):
    client, conn = browser
    fields = {"username": "new-learner", "password": PASSWORD}
    if email is not None:
        fields["email"] = email
    response = client.post("/signup", data=fields)
    assert response.status_code == 200
    assert "required" in response.text if not email or not email.strip() else "valid email address" in response.text
    assert conn.execute("SELECT 1 FROM users WHERE username='new-learner'").fetchone() is None


@pytest.mark.parametrize("email", [" Learner@GMAIL.com ", " Learner@Example.com "])
def test_signup_saves_email_without_treating_it_as_verified_and_can_sign_in(browser, email):
    client, conn = browser
    response = client.post("/signup", data={"username": "new-learner", "password": PASSWORD, "email": email})
    assert response.status_code == 302
    row = conn.execute("SELECT email, email_verified FROM users WHERE username='new-learner'").fetchone()
    assert tuple(row) == (email.strip().lower(), 0)
    assert login(client, "new-learner").status_code == 302


def test_signup_rejects_duplicate_email_and_keeps_form_values(browser):
    client, conn = browser
    add_user(conn, "existing", "learner@gmail.com")
    response = client.post("/signup", data={"username": "new-learner", "password": PASSWORD, "email": " LEARNER@GMAIL.COM "})
    assert "email is already in use" in response.text
    assert 'value="new-learner"' in response.text
    assert 'value="learner@gmail.com"' in response.text
    assert PASSWORD not in response.text
    assert conn.execute("SELECT 1 FROM users WHERE username='new-learner'").fetchone() is None


def test_signup_form_requires_an_email(browser):
    client, _ = browser
    from html.parser import HTMLParser

    class Inputs(HTMLParser):
        def __init__(self):
            super().__init__()
            self.email = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "input" and attrs.get("name") == "email":
                self.email = attrs

    parser = Inputs()
    parser.feed(client.get("/signup").text)
    assert parser.email["type"] == "email"
    assert "required" in parser.email
    assert parser.email["autocomplete"] == "email"
