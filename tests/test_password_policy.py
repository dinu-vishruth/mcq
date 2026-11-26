"""Strong-password checks must agree across every credential-writing path."""
import pytest
from werkzeug.security import check_password_hash, generate_password_hash

from core.services.account_recovery import consume_token, issue_token
from core.services.password_policy import validate_password
from tests.test_admin_dashboard import PASSWORD, sign_in
from tests.test_admin_dashboard import workspace as workspace


@pytest.mark.parametrize("password", ["short", "123456789012345", "Password123456789!", "aaaaaaaaaaaaaaaaaaaa", "qwerty12345678901", "x" * 129, "    stone      ", "abcabcabcabcabcabc"])
def test_short_common_repeated_and_oversized_passwords_are_rejected(password):
    with pytest.raises(ValueError):
        validate_password(password)


@pytest.mark.parametrize("password", ["Cedar42!", "cedar river orbit lantern", "雪の山と川と空の思い出を大切にする", "  Cedar river orbit 42!  "])
def test_unique_passphrases_unicode_and_spaces_are_allowed(password):
    validate_password(password)


def test_signup_rejects_weak_passwords_on_the_server(workspace):
    _, conn, factory = workspace
    response = factory().post("/signup", data={"username": "new", "email": "new@example.com", "password": "password123456789"})
    assert response.status_code == 200
    assert b"less predictable" in response.data
    assert conn.execute("SELECT 1 FROM users WHERE username='new'").fetchone() is None


def test_legacy_weak_password_only_allows_a_password_upgrade(workspace):
    _, conn, factory = workspace
    conn.execute("UPDATE users SET password_hash=? WHERE username='learner'", (generate_password_hash("old-pwd"),))
    conn.commit()
    client = factory()
    response = client.post("/login", data={"username": "learner", "password": "old-pwd"})
    assert response.headers["Location"] == "/profile"
    assert client.get("/dashboard").headers["Location"] == "/profile"
    assert client.get("/api/prefs").status_code == 403
    assert b'"password_upgrade_required": true' in client.get("/profile").data
    assert b"stronger password" in client.post("/profile", data={"username": "learner"}).data
    assert client.post("/profile", data={"username": "learner", "password": PASSWORD, "current_password": "old-pwd"}).status_code == 302
    assert client.get("/api/prefs").status_code == 200


def test_password_change_requires_current_password_and_revokes_other_sessions(workspace):
    _, conn, factory = workspace
    first, second = sign_in(factory, "learner"), sign_in(factory, "learner")
    replacement = "Cobalt mountain morning 27!"
    response = first.post("/profile", data={"username": "learner", "password": replacement})
    assert b"Enter your current password" in response.data
    response = first.post("/profile", data={"username": "learner", "password": "weak", "current_password": PASSWORD})
    assert b"at least 8" in response.data
    assert first.post("/profile", data={"username": "learner", "password": replacement, "current_password": PASSWORD}).status_code == 302
    assert first.get("/api/prefs").status_code == 200
    assert second.get("/api/prefs").status_code == 401
    row = conn.execute("SELECT * FROM users WHERE username='learner'").fetchone()
    assert check_password_hash(row["password_hash"], replacement)


def test_recovery_rejects_weak_password_without_consuming_the_token(workspace):
    _, conn, _ = workspace
    conn.execute("UPDATE users SET email_verified=1 WHERE username='learner'")
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE username='learner'").fetchone()
    token = issue_token(conn, row, "reset")
    assert not consume_token(conn, token, "Password123456789!")
    assert consume_token(conn, token, "Cedar mountain orbit 47!")
    assert not consume_token(conn, token, PASSWORD)


def test_signup_and_login_preserve_password_whitespace(workspace):
    _, conn, factory = workspace
    password = "  Cedar mountain orbit 47!  "
    client = factory()
    assert client.post("/signup", data={"username": "spaces", "email": "spaces@example.com", "password": password}).status_code == 302
    row = conn.execute("SELECT password_hash FROM users WHERE username='spaces'").fetchone()
    assert check_password_hash(row["password_hash"], password)
    assert not check_password_hash(row["password_hash"], password.strip())
    assert client.post("/login", data={"username": "spaces", "password": password}).status_code == 302
