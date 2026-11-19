"""Pair-scoped lockout and literal password handling on signup/login/profile."""
import pytest
from fastapi.testclient import TestClient

import config
from tests._client import FlaskLikeClient
from tests.production_support import TEST_PASSWORD


def _client_at(app, ip):
    client = FlaskLikeClient(app, config.SECRET_KEY)
    client._client = TestClient(app, client=(ip, 12345), follow_redirects=False)
    return client


def test_lockout_applies_to_username_ip_pair_only(production_app):
    production_app.client("alice")
    production_app.client("bob")
    first = _client_at(production_app.app, "1.2.3.4")
    for _ in range(5):
        first.post("/login", data={"username": "alice", "password": "intentionally-wrong"})
    assert "Too many" in first.post("/login", data={"username": "alice", "password": TEST_PASSWORD}).text
    assert first.post("/login", data={"username": "bob", "password": TEST_PASSWORD}).status_code == 302
    second = _client_at(production_app.app, "5.6.7.8")
    assert second.post("/login", data={"username": "alice", "password": TEST_PASSWORD}).status_code == 302


@pytest.mark.parametrize("password", ["  Roundtrip-913!", "Roundtrip-913!  "])
def test_signup_and_login_preserve_password_spaces(production_app, password):
    client = FlaskLikeClient(production_app.app, config.SECRET_KEY)
    assert client.post("/signup", data={"username": "  spaced  ", "password": password}).status_code == 302
    assert "Invalid username or password" in client.post("/login", data={
        "username": "spaced", "password": password.strip()}).text
    assert client.post("/login", data={"username": "spaced", "password": password}).status_code == 302


def test_profile_password_preserves_spaces(production_app):
    client, _ = production_app.client()
    password = "Updated-Pass-382! "
    assert client.post("/profile", data={"username": "alice", "password": password}).status_code == 302
    client.get("/logout")
    assert "Invalid username or password" in client.post("/login", data={"username": "alice", "password": password.strip()}).text
    assert client.post("/login", data={"username": "alice", "password": password}).status_code == 302


@pytest.mark.parametrize("password", ["Short-1!", "onlylowercase", "ONLYUPPERCASE", "123456789012", "!!!!!!!!!!!!"])
def test_signup_rejects_short_or_single_class_passwords(production_app, password):
    client = FlaskLikeClient(production_app.app, config.SECRET_KEY)
    response = client.post("/signup", data={"username": "weak", "password": password})
    assert response.status_code == 200
    assert "Password must" in response.text


def test_profile_rejects_single_class_replacement(production_app):
    client, _ = production_app.client()
    assert "Password must" in client.post("/profile", data={"username": "alice", "password": "onlylowercase"}).text
