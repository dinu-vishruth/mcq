"""Demo account creation must be explicit, random and absent from public pages."""
from werkzeug.security import check_password_hash

import config
from core.models.db import get_db
from tests.production_support import TEST_PASSWORD


def test_empty_database_does_not_seed_by_default(production_app):
    conn = get_db()
    try:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    finally:
        conn.close()


def test_opt_in_creates_two_demo_users_only_once(production_app, monkeypatch, capsys):
    from fastapi_app.database import init_db
    monkeypatch.setattr(config, "SEED_DEMO_USERS", True)
    init_db()
    conn = get_db()
    try:
        rows = conn.execute("SELECT username, password_hash FROM users ORDER BY username").fetchall()
        assert [row["username"] for row in rows] == ["student", "teacher"]
        assert all(not check_password_hash(row["password_hash"], TEST_PASSWORD) for row in rows)
        assert rows[0]["password_hash"] != rows[1]["password_hash"]
    finally:
        conn.close()
    output = capsys.readouterr().out
    assert "student" in output and "teacher" in output
    init_db()
    assert "password=" not in capsys.readouterr().out


def test_public_login_page_never_advertises_demo_credentials(production_app):
    from tests._client import FlaskLikeClient
    client = FlaskLikeClient(production_app.app, config.SECRET_KEY)
    assert "DemoPass" not in client.get("/").text
