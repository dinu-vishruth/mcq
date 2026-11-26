"""Account upgrades preserve existing identities and their dependent data."""
import sqlite3

import pytest

from core.models.legacy_roles import upgrade_legacy_user_roles


@pytest.mark.parametrize("foreign_keys", [0, 1])
def test_legacy_role_upgrade_preserves_data_constraints_and_id_sequence(tmp_path, foreign_keys):
    with sqlite3.connect(tmp_path / "legacy.db") as conn:
        conn.execute(f"PRAGMA foreign_keys={foreign_keys}")
        conn.executescript("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('teacher', 'student')),
                email TEXT, custom_column TEXT DEFAULT 'keep'
            );
            CREATE UNIQUE INDEX contact_email ON users(lower(email)) WHERE email IS NOT NULL;
            CREATE TABLE owned (user_id INTEGER REFERENCES users(id) ON DELETE CASCADE);
            CREATE TABLE audit (user_id INTEGER);
            CREATE TRIGGER new_user AFTER INSERT ON users BEGIN INSERT INTO audit VALUES (new.id); END;
            INSERT INTO users(id,username,role,email) VALUES (7,'existing','student','old@example.com');
            INSERT INTO owned VALUES (7);
            INSERT INTO users(id,username,role) VALUES (50,'removed','teacher');
            DELETE FROM users WHERE id=50;
        """)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO users(username,role) VALUES ('before','user')")
        conn.rollback()

        upgrade_legacy_user_roles(conn)
        upgrade_legacy_user_roles(conn)

        assert conn.execute("SELECT * FROM users").fetchall() == [(7, 'existing', 'student', 'old@example.com', 'keep')]
        assert conn.execute("SELECT * FROM owned").fetchall() == [(7,)]
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == foreign_keys
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.execute("INSERT INTO users(username,role,email) VALUES ('learner','user','new@example.com')")
        new_id = conn.execute("SELECT id FROM users WHERE username='learner'").fetchone()[0]
        assert new_id > 50
        assert conn.execute("SELECT user_id FROM audit ORDER BY rowid DESC LIMIT 1").fetchone()[0] == new_id
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO users(username,role,email) VALUES ('duplicate','user','NEW@example.com')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO users(username,role) VALUES ('invalid','admin')")


def test_fresh_users_schema_needs_no_rebuild():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, role TEXT DEFAULT 'user')")
        upgrade_legacy_user_roles(conn)
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='users_role_upgrade'").fetchone()


def test_role_upgrade_failure_rolls_back_and_restores_foreign_keys():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, role TEXT CHECK(role IN ('student', 'teacher')))")
        conn.execute("INSERT INTO users VALUES (1,'student')")
        conn.commit()
        conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DROP_TABLE else sqlite3.SQLITE_OK)
        with pytest.raises(sqlite3.DatabaseError):
            upgrade_legacy_user_roles(conn)
        conn.set_authorizer(None)
        assert conn.execute("SELECT * FROM users").fetchall() == [(1, 'student')]
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='users_role_upgrade'").fetchone()


def test_legacy_database_boot_supports_demo_and_email_signup(tmp_path, monkeypatch):
    import config
    from core.embeddings import reset_embedder
    from core.vectorstore import reset_vector_store
    from fastapi_app.database import init_db

    filename = tmp_path / "historical.db"
    with sqlite3.connect(filename) as conn:
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, "
                     "password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('teacher','student')), "
                     "created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("INSERT INTO users VALUES (7,'original','original-hash','student','2020-01-01')")
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "DB_PATH", str(filename))
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")
    monkeypatch.setattr(config, "VECTOR_STORE", "sqlite")
    reset_embedder()
    reset_vector_store()
    try:
        init_db()
        init_db()
        from fastapi_app.main import create_app
        from tests._client import make_client_factory
        client = make_client_factory(create_app(), config.SECRET_KEY)()
        assert client.post("/demo").headers["Location"] == "/dashboard"
        assert b'data-page="dashboard"' in client.get("/dashboard").data
        items = client.get("/api/knowledge").get_json()["items"]
        assert len(items) == 1 and items[0]["indexed"]
        assert client.post("/signup", data={"username": "new-learner", "email": "learner@gmail.com", "password": "Local cedar river orbit!"}).status_code == 302
        with sqlite3.connect(filename) as conn:
            assert conn.execute("SELECT id,password_hash,role FROM users WHERE username='original'").fetchone() == (7, 'original-hash', 'student')
            assert conn.execute("SELECT email,email_verified FROM users WHERE username='new-learner'").fetchone() == ('learner@gmail.com', 0)
    finally:
        reset_embedder()
        reset_vector_store()
