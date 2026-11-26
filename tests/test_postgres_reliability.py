"""Real PostgreSQL tests use only the explicitly named isolated CI database."""
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

import config

pytestmark = pytest.mark.skipif(not os.getenv("MCQ_TEST_DATABASE_URL"), reason="isolated Postgres URL required")


def test_concurrent_boot_and_identity_uniqueness(monkeypatch):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from core.models.db import get_db
    from fastapi_app.database import init_db

    url = os.environ["MCQ_TEST_DATABASE_URL"]
    schema = "reliability_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            monkeypatch.setattr(config, "DATABASE_URL", make_conninfo(url, options=f"-c search_path={schema}"))
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda _: init_db(), range(3)))
            conn = get_db()
            try:
                conn.execute("INSERT INTO users(username,password_hash,role,email) VALUES ('one','x','user','A@example.com')")
                conn.commit()
                import sqlite3
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute("INSERT INTO users(username,password_hash,role,email) VALUES ('two','x','user',' a@EXAMPLE.COM ')")
                conn.rollback()
                assert conn.execute("SELECT count(*) FROM sql_migrations").fetchone()[0] >= 8
                assert conn.execute("SELECT count(*) FROM users WHERE username='one'").fetchone()[0] == 1
                conn.execute("INSERT INTO documents(id,owner,title) VALUES (12,'one','evidence')")
                conn.commit()
                from core.repositories import job_repo
                from core.services.generation_metrics import GenerationMetrics
                uid = conn.execute("SELECT id FROM users WHERE username='one'").fetchone()[0]
                jid = job_repo.create(user_id=uid, username="one", document_id=12, params={}, total=1)
                assert job_repo.claim(jid, "first-worker")
                assert not job_repo.claim(jid, "duplicate-worker")
                key = job_repo.publish(job_repo.get(jid), "first-worker", [{"question": "test"}], "easy", 60, GenerationMetrics())
                assert job_repo.public_view(job_repo.get(jid))["result"]["session_key"] == key
            finally:
                conn.close()
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
