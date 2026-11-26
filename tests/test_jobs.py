"""Background generation job tests (offline, stubbed pipeline + sqlite).

Covers:
  * job_repo lifecycle and ownership
  * GET /api/jobs/{id} shape + 404 for another user's job
  * run_generation_job stores a session and reports done
  * the per-job deadline reaches the pipeline (not the global 55s cap)
"""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="mcq_jobs_")
_DB = os.path.join(_TMP, "jobs.db")
os.environ.setdefault("SEED_DEMO_USERS", "true")

import config

config.DB_PATH = _DB
config.EMBEDDING_BACKEND = "hashing"
config.VECTOR_STORE = "sqlite"

import app as app_module  # noqa: E402
from fastapi_app.database import init_db  # noqa: E402

init_db()

FIXED_MCQS = [
    {"question": "Q1?",
     "options": [{"label": "A", "text": "a"}, {"label": "B", "text": "b"},
                 {"label": "C", "text": "c"}, {"label": "D", "text": "d"}],
     "answer_text": "a"},
]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from config import SECRET_KEY
        from tests._client import make_client_factory
        cls._client_factory = staticmethod(make_client_factory(app_module.app, SECRET_KEY))

    def setUp(self):
        config.DB_PATH = _DB
        conn = sqlite3.connect(_DB)
        conn.execute("DELETE FROM generation_jobs")
        conn.commit()
        conn.close()

    def _seed_and_login(self, c, username):
        from werkzeug.security import generate_password_hash
        conn = sqlite3.connect(_DB)
        try:
            conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'teacher')",
                         (username, generate_password_hash("Cedar river orbit 42!")))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()
        c.post("/login", data={"username": username, "password": "Cedar river orbit 42!"})

    def _make_doc(self, owner, title="Doc"):
        from core.services.ingestion_service import ingest_document
        return ingest_document(
            ("Transactions must be atomic and durable. Indexes speed reads. ") * 20,
            owner=owner, title=title, source_type="paste")["document_id"]


class TestJobRepository(Base):
    def test_lifecycle_and_public_view(self):
        from core.repositories import job_repo

        job_id = job_repo.create(user_id=1, username="u1", document_id=5,
                                 params={"num_questions": 3}, total=3)
        view = job_repo.public_view(job_repo.get(job_id))
        self.assertEqual(view["status"], "queued")
        self.assertEqual(view["progress"], 0)
        self.assertEqual(view["total"], 3)

        job_repo.set_progress(job_id, 2)
        job_repo.set_done(job_id, {"session_key": "abc", "count": 2, "fallback_count": 1,
                                   "mcqs": [{"answer_text": "private"}]})
        view = job_repo.public_view(job_repo.get(job_id))
        self.assertEqual(view["status"], "done")
        self.assertEqual(view["result"]["session_key"], "abc")
        self.assertEqual(view["result"]["fallback_count"], 1)
        self.assertNotIn("mcqs", view["result"])

    def test_ownership_filter(self):
        from core.repositories import job_repo

        job_id = job_repo.create(user_id=1, username="owner", document_id=5,
                                 params={}, total=1)
        self.assertIsNotNone(job_repo.get_for_user(job_id, "owner"))
        self.assertIsNone(job_repo.get_for_user(job_id, "someone-else"))


class TestJobRoutes(Base):
    def test_status_requires_auth_and_ownership(self):
        from core.repositories import job_repo

        c = self._client_factory()
        self._seed_and_login(c, "jobuser")

        job_id = job_repo.create(user_id=1, username="jobuser", document_id=5,
                                 params={}, total=1)
        r = c.get(f"/api/jobs/{job_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], job_id)

        other = self._client_factory()
        self._seed_and_login(other, "intruder")
        self.assertEqual(other.get(f"/api/jobs/{job_id}/events").status_code, 404)

        anon = self._client_factory()
        self.assertEqual(anon.get(f"/api/jobs/{job_id}").status_code, 401)

    def test_completed_job_streams_terminal_event(self):
        from core.repositories import job_repo

        c = self._client_factory()
        self._seed_and_login(c, "streamuser")
        job_id = job_repo.create(user_id=1, username="streamuser", document_id=5,
                                 params={}, total=1)
        job_repo.set_done(job_id, {"session_key": "k", "count": 1, "mcqs": []})
        r = c.get(f"/api/jobs/{job_id}/events")
        self.assertEqual(r.status_code, 200)
        self.assertIn("done", r.text)


class TestJobExecution(Base):
    def test_run_generation_job_reports_progress_and_done(self):
        from core.repositories import job_repo
        from core.services import job_runner

        doc_id = self._make_doc("worker")
        job_id = job_runner.create_job(
            user_id=1, username="worker", document_id=doc_id, num_questions=1,
            difficulty="medium", topic=None, timer=300, deadline_seconds=900,
        )

        seen = {}

        def fake_generate(document_id, **kwargs):
            seen["deadline"] = kwargs.get("deadline_seconds")
            cb = kwargs.get("on_progress")
            if cb:
                cb(1)
            return FIXED_MCQS

        with patch("core.services.mcq_pipeline.generate_from_document", fake_generate):
            job_runner.run_generation_job(job_id)

        view = job_repo.public_view(job_repo.get(job_id))
        self.assertEqual(view["status"], "done")
        self.assertEqual(view["progress"], 1)
        self.assertTrue(view["result"]["session_key"])
        # Per-job budget, not the global serverless default.
        self.assertEqual(seen["deadline"], 900)

    def test_failed_job_records_error(self):
        from core.repositories import job_repo
        from core.services import job_runner

        doc_id = self._make_doc("failer")
        job_id = job_runner.create_job(
            user_id=1, username="failer", document_id=doc_id, num_questions=1,
            difficulty="medium", topic=None, timer=300,
        )

        def boom(*a, **kw):
            raise ValueError("no indexed content")

        with patch("core.services.mcq_pipeline.generate_from_document", boom):
            with self.assertRaises(ValueError):
                job_runner.run_generation_job(job_id)

        view = job_repo.public_view(job_repo.get(job_id))
        self.assertEqual(view["status"], "failed")
        self.assertIn("verified quiz generation", view["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
