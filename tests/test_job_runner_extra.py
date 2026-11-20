"""Job-runner extras: enqueue path and backend resolution (no Redis needed)."""
import os
import sys
import types
import unittest
from unittest.mock import patch

from core.services import job_runner


class _FakeQueue:
    def __init__(self, raise_on_enqueue=False):
        self.calls = []
        self.raise_on_enqueue = raise_on_enqueue

    def enqueue(self, func, job_id, job_timeout=None):
        if self.raise_on_enqueue:
            raise RuntimeError("redis down")
        self.calls.append((func, job_id, job_timeout))
        return "rq-job"


class TestEnqueue(unittest.TestCase):
    def setUp(self):
        # Point at a throwaway DB so create_job has a table.
        import tempfile

        self._db = os.path.join(tempfile.mkdtemp(), "jobs.db")
        os.environ.setdefault("SEED_DEMO_USERS", "true")
        import config

        config.DB_PATH = self._db
        from fastapi_app.database import init_db

        init_db()
        self._orig_queue = job_runner._queue

    def tearDown(self):
        job_runner._queue = self._orig_queue

    def _make_job(self):
        return job_runner.create_job(
            user_id=1, username="u", document_id=1, num_questions=2,
            difficulty="medium", topic=None, timer=300,
        )

    def test_enqueue_success(self):
        fake = _FakeQueue()
        job_runner._queue = lambda: fake
        job_id = self._make_job()
        self.assertTrue(job_runner.enqueue_job(job_id))
        self.assertEqual(len(fake.calls), 1)
        _, enqueued_id, timeout = fake.calls[0]
        self.assertEqual(enqueued_id, job_id)
        self.assertGreaterEqual(timeout, 900)

    def test_enqueue_failure_returns_false(self):
        fake = _FakeQueue(raise_on_enqueue=True)
        job_runner._queue = lambda: fake
        job_id = self._make_job()
        self.assertFalse(job_runner.enqueue_job(job_id))


class TestBackendResolution(unittest.TestCase):
    def setUp(self):
        self._saved = {}
        import config

        self._config = config
        self._saved["REDIS_URL"] = config.REDIS_URL
        self._saved["GENERATION_BACKEND"] = config.GENERATION_BACKEND
        config.REDIS_URL = ""
        config.GENERATION_BACKEND = "auto"

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(self._config, k, v)

    def test_auto_without_redis_is_inline(self):
        self.assertEqual(job_runner.generation_backend(), "inline")

    def test_explicit_inline(self):
        self._config.GENERATION_BACKEND = "inline"
        self.assertEqual(job_runner.generation_backend(), "inline")

    def test_explicit_rq(self):
        self._config.GENERATION_BACKEND = "rq"
        self.assertEqual(job_runner.generation_backend(), "rq")

    def test_auto_with_redis_and_modules_is_rq(self):

        self._config.REDIS_URL = "redis://localhost:6379"
        self._config.GENERATION_BACKEND = "auto"
        fake_redis = types.ModuleType("redis")
        fake_rq = types.ModuleType("rq")
        with patch.dict(sys.modules, {"redis": fake_redis, "rq": fake_rq}):
            self.assertEqual(job_runner.generation_backend(), "rq")

    def test_auto_with_redis_but_missing_modules_is_inline(self):
        self._config.REDIS_URL = "redis://localhost:6379"
        # Remove redis/rq from sys.modules if present (they aren't installed here).
        with patch.dict(sys.modules, {"redis": None, "rq": None}):
            self.assertEqual(job_runner.generation_backend(), "inline")

    def test_unknown_backend_falls_back_to_inline(self):
        self._config.GENERATION_BACKEND = "mystery"
        self.assertEqual(job_runner.generation_backend(), "inline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
