"""Phase 2 reliability regressions: virtual clocks, mocked providers, isolated DB."""
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
from core.llm.base import RateLimitError
from core.services import generation_budget as budgets
from core.services.generation_admission import GenerationBusy, admission
from tests.test_features import _DB, FIXED_MCQS, Base


class TestBudgets(unittest.TestCase):
    def test_nested_fallback_cannot_reset_deadline(self):
        now = [0.0]
        with patch('time.monotonic', side_effect=lambda: now[0]), budgets.budget(10):
            now[0] = 5
            with budgets.budget(900):
                self.assertEqual(budgets.remaining(), 5)
                now[0] = 11
                with self.assertRaises(budgets.GenerationDeadlineExceeded):
                    budgets.check()
        self.assertIsNone(budgets.remaining())

    def test_timeout_uses_remaining_budget_without_mutating_provider(self):
        with budgets.budget(10):
            timeout = budgets.network_timeout(45)
            self.assertLessEqual(timeout.total, 2)
            self.assertLessEqual(timeout.connect_timeout, 2)
        self.assertEqual(budgets.network_timeout(45), 45)

    def test_api_budget_starts_before_source_lookup(self):
        from fastapi_app.deps import User
        from fastapi_app.routers.api import PracticeIn, practice_generate

        def prework(user, payload):
            self.assertIsNotNone(budgets.remaining())
            self.assertLessEqual(budgets.remaining(), config.PIPELINE_DEADLINE_SECONDS)
            return {'ok': True}

        with patch('fastapi_app.routers.api._admit_generation', side_effect=prework):
            self.assertEqual(practice_generate(User(1, 'test'), PracticeIn(document_id=1)), {'ok': True})

    def test_retry_after_that_cannot_fit_never_sleeps_or_retries(self):
        from tests.test_llm_retry import _FlakyProvider
        provider = _FlakyProvider(99, retry_after=60)
        with budgets.budget(10), patch('time.sleep') as sleep:
            with self.assertRaises(RateLimitError):
                provider.complete_json([])
        self.assertEqual(provider.calls, 1)
        sleep.assert_not_called()

    def test_expired_round_keeps_only_previously_accepted_questions(self):
        from core.services.mcq_pipeline import _collect_mcqs
        now = [0.0]
        accepted = {**FIXED_MCQS[0], 'verification_status': 'verified'}

        def slow(*args, **kwargs):
            now[0] = 20
            return [FIXED_MCQS[1]]

        with patch('time.monotonic', side_effect=lambda: now[0]), budgets.budget(10), \
                patch('core.services.mcq_pipeline.accept_candidates', side_effect=budgets.GenerationDeadlineExceeded):
            result = _collect_mcqs('source', 2, 'medium', accepted=[accepted], candidate_generator=slow)
        self.assertEqual(result, [accepted])

    def test_no_new_work_starts_after_deadline(self):
        from core.services.mcq_pipeline import _legacy_fallback
        with budgets.budget(0), patch('models.mcq_generator.generate_legacy_candidates') as generate:
            with self.assertRaises(budgets.GenerationDeadlineExceeded):
                _legacy_fallback('source', 1, 'medium')
            generate.assert_not_called()

    def test_usage_is_request_local_for_shared_provider(self):
        from tests.test_llm_retry import _FlakyProvider
        provider = _FlakyProvider(0)
        provider.last_usage = {'owner': 'main'}
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertIsNone(pool.submit(lambda: provider.last_usage).result())
        self.assertEqual(provider.last_usage, {'owner': 'main'})

    def test_mmr_reuses_twenty_stored_vectors_without_candidate_embedding(self):
        from core.rag.retrieval.retriever import Retriever
        from core.vectorstore.base import SearchHit
        embedder = Mock()
        embedder.embed.return_value = [1.0, 0.0]
        hits = [SearchHit(str(i), 1, i, str(i), 0.5, [1.0, 0.0]) for i in range(20)]
        with patch('core.rag.retrieval.retriever.get_embedder', return_value=embedder), \
                patch('core.rag.retrieval.retriever.get_vector_store',
                      return_value=SimpleNamespace(query=lambda *args: hits)):
            result = Retriever().retrieve(1, 'query', 5)
        self.assertEqual(len(result), 5)
        embedder.embed.assert_called_once_with('query')
        embedder.embed_batch.assert_not_called()


class TestAtomicGeneration(Base):
    def test_same_user_cannot_run_simultaneous_generations(self):
        entered, release = Event(), Event()
        name = self._testMethodName

        def work():
            with admission(name, seconds=60):
                entered.set()
                self.assertTrue(release.wait(5))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(work)
            self.assertTrue(entered.wait(5))
            try:
                with self.assertRaises(GenerationBusy), admission(name, seconds=60):
                    pass
            finally:
                release.set()
            future.result()

    def test_failure_consumes_attempt_quota_and_releases_execution_lease(self):
        name = self._testMethodName
        with patch.object(config, 'GENERATION_COOLDOWN_SECONDS', 0), \
                patch.object(config, 'GENERATION_MAX_PER_5MIN', 1):
            with self.assertRaises(ValueError), admission(name, seconds=60):
                raise ValueError('provider failed')
            with sqlite3.connect(_DB) as conn:
                self.assertEqual(conn.execute('SELECT count(*) FROM generation_leases WHERE scope=?',
                                              ('user:' + name,)).fetchone()[0], 0)
            with self.assertRaises(GenerationBusy), admission(name, seconds=60):
                pass

    def test_expired_lease_is_reclaimed(self):
        name = self._testMethodName
        with sqlite3.connect(_DB) as conn:
            conn.execute('INSERT INTO generation_leases VALUES (?,?,0)', ('user:' + name, 'old'))
        with admission(name, seconds=60):
            pass

    def test_duplicate_job_delivery_creates_exactly_one_session(self):
        from core.repositories import job_repo
        from core.services.job_runner import create_job, run_generation_job
        job = create_job(user_id=1, username=self._testMethodName, document_id=1,
                         num_questions=1, difficulty='medium', topic=None, timer=60)
        entered, release = Event(), Event()

        def generate(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return FIXED_MCQS[:1]

        with patch('core.services.mcq_pipeline.generate_from_document', side_effect=generate) as generator:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(run_generation_job, job)
                self.assertTrue(entered.wait(5))
                try:
                    pool.submit(run_generation_job, job).result(5)
                finally:
                    release.set()
                first.result(5)
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(job_repo.get(job)['status'], 'done')
        with sqlite3.connect(_DB) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM sessions WHERE teacher=?',
                                          (self._testMethodName,)).fetchone()[0], 1)

    def test_stale_running_job_becomes_recoverable_failure(self):
        from core.repositories import job_repo
        job = job_repo.create(user_id=1, username='stale', document_id=1,
                              params={'deadline_seconds': 60}, total=1)
        job_repo.claim(job)
        with sqlite3.connect(_DB) as conn:
            conn.execute("UPDATE generation_jobs SET updated_at='2000-01-01T00:00:00' WHERE id=?", (job,))
        self.assertEqual(job_repo.get(job)['status'], 'failed')
        self.assertFalse(job_repo.claim(job))

    def test_provider_capacity_is_shared_across_users(self):
        with patch.object(config, 'GENERATION_MAX_CONCURRENT', 1):
            with admission(self._testMethodName + 'a', seconds=60):
                with self.assertRaises(GenerationBusy), admission(self._testMethodName + 'b', seconds=60):
                    pass

    def test_queue_failure_never_executes_long_job_inline(self):
        client = self._client_factory()
        self._seed_and_login(client, self._testMethodName)
        with patch('core.repositories.document_repo.get', return_value={'owner': self._testMethodName}), \
                patch('core.services.generation_status.require_generation'), \
                patch('core.services.job_runner.generation_backend', return_value='rq'), \
                patch('core.services.job_runner.enqueue_job', return_value=False), \
                patch('core.services.job_runner.run_generation_job') as worker:
            response = client.post('/api/practice/generate', json={'document_id': 1})
        self.assertEqual(response.status_code, 503)
        worker.assert_not_called()

    def test_stored_vector_mismatch_is_explicit_and_reindexable(self):
        from core.repositories import document_repo
        from core.vectorstore.sqlite_numpy import SqliteNumpyStore
        doc = document_repo.create(self._testMethodName, self._testMethodName, 'model test', 'paste', 10)
        old = SqliteNumpyStore('old', 2)
        old.upsert(doc, [{'chunk_index': 0, 'content': self._testMethodName, 'vector': [1., 0.]}])
        new = SqliteNumpyStore('new', 3)
        self.assertFalse(new.has_document(doc))
        with self.assertRaises(ValueError):
            new.query(doc, [1., 0., 0.], 1)
        new.upsert(doc, [{'chunk_index': 0, 'content': self._testMethodName, 'vector': [1., 0., 0.]}])
        self.assertTrue(new.has_document(doc))
        self.assertEqual(new.query(doc, [1., 0., 0.], 1)[0].vector, [1., 0., 0.])
