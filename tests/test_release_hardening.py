"""Offline security, deadline, concurrency and evidence regressions."""
import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi import HTTPException

import config
from core.agents.fact_verification import FactVerificationAgent
from core.llm.base import LLMProvider, RateLimitError
from core.llm.budget import BudgetExceeded, generation_budget, http_timeout, remaining, time_budget
from core.models.questions import attach_evidence, validate_questions
from core.repositories import job_repo, session_repo
from core.services.generation_limits import generation_slot
from tests.test_features import _DB, FIXED_MCQS, Base, app_module


@pytest.mark.parametrize("change", [
    {"question": 123}, {"answer_text": "not an option"},
    {"options": [{"text": " a "}, {"text": "A"}, {"text": "b"}, {"text": "c"}]},
    {"options": [None] * 4},
])
def test_same_mcq_contract_rejects_malformed_generation(change):
    from core.agents.quality_assurance import QualityAssuranceAgent
    from models.mcq_generator import _structurally_valid

    question = {**FIXED_MCQS[0], **change}
    assert not _structurally_valid([question])
    assert not QualityAssuranceAgent().run([question])[0]


@pytest.mark.parametrize("payload", [None, 1, {}, [None, 1, {}]])
def test_validator_handles_non_question_payloads(payload):
    valid, rejected = validate_questions(payload)
    assert valid == []
    assert rejected


@pytest.mark.parametrize("verdict", [
    {"index": 0, "passed": "false", "confidence": 1.0},
    {"index": 0, "passed": True, "confidence": "0.9"},
    {"index": 0, "passed": True, "confidence": float("nan")},
    {"index": True, "passed": True, "confidence": 1.0},
    {"index": 0, "passed": True},
])
def test_invalid_verdict_never_grants_verified_status(verdict):
    llm = Mock()
    llm.complete_json.return_value = {"results": [verdict]}
    with patch("core.agents.fact_verification.get_llm", return_value=llm):
        questions, _ = FactVerificationAgent().run([FIXED_MCQS[0]], "source")
    assert questions[0]["verification"] == "unverified"


def test_evidence_is_checked_against_actual_context_and_not_model_ids():
    source = "WHERE filters rows before grouping. HAVING filters aggregated groups."
    question = {**FIXED_MCQS[0], "source_hint": "rows before grouping", "verification": "verified",
                "sources": [{"quote": "forged", "document_id": 999}]}
    hits = [SimpleNamespace(content=source, chunk_index=3)]
    checked = attach_evidence([question], source, hits=hits, document_id=42)[0]
    assert checked["sources"] == [{"quote": "rows before grouping", "document_id": 42, "chunk_index": 3}]
    missing = attach_evidence([{**question, "source_hint": "not in the source"}], source)[0]
    assert "sources" not in missing
    # An LLM cannot confer either status or IDs through its raw payload.
    parsed, _ = validate_questions([question])
    assert "verification" not in parsed[0] and "sources" not in parsed[0]


def test_nested_budgets_shrink_timeouts_and_never_reset_outer_deadline():
    clock = [100.0]
    with patch("core.llm.budget.time.monotonic", side_effect=lambda: clock[0]):
        with time_budget(10):
            clock[0] = 103
            assert http_timeout(45).total == 7
            with time_budget(100):
                assert remaining() == 7
            with time_budget(2):
                assert remaining() == 2
            clock[0] = 110
            with pytest.raises(BudgetExceeded):
                remaining()
    assert remaining() is None


def test_retry_cannot_sleep_beyond_remaining_budget():
    class Limited(LLMProvider):
        def _complete_once(self, *args, **kwargs):
            raise RateLimitError("limited", retry_after=5)

    with patch("time.sleep") as sleep, time_budget(1):
        with pytest.raises(RateLimitError):
            Limited("test", "test").complete_json([])
        sleep.assert_not_called()


def test_nested_generator_inherits_worker_budget_unless_explicitly_shortened():
    @generation_budget
    def helper(**kwargs):
        return remaining()

    with patch.object(config, "PIPELINE_DEADLINE_SECONDS", 1), time_budget(10):
        assert helper() > 9
        assert helper(deadline_seconds=1) <= 1


def test_legacy_fallback_never_mutates_pipeline_setting():
    from core.services.mcq_pipeline import _legacy_fallback

    with patch.object(config, "AI_PIPELINE", "rag"), patch(
        "models.mcq_generator.generate_mcqs_legacy", return_value=FIXED_MCQS,
    ):
        _legacy_fallback("source", 2, "medium")
        assert config.AI_PIPELINE == "rag"


class TestReleaseIntegration(Base):
    def setUp(self):
        super().setUp()
        self.name = self._testMethodName
        self.client = self._client_factory()
        self._seed_and_login(self.client, self.name)
        with sqlite3.connect(_DB) as conn:
            self.uid = conn.execute("SELECT id FROM users WHERE username=?", (self.name,)).fetchone()[0]

    def test_private_key_cannot_start_resume_or_submit_another_users_quiz(self):
        key = session_repo.create_session(self.name, "medium", 60, FIXED_MCQS, owner_id=self.uid)
        other = self._client_factory()
        self._seed_and_login(other, self.name + "_other")
        assert len(key) == 32
        joined = other.post("/student_login", data={"session_key": key})
        assert "not found in your account" in joined.text
        with other.session_transaction() as session:
            session["session_key"] = key
        assert other.get("/mcq_test").status_code == 302
        assert other.post("/submit", data={"q-0": FIXED_MCQS[0]["answer_text"]}).status_code == 302
        assert other.post("/api/quiz-attempt", json={"session_key": key}).status_code == 404
        with sqlite3.connect(_DB) as conn:
            assert conn.execute("SELECT COUNT(*) FROM quiz_attempts WHERE session_key=?", (key,)).fetchone()[0] == 0

    def test_concurrent_requests_admit_one_generation_even_with_throttles_disabled(self):
        admitted = threading.Event()
        release = threading.Event()

        def first():
            with generation_slot(self.uid, self.name):
                admitted.set()
                release.wait(timeout=5)

        with patch.multiple(config, GENERATION_COOLDOWN_SECONDS=0, GENERATION_MAX_PER_5MIN=0):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(first)
                try:
                    assert admitted.wait(timeout=3)
                    with pytest.raises(HTTPException) as error:
                        with generation_slot(self.uid, self.name):
                            pytest.fail("A second concurrent generation was admitted")
                    assert error.value.status_code == 429
                finally:
                    release.set()
                    future.result(timeout=5)
            with generation_slot(self.uid, self.name):
                pass

    def test_failed_attempt_releases_slot_but_counts_toward_limit(self):
        with patch.multiple(config, GENERATION_COOLDOWN_SECONDS=0, GENERATION_MAX_PER_5MIN=1):
            with pytest.raises(ValueError), generation_slot(self.uid, self.name):
                raise ValueError("provider failed")
            with pytest.raises(HTTPException) as error:
                with generation_slot(self.uid, self.name):
                    pass
            assert error.value.status_code == 429
        with sqlite3.connect(_DB) as conn:
            assert conn.execute("SELECT active FROM generation_requests WHERE user_id=?", (self.uid,)).fetchone()[0] == 0

    def test_job_delivery_cannot_generate_a_second_session(self):
        from core.services.job_runner import create_job, run_generation_job

        job = create_job(user_id=self.uid, username=self.name, document_id=1,
                         num_questions=2, difficulty="medium", topic=None, timer=60)
        with patch("core.services.mcq_pipeline.generate_from_document", return_value=FIXED_MCQS) as generate:
            run_generation_job(job)
            key = job_repo.public_view(job_repo.get(job))["result"]["session_key"]
            run_generation_job(job)
            assert generate.call_count == 1
            assert job_repo.public_view(job_repo.get(job))["result"]["session_key"] == key

    def test_expired_queued_reservation_reports_failure_and_cannot_start(self):
        from core.services.job_runner import create_job, run_generation_job

        with patch.multiple(config, GENERATION_COOLDOWN_SECONDS=0, GENERATION_MAX_PER_5MIN=0):
            with generation_slot(self.uid, self.name) as reservation:
                job = create_job(user_id=self.uid, username=self.name, document_id=1,
                                 num_questions=2, difficulty="medium", topic=None, timer=60,
                                 reservation_id=reservation.id)
                reservation.transferred = True
            with sqlite3.connect(_DB) as conn:
                conn.execute("UPDATE generation_requests SET expires_at=0 WHERE id=?", (reservation.id,))
            assert job_repo.get_for_user(job, user_id=self.uid)["status"] == "failed"
            with patch("core.services.mcq_pipeline.generate_from_document") as generate:
                run_generation_job(job)
            generate.assert_not_called()
            with generation_slot(self.uid, self.name):
                pass

    def test_queue_failure_does_not_run_inline_or_leave_an_active_reservation(self):
        uploaded = self.client.post("/ingest_resource", data={
            "extracted_text": "Rows are filtered before grouping. " * 20, "title": "Queue material",
        })
        assert uploaded.status_code == 302
        document_id = self.client.get("/api/knowledge").json()["items"][0]["id"]
        with patch("core.services.generation_status.generation_available", return_value=True), patch(
            "core.services.job_runner.generation_backend", return_value="rq",
        ), patch("core.services.job_runner.enqueue_job", return_value=False), patch(
            "core.services.mcq_pipeline.generate_from_document",
        ) as generate:
            response = self.client.post("/api/practice/generate", json={"document_id": document_id})
        assert response.status_code == 503
        generate.assert_not_called()
        with sqlite3.connect(_DB) as conn:
            assert conn.execute("SELECT active FROM generation_requests WHERE user_id=?", (self.uid,)).fetchone()[0] == 0

    def test_stored_source_explanation_survives_submission_without_another_llm(self):
        question = {**FIXED_MCQS[0], "explanation": "WHERE filters individual rows before aggregation.",
                    "verification": "verified", "sources": [{"quote": "rows before grouping", "document_id": 1, "chunk_index": 0}]}
        key = session_repo.create_session(self.name, "medium", 60, [question], owner_id=self.uid)
        self.client.post("/student_login", data={"session_key": key})
        page = self.client.get("/mcq_test")
        assert "source_hint" not in page.text and "WHERE filters individual" not in page.text
        with patch("models.explanation_engine.explain_answers") as explain, patch("core.services.learning_service.analyse_submission"):
            result = self.client.post("/submit", data={"q-0": question["answer_text"]})
            feedback = self.client.post(f"/api/results/{key}/enrich", json={})
            assert feedback.status_code == 200
        explain.assert_not_called()
        assert result.status_code == 303
        with sqlite3.connect(_DB) as conn:
            details = json.loads(conn.execute("SELECT detail_json FROM results WHERE session_key=?", (key,)).fetchone()[0])
        assert details[0]["explanation"] == question["explanation"]
        assert details[0]["sources"] == question["sources"]
        assert details[0]["verification"] == "verified"

    def test_feedback_is_private_and_a_crashed_claim_does_not_replay_analysis(self):
        key = session_repo.create_session(self.name, "medium", 60, FIXED_MCQS, owner_id=self.uid)
        self.client.post("/student_login", data={"session_key": key})
        self.client.post("/submit", data={})
        other = self._client_factory()
        self._seed_and_login(other, self.name + "_other")
        assert other.post(f"/api/results/{key}/enrich", json={}).status_code == 404
        with sqlite3.connect(_DB) as conn:
            conn.execute("UPDATE results SET enrichment_status='running',enrichment_deadline=0 WHERE session_key=?", (key,))
        with patch("core.services.learning_service.analyse_submission") as analysis:
            feedback = self.client.post(f"/api/results/{key}/enrich", json={})
        assert feedback.json()["failed"]
        assert not feedback.json()["pending"]
        analysis.assert_not_called()

    def test_slow_enrichment_does_not_block_health_on_same_event_loop(self):
        key = session_repo.create_session(self.name, "medium", 60, FIXED_MCQS, owner_id=self.uid)
        self.client.post("/student_login", data={"session_key": key})
        entered, release = threading.Event(), threading.Event()

        def explain(details):
            entered.set()
            release.wait(timeout=3)
            return ["Explanation"] * len(details)

        token = self.client._csrf()
        cookies = self.client._client.cookies

        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                         base_url="http://testserver", cookies=cookies) as client:
                task = asyncio.create_task(client.post(f"/api/results/{key}/enrich", json={}, headers={"X-CSRFToken": token}))
                try:
                    async def wait_started():
                        while not entered.is_set():
                            await asyncio.sleep(.01)
                    await asyncio.wait_for(wait_started(), timeout=1)
                    assert not task.done(), "Enrichment ran directly on the event loop"
                    response = await asyncio.wait_for(client.get("/health"), timeout=.5)
                    assert response.status_code == 200
                    with sqlite3.connect(_DB) as conn:
                        assert conn.execute("SELECT COUNT(*) FROM results WHERE session_key=?", (key,)).fetchone()[0] == 1
                finally:
                    release.set()
                    response = await task
                    assert response.status_code == 200, response.text

        with patch("models.explanation_engine.explain_answers", side_effect=explain) as provider, patch("core.services.learning_service.analyse_submission"):
            result = self.client.post("/submit", data={})
            assert result.status_code == 303
            provider.assert_not_called()  # Result availability never waits for AI.
            asyncio.run(check())
