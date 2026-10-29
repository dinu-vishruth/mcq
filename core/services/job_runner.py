"""Background generation jobs: orchestration + worker entry point.

Two execution modes share the same core (``run_generation_job``):

  * inline  -- the web process calls it inside the request (historical
               behaviour; used when no Redis is configured).
  * rq      -- POST /api/practice/generate creates a job row and enqueues this
               function; a separate ``python worker.py`` process executes it.

The job row, not the queue, is the source of truth: GET /api/jobs/{id} can be
served by any web instance, and a retried/duplicated queue delivery is harmless
because results overwrite the same row id.
"""
from __future__ import annotations

import json

import config
from core.repositories import job_repo


def generation_backend() -> str:
    """Resolve GENERATION_BACKEND=auto to the backend that can actually run."""
    choice = config.GENERATION_BACKEND
    if choice == "auto":
        if not config.REDIS_URL:
            return "inline"
        try:
            import redis  # noqa: F401
            import rq  # noqa: F401
        except ImportError:
            return "inline"
        return "rq"
    return choice if choice in ("inline", "rq") else "inline"


def _queue():
    from redis import Redis
    from rq import Queue

    return Queue(config.QUEUE_NAME, connection=Redis.from_url(config.REDIS_URL))


def create_job(*, user_id: int, username: str, document_id: int, num_questions: int,
               difficulty: str, topic: str | None, timer: int,
               deadline_seconds: int | None = None) -> str:
    params = {
        "num_questions": num_questions,
        "difficulty": difficulty,
        "topic": topic,
        "timer": timer,
        # Resolution order: caller override -> job default. Never the serverless
        # request cap: a queued job isn't bound by maxDuration.
        "deadline_seconds": deadline_seconds or config.GENERATION_JOB_DEADLINE_SECONDS,
    }
    return job_repo.create(
        user_id=user_id, username=username, document_id=document_id,
        params=params, total=num_questions,
    )


def enqueue_job(job_id: str) -> bool:
    """Try to enqueue. False means the caller should run inline instead."""
    job = job_repo.get(job_id)
    if job is None:
        return False
    try:
        queue = _queue()
        timeout = json.loads(job["params_json"]).get("deadline_seconds", 900) + 60
        queue.enqueue(run_generation_job, job_id, job_timeout=timeout)
        return True
    except Exception as exc:
        print(f"[jobs] Could not enqueue {job_id} ({exc}); running inline.")
        return False


def run_generation_job(job_id: str) -> None:
    """Execute one generation job. Safe to call from the worker or inline."""
    job = job_repo.get(job_id)
    if job is None:
        return

    from core.services.mcq_pipeline import generate_from_document
    from utils.session_manager import create_session_key

    params = json.loads(job.get("params_json") or "{}")
    num_questions = int(params.get("num_questions") or 5)
    difficulty = params.get("difficulty") or "medium"
    topic = params.get("topic")
    timer = int(params.get("timer") or 600)
    deadline = int(params.get("deadline_seconds") or config.GENERATION_JOB_DEADLINE_SECONDS)

    job_repo.set_running(job_id)
    try:
        mcqs = generate_from_document(
            job["document_id"],
            num_questions=num_questions,
            difficulty=difficulty,
            topic=topic,
            deadline_seconds=deadline,
            on_progress=lambda n: job_repo.set_progress(job_id, n),
        )
        if not mcqs:
            raise ValueError("No questions generated. Try different settings.")

        session_key = create_session_key(
            teacher=job["username"], difficulty=difficulty, timer=timer, mcqs=mcqs,
            document_id=job["document_id"],
        )
        job_repo.set_done(job_id, {
            "session_key": session_key,
            "count": len(mcqs),
            "mcqs": mcqs,
        })
    except Exception as exc:
        job_repo.set_failed(job_id, str(exc))
        raise
