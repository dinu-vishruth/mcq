"""RQ worker entry point for background MCQ generation.

Run alongside the web app whenever GENERATION_BACKEND=rq (or auto with
REDIS_URL set):

    python worker.py

It boots the same database schema the app uses (important: the worker must see
the same DATABASE_URL), then blocks on the configured queue.
"""
from __future__ import annotations

import config
from core.observability import configure_logging
from core.services.job_runner import generation_backend


def main() -> None:
    configure_logging()
    if not config.REDIS_URL:
        raise SystemExit("REDIS_URL must be set to run the generation worker.")
    try:
        from redis import Redis
        from rq import Queue, Worker
    except ImportError as exc:
        raise SystemExit(
            "The rq worker needs the optional dependencies: pip install redis rq"
        ) from exc

    # Same idempotent boot schema as the web process. Without this a fresh
    # Postgres database would have no generation_jobs table when the worker
    # claims its first job.
    from fastapi_app.database import init_db

    init_db()

    if generation_backend() != "rq":
        raise SystemExit(
            f"GENERATION_BACKEND resolved to {generation_backend()!r}; "
            "set GENERATION_BACKEND=rq (and REDIS_URL) before starting the worker."
        )

    connection = Redis.from_url(config.REDIS_URL)
    queues = [Queue(config.QUEUE_NAME, connection=connection)]
    print(f"[worker] listening on queue '{config.QUEUE_NAME}' at {config.REDIS_URL}")
    Worker(queues, connection=connection).work()


if __name__ == "__main__":
    main()
