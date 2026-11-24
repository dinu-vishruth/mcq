"""Background generation jobs: data access.

A job row is the contract between the web process (which enqueues), the worker
(which runs) and GET /api/jobs/{id} (which reports). Keep the status vocabulary
small and explicit: queued -> running -> done | failed.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from core.models.db import get_db

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now() -> str:
    return datetime.utcnow().isoformat()


def create(*, user_id: int, username: str, document_id: int, params: dict,
           total: int) -> str:
    job_id = uuid.uuid4().hex
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO generation_jobs "
            "(id, user_id, username, document_id, status, progress, total, params_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
            (job_id, user_id, username, document_id, STATUS_QUEUED, total,
             json.dumps(params), _now(), _now()),
        )
        conn.commit()
        return job_id
    finally:
        conn.close()


def get(job_id: str):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM generation_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        job = dict(row)
        if job["status"] in (STATUS_QUEUED, STATUS_RUNNING):
            params = json.loads(job.get("params_json") or "{}")
            last = datetime.fromisoformat(job["updated_at"]).replace(tzinfo=timezone.utc)
            limit = int(params.get("deadline_seconds") or 900) + 60
            if (datetime.now(timezone.utc) - last).total_seconds() > limit:
                failed = conn.execute("UPDATE generation_jobs SET status=?,error=?,updated_at=? "
                                      "WHERE id=? AND status=? AND updated_at=? RETURNING id",
                                      (STATUS_FAILED, "Generation worker expired. Please retry.", _now(),
                                       job_id, job["status"], job["updated_at"])).fetchone()
                conn.commit()
                if failed:
                    job.update(status=STATUS_FAILED, error="Generation worker expired. Please retry.")
        return job
    finally:
        conn.close()


def get_for_user(job_id: str, username: str):
    job = get(job_id)
    if job is None or job.get("username") != username:
        return None
    return job


def claim(job_id: str) -> bool:
    conn = get_db()
    try:
        row = conn.execute("UPDATE generation_jobs SET status=?, updated_at=? WHERE id=? AND status=? RETURNING id",
                           (STATUS_RUNNING, _now(), job_id, STATUS_QUEUED)).fetchone()
        conn.commit()
        return row is not None
    finally:
        conn.close()


def set_running(job_id: str) -> None:
    _update(job_id, status=STATUS_RUNNING)


def set_progress(job_id: str, progress: int) -> None:
    conn = get_db()
    try:
        conn.execute(
            "UPDATE generation_jobs SET progress=?, updated_at=? WHERE id=?",
            (progress, _now(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_done(job_id: str, result: dict) -> None:
    _update(job_id, status=STATUS_DONE, result_json=json.dumps(result),
            progress=result.get("count", 0))


def set_failed(job_id: str, error: str) -> None:
    from core.services.ai_safety import redact
    _update(job_id, status=STATUS_FAILED, error=redact(str(error))[:2000])


def _update(job_id: str, **fields) -> None:
    keys = list(fields)
    assignments = ", ".join(f"{k}=?" for k in keys)
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE generation_jobs SET {assignments}, updated_at=? WHERE id=?",
            (*(fields[k] for k in keys), _now(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def public_view(job: dict) -> dict:
    """Shape returned to the client. Job ids are unguessable and ownership is
    checked by the route, so exposing params/result is safe."""
    result = None
    if job.get("result_json"):
        try:
            result = json.loads(job["result_json"])
        except (TypeError, ValueError):
            result = None
    return {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job.get("progress") or 0,
        "total": job.get("total") or 0,
        "error": job.get("error"),
        "result": result,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }
