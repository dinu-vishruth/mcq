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
            now = datetime.now(timezone.utc)
            if (now - last).total_seconds() > limit:
                failed = conn.execute("UPDATE generation_jobs SET status=?,error=?,updated_at=? "
                                      "WHERE id=? AND status=? AND updated_at=? RETURNING id",
                                      (STATUS_FAILED, "Generation worker expired. Please retry.", _now(),
                                       job_id, job["status"], job["updated_at"])).fetchone()
                conn.commit()
                if failed:
                    job.update(status=STATUS_FAILED, error="Generation worker expired. Please retry.")
            elif job["status"] == STATUS_RUNNING and job.get("heartbeat_at"):
                heartbeat = datetime.fromisoformat(job["heartbeat_at"]).replace(tzinfo=timezone.utc)
                started = datetime.fromisoformat(job["started_at"]).replace(tzinfo=timezone.utc)
                if (now - heartbeat).total_seconds() > 90 or (now - started).total_seconds() > limit:
                    status = STATUS_QUEUED if job["attempt_count"] < 2 else STATUS_FAILED
                    recovered = conn.execute("UPDATE generation_jobs SET status=?,run_token=NULL,progress=0,error=?,updated_at=? "
                                             "WHERE id=? AND status=? AND run_token=? RETURNING id",
                                             (status, "Worker stopped. Retry queued." if status == STATUS_QUEUED else "Worker failed repeatedly. Please retry.",
                                              _now(), job_id, STATUS_RUNNING, job["run_token"])).fetchone()
                    conn.commit()
                    if recovered:
                        job.update(status=status, run_token=None, progress=0, recovered=status == STATUS_QUEUED)
        return job
    finally:
        conn.close()


def get_for_user(job_id: str, username: str):
    job = get(job_id)
    if job is None or job.get("username") != username:
        return None
    return job


def claim(job_id: str, token: str | None = None) -> bool:
    conn = get_db()
    try:
        row = conn.execute("UPDATE generation_jobs SET status=?, updated_at=?,heartbeat_at=?,started_at=?,"
                           "run_token=?,attempt_count=attempt_count+1,error=NULL WHERE id=? AND status=? RETURNING id",
                           (STATUS_RUNNING, _now(), _now(), _now(), token or uuid.uuid4().hex, job_id, STATUS_QUEUED)).fetchone()
        conn.commit()
        return row is not None
    finally:
        conn.close()


def heartbeat(job_id, token):
    conn = get_db()
    try:
        conn.execute("UPDATE generation_jobs SET heartbeat_at=? WHERE id=? AND run_token=? AND status=?",
                     (_now(), job_id, token, STATUS_RUNNING))
        conn.commit()
    finally:
        conn.close()


def publish(job, token, mcqs, difficulty, timer, metrics):
    """Commit one session and the terminal job result in the same transaction."""
    conn = get_db()
    try:
        owned = conn.execute("UPDATE generation_jobs SET updated_at=? WHERE id=? AND run_token=? AND status=? RETURNING id",
                             (_now(), job["id"], token, STATUS_RUNNING)).fetchone()
        document = conn.execute("SELECT owner FROM documents WHERE id=?", (job["document_id"],)).fetchone()
        if owned is None or document is None or document["owner"] != job["username"]:
            raise ValueError("Job ownership changed or its worker lease expired")
        key = uuid.uuid4().hex[:16]
        conn.execute("INSERT INTO sessions(session_key,teacher,created_at,difficulty,timer,mcqs_json,document_id) VALUES(?,?,?,?,?,?,?)",
                     (key, job["username"], _now(), difficulty, timer, json.dumps(mcqs), job["document_id"]))
        result = metrics.result(key, count=len(mcqs), requested=job["total"])
        conn.execute("UPDATE generation_jobs SET status=?,result_json=?,progress=?,updated_at=? WHERE id=? AND run_token=?",
                     (STATUS_DONE, json.dumps(result), len(mcqs), _now(), job["id"], token))
        conn.commit()
        return key
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_running(job_id: str) -> None:
    _update(job_id, status=STATUS_RUNNING)


def set_progress(job_id: str, progress: int, token=None) -> None:
    conn = get_db()
    try:
        sql = "UPDATE generation_jobs SET progress=?, updated_at=? WHERE id=?"
        params = (progress, _now(), job_id)
        if token:
            sql += " AND run_token=? AND status=?"
            params += (token, STATUS_RUNNING)
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def defer_busy(job_id, token):
    conn = get_db()
    try:
        conn.execute("UPDATE generation_jobs SET status=?,run_token=NULL,attempt_count=attempt_count-1,updated_at=? "
                     "WHERE id=? AND run_token=? AND status=?",
                     (STATUS_QUEUED, _now(), job_id, token, STATUS_RUNNING))
        conn.commit()
    finally:
        conn.close()


def set_done(job_id: str, result: dict) -> None:
    _update(job_id, status=STATUS_DONE, result_json=json.dumps(result),
            progress=result.get("count", 0))


def set_failed(job_id: str, error: str, token=None) -> None:
    from core.services.ai_safety import redact
    if token:
        conn = get_db()
        try:
            conn.execute("UPDATE generation_jobs SET status=?,error=?,updated_at=? WHERE id=? AND run_token=? AND status=?",
                         (STATUS_FAILED, redact(str(error))[:2000], _now(), job_id, token, STATUS_RUNNING))
            conn.commit()
        finally:
            conn.close()
        return
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
    """Expose progress and completion metadata, including for older saved jobs."""
    result = None
    if job.get("result_json"):
        try:
            stored = json.loads(job["result_json"])
            if isinstance(stored, dict):
                fields = ("session_key", "count", "requested_count", "partial", "quality_policy",
                          "fallback_attempted", "fallback_count")
                result = {key: stored[key] for key in fields if key in stored}
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
