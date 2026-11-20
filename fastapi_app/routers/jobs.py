"""Job status endpoints for background generation.

GET /api/jobs/{id}         -> current status/result (JSON)
GET /api/jobs/{id}/events  -> server-sent events, one per progress change,
                              ending on done/failed. Polling the row means any
                              web instance can serve a stream, which matters on
                              serverless hosts where the enqueuing instance may
                              already be gone.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.repositories import job_repo

from ..deps import CurrentApiUser

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

#: Upper bound on one SSE connection (15 min), matching the default job
#: deadline. The client is expected to fall back to polling afterwards.
SSE_MAX_SECONDS = 900
SSE_POLL_SECONDS = 1.0


def _owned_job(job_id: str, username: str) -> dict:
    job = job_repo.get_for_user(job_id, username)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/{job_id}")
async def job_status(job_id: str, user: CurrentApiUser):
    return job_repo.public_view(_owned_job(job_id, user.username))


@router.get("/{job_id}/events")
async def job_events(job_id: str, user: CurrentApiUser):
    job = _owned_job(job_id, user.username)
    if job["status"] in (job_repo.STATUS_DONE, job_repo.STATUS_FAILED):
        # Already finished: one event and close, so a late subscriber still gets
        # the terminal state instead of hanging.
        async def one():
            yield f"data: {json.dumps(job_repo.public_view(job))}\n\n"

        return StreamingResponse(one(), media_type="text/event-stream")

    async def stream():
        last = None
        for _ in range(int(SSE_MAX_SECONDS / SSE_POLL_SECONDS)):
            current = job_repo.get(job_id)
            if current is None:
                break
            view = job_repo.public_view(current)
            payload = {"status": view["status"], "progress": view["progress"],
                       "total": view["total"], "error": view["error"]}
            if payload != last:
                yield f"data: {json.dumps(payload)}\n\n"
                last = payload
            if view["status"] in (job_repo.STATUS_DONE, job_repo.STATUS_FAILED):
                # Terminal event carries the result too.
                yield f"data: {json.dumps(view)}\n\n"
                break
            await asyncio.sleep(SSE_POLL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
