"""JSON API consumed by the React frontend.

Port of core/routes/api.py. Every path, method and response body is unchanged so
the existing frontend code needs no edits. Pydantic models replace the manual
``request.get_json(silent=True) or {}`` + clamp dance: validation and coercion
are declared once on the model and FastAPI returns a 422 automatically.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from ..deps import CurrentApiUser, Db
from ..services.dashboard import build_dashboard, load_prefs

router = APIRouter(prefix="/api", tags=["api"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class PrefsIn(BaseModel):
    """Onboarding/settings payload. Bounds live here rather than in the handler."""
    goal: str = Field("", max_length=64)
    style: str = Field("", max_length=32)
    daily_minutes: int = Field(30, ge=5, le=600)


class ReviewIn(BaseModel):
    topic: str = Field(..., min_length=1)


class PracticeIn(BaseModel):
    document_id: int | None = Field(None, ge=1)
    upload_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")
    num_questions: int = Field(5, ge=1, le=30)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    topic: str | None = None
    #: Seconds. Upper bound matches the 180-minute limit on /upload.
    timer: int = Field(600, ge=30, le=10800)
    #: Optional per-job pipeline budget (queue mode only). Not sent by the
    #: current frontend; defaults to GENERATION_JOB_DEADLINE_SECONDS.
    deadline_seconds: int | None = Field(None, ge=60, le=3600)

    @model_validator(mode="after")
    def one_source(self):
        if (self.document_id is None) == (self.upload_id is None):
            raise ValueError("Choose exactly one saved resource or temporary upload.")
        return self


# --------------------------------------------------------------------------- #
# Preferences
# --------------------------------------------------------------------------- #

@router.get("/prefs")
async def get_prefs(user: CurrentApiUser, conn: Db):
    prefs = load_prefs(conn, user.id)
    return {"prefs": prefs, "needs_onboarding": not (prefs and prefs.get("onboarded"))}


@router.post("/prefs")
async def save_prefs(user: CurrentApiUser, conn: Db, payload: PrefsIn):
    now = datetime.utcnow().isoformat()
    exists = conn.execute("SELECT 1 FROM user_prefs WHERE user_id=?", (user.id,)).fetchone()
    if exists:
        conn.execute(
            "UPDATE user_prefs SET goal=?, style=?, daily_minutes=?, onboarded=1, updated_at=? "
            "WHERE user_id=?",
            (payload.goal, payload.style, payload.daily_minutes, now, user.id),
        )
    else:
        conn.execute(
            "INSERT INTO user_prefs (user_id, goal, style, daily_minutes, onboarded, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (user.id, payload.goal, payload.style, payload.daily_minutes, now),
        )
    conn.commit()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

@router.get("/dashboard")
async def dashboard_data(user: CurrentApiUser, conn: Db):
    """Same payload the /dashboard page already embeds.

    Kept so the React screen can refresh without a full reload; the page itself
    no longer needs to call this on load.
    """
    return build_dashboard(conn, user.id, user.username)


# --------------------------------------------------------------------------- #
# Data export
# --------------------------------------------------------------------------- #

@router.get("/export")
async def export_data(user: CurrentApiUser, conn: Db):
    """The user's own learning data as a JSON download, scoped to their user_id."""
    account = conn.execute(
        "SELECT username, email, created_at FROM users WHERE id=?", (user.id,)
    ).fetchone()
    results = conn.execute(
        "SELECT session_key, score, total, submitted_at, time_spent FROM results "
        "WHERE user_id=? ORDER BY submitted_at",
        (user.id,),
    ).fetchall()
    prefs = load_prefs(conn, user.id)
    try:
        weak = conn.execute(
            "SELECT topic, wrong_count, total_count FROM weak_topics WHERE user_id=? "
            "ORDER BY wrong_count DESC",
            (user.id,),
        ).fetchall()
    except sqlite3.OperationalError:
        weak = []

    return JSONResponse(
        {
            "account": dict(account) if account else {},
            "prefs": prefs,
            "results": [dict(r) for r in results],
            "weak_topics": [dict(w) for w in weak],
            "exported_at": datetime.utcnow().isoformat(),
        },
        headers={"Content-Disposition": 'attachment; filename="mcq_generator_data.json"'},
    )


# --------------------------------------------------------------------------- #
# Knowledge (saved resources)
# --------------------------------------------------------------------------- #

@router.get("/knowledge")
async def knowledge_list(user: CurrentApiUser):
    from core.repositories import document_repo

    items = []
    for doc in document_repo.list_by_owner(user.username):
        chunk_count = doc.get("chunk_count") or 0
        items.append({
            "id": doc["id"],
            "title": doc.get("title") or "Untitled",
            "subject": (doc.get("source_type") or "document").upper(),
            "created_at": doc.get("created_at"),
            "indexed": doc.get("status") == "ready",
            "status": doc.get("status"),
            "topic_count": chunk_count,
            # Estimate study time: ~1.5 min per chunk of material.
            "est_minutes": max(5, round(chunk_count * 1.5)),
        })
    return {"items": items}


@router.delete("/knowledge/{doc_id}")
async def knowledge_delete(doc_id: int, user: CurrentApiUser):
    from core.repositories import document_repo

    if not document_repo.delete(doc_id, user.username):
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Weak topics
# --------------------------------------------------------------------------- #

@router.get("/weak-topics")
async def weak_topics_data(user: CurrentApiUser):
    weak = []
    try:
        from core.services.learning_service import get_weak_topics
        weak = get_weak_topics(user.id, limit=20)
    except Exception as exc:
        print(f"[WARNING] weak-topics API skipped: {exc}")

    from core.repositories import learning_repo

    for item in weak:
        pct = item.get("pct", 0)
        item["severity"] = "high" if pct >= 60 else "medium" if pct >= 30 else "low"
        # How many missed questions we can actually rebuild into a review quiz.
        # The UI only offers "Review" when this is > 0, so no dead buttons.
        try:
            item["reviewable"] = len(
                learning_repo.missed_questions_for_topic(user.id, item["topic"], limit=30)
            )
        except Exception:
            item["reviewable"] = 0
    return {"items": weak}


@router.post("/weak-topics/review")
async def weak_topics_review(user: CurrentApiUser, payload: ReviewIn):
    """Rebuild a quiz from the ACTUAL questions the user missed for a topic.

    A genuine re-attempt of the exact questions gotten wrong — no regeneration,
    no LLM — so it works offline and in any pipeline mode. Returns a session_key
    the frontend hands to /student_login.
    """
    from core.repositories import learning_repo

    mcqs = learning_repo.missed_questions_for_topic(user.id, payload.topic.strip(), limit=20)
    if not mcqs:
        raise HTTPException(status_code=404, detail="No reviewable questions for this topic yet.")

    from utils.session_manager import create_session_key

    # Timer scales with the set (~45s/question), clamped to a sane range.
    session_key = create_session_key(
        teacher=user.username,
        difficulty="review",
        timer=max(60, min(1800, len(mcqs) * 45)),
        mcqs=mcqs,
    )
    return {"ok": True, "session_key": session_key, "count": len(mcqs)}


# --------------------------------------------------------------------------- #
# Review evidence
# --------------------------------------------------------------------------- #

@router.get("/review")
@router.get("/achievements", include_in_schema=False)
async def review_data(user: CurrentApiUser, conn: Db):
    from ..services.review import build_review
    data = build_review(conn, user.id, user.username)
    return {key: value for key, value in data.items() if key != "questions"}


@router.post("/review/start")
async def start_review(user: CurrentApiUser, conn: Db):
    from utils.session_manager import create_session_key

    from ..services.review import build_review
    questions = build_review(conn, user.id, user.username)["questions"]
    if not questions:
        raise HTTPException(404, "No saved questions are available to review yet.")
    key = create_session_key(teacher=user.username, difficulty="review",
                             timer=max(60, len(questions) * 60), mcqs=questions)
    return {"session_key": key, "count": len(questions)}


# --------------------------------------------------------------------------- #
# Practice generation
# --------------------------------------------------------------------------- #

@router.get("/generation/status")
def generation_status(user: CurrentApiUser):
    import config
    from core.services.generation_status import generation_available
    available = generation_available()
    return {"available": available,
            "message": "" if available else config.USER_FACING_AI_UNAVAILABLE}


@router.get("/quiz-uploads/{upload_id}")
def quiz_upload(upload_id: str, user: CurrentApiUser):
    from core.repositories import quiz_upload_repo
    source = quiz_upload_repo.get(upload_id, user.id)
    if source is None:
        raise HTTPException(status_code=404, detail="Quiz upload expired or was already used. Upload it again.")
    return {"id": source["id"], "title": source["title"]}


@router.post("/practice/generate")
def practice_generate(user: CurrentApiUser, payload: PracticeIn):
    """Generate an MCQ set from an indexed Knowledge document and create a session.

    Reuses the RAG/agent pipeline (falling back to legacy when a document has no
    retrievable context). Returns the session_key the Quiz screen loads via the
    existing /student_login flow.

    Deliberately sync rather than async: the pipeline blocks on ``requests.post``,
    and on a single-worker container an ``async def`` handler would hold the event
    loop for the whole generation -- including the health check, which the host
    reads as an unresponsive service. Sync handlers run in the threadpool.

    Queue mode (GENERATION_BACKEND=rq): the work is enqueued and this endpoint
    returns ``{"ok": true, "job_id": ..., "status": "queued"}`` instead of a
    session_key; the client polls GET /api/jobs/{id}. The default inline mode
    keeps the original response shape unchanged -- see README "Compatibility".
    """
    from core.repositories import document_repo, job_repo, quiz_upload_repo
    from core.services.generation_status import require_generation
    from core.services.personalization import learning_profile

    source = None
    if payload.upload_id:
        source = quiz_upload_repo.get(payload.upload_id, user.id)
        if source is None:
            raise HTTPException(status_code=404, detail="Quiz upload expired or was already used. Upload it again.")
    else:
        doc = document_repo.get(payload.document_id)
        if doc is None or doc["owner"] != user.username:
            raise HTTPException(status_code=404, detail="not found")

    require_generation()

    # Per-user throttle before any costly work (queue mode included: enqueuing is
    # cheap but the job still burns LLM calls). Uses its own connection because
    # this is a sync handler running in the threadpool -- see rate_limits.
    from ..rate_limits import check_generation_throttle_for_user

    throttle_message = check_generation_throttle_for_user(user.username)
    if throttle_message:
        raise HTTPException(status_code=429, detail=throttle_message)

    from core.services.job_runner import (
        create_job,
        enqueue_job,
        generation_backend,
        run_generation_job,
    )

    if source is not None:
        # One-off text never enters document ingestion, embeddings or Library.
        # Retain it on failure for retry, discard it once the quiz is stored.
        from core.services.mcq_pipeline import generate_from_text
        from utils.session_manager import create_session_key
        try:
            with learning_profile(user.id):
                mcqs = generate_from_text(source["content"], payload.num_questions,
                                          payload.difficulty, topic=payload.topic)
            if not mcqs:
                raise ValueError("No questions generated. Try different settings.")
            session_key = create_session_key(
                teacher=user.username, difficulty=payload.difficulty,
                timer=payload.timer, mcqs=mcqs,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        quiz_upload_repo.delete(payload.upload_id, user.id)
        return {"ok": True, "session_key": session_key, "count": len(mcqs)}

    if generation_backend() == "rq":
        job_id = create_job(
            user_id=user.id, username=user.username, document_id=payload.document_id,
            num_questions=payload.num_questions, difficulty=payload.difficulty,
            topic=(payload.topic or "").strip() or None, timer=payload.timer,
            deadline_seconds=payload.deadline_seconds,
        )
        if enqueue_job(job_id):
            return {"ok": True, "job_id": job_id, "status": "queued"}
        # Redis/queue hiccup: still do the work rather than fail the request.
        try:
            run_generation_job(job_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        job = job_repo.get(job_id) or {}
        result = job_repo.public_view(job).get("result") or {}
        return {"ok": True, "session_key": result.get("session_key"),
                "count": result.get("count", 0), "job_id": job_id}

    try:
        from core.services.mcq_pipeline import generate_from_document
        with learning_profile(user.id):
            mcqs = generate_from_document(
                payload.document_id,
                num_questions=payload.num_questions,
                difficulty=payload.difficulty,
                topic=(payload.topic or "").strip() or None,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if not mcqs:
        raise HTTPException(status_code=502, detail="No questions generated. Try different settings.")

    from utils.session_manager import create_session_key

    session_key = create_session_key(
        teacher=user.username, difficulty=payload.difficulty, timer=payload.timer, mcqs=mcqs,
        document_id=payload.document_id,
    )
    return {"ok": True, "session_key": session_key, "count": len(mcqs)}
