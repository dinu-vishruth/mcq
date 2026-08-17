"""JSON API consumed by the React frontend.

Port of core/routes/api.py. Every path, method and response body is unchanged so
the existing frontend code needs no edits. Pydantic models replace the manual
``request.get_json(silent=True) or {}`` + clamp dance: validation and coercion
are declared once on the model and FastAPI returns a 422 automatically.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..deps import CurrentApiUser, Db
from ..services.dashboard import (build_dashboard, level_for_xp, load_prefs,
                                  study_recommendations)

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
    document_id: int
    num_questions: int = Field(5, ge=1, le=30)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    topic: str | None = None
    #: Seconds. Upper bound matches the 180-minute limit on /upload.
    timer: int = Field(600, ge=30, le=10800)


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
# Achievements
# --------------------------------------------------------------------------- #

@router.get("/achievements")
async def achievements_data(user: CurrentApiUser, conn: Db):
    agg = conn.execute(
        "SELECT COUNT(*) quizzes, COALESCE(SUM(score),0) correct, "
        "COALESCE(SUM(total),0) answered, "
        "COALESCE(MAX(CASE WHEN total>0 THEN score*100/total ELSE 0 END),0) best "
        "FROM results WHERE user_id=?",
        (user.id,),
    ).fetchone()
    prefs = load_prefs(conn, user.id) or {}

    xp = prefs.get("xp") or (agg["correct"] * 10)
    streak = prefs.get("streak") or 0
    quizzes = agg["quizzes"] or 0
    best = round(agg["best"] or 0)

    badges = [
        {"key": "first_steps", "label": "First Steps", "desc": "Complete 1 quiz", "earned": quizzes >= 1, "icon": "footprints"},
        {"key": "on_fire", "label": "On Fire", "desc": "3-day streak", "earned": streak >= 3, "icon": "flame"},
        {"key": "dedicated", "label": "Dedicated", "desc": "7-day streak", "earned": streak >= 7, "icon": "calendar"},
        {"key": "sharpshooter", "label": "Sharpshooter", "desc": "Score 90%+", "earned": best >= 90, "icon": "target"},
        {"key": "xp_hunter", "label": "XP Hunter", "desc": "Earn 500 XP", "earned": xp >= 500, "icon": "zap"},
        {"key": "century", "label": "Centurion", "desc": "Complete 10 quizzes", "earned": quizzes >= 10, "icon": "trophy"},
    ]
    milestones = [
        {"label": "Quizzes completed", "value": quizzes, "next": 10 if quizzes < 10 else 25},
        {"label": "Current streak", "value": streak, "next": 7 if streak < 7 else 30},
        {"label": "Total XP", "value": xp, "next": ((xp // 500) + 1) * 500},
    ]
    return {
        "xp": xp, "level": level_for_xp(xp), "streak": streak, "quizzes": quizzes,
        "badges": badges, "milestones": milestones,
        "earned_count": sum(1 for b in badges if b["earned"]), "total_badges": len(badges),
    }


# --------------------------------------------------------------------------- #
# Practice generation
# --------------------------------------------------------------------------- #

@router.post("/practice/generate")
async def practice_generate(user: CurrentApiUser, payload: PracticeIn):
    """Generate an MCQ set from an indexed Knowledge document and create a session.

    Reuses the RAG/agent pipeline (falling back to legacy when a document has no
    retrievable context). Returns the session_key the Quiz screen loads via the
    existing /student_login flow.
    """
    from core.repositories import document_repo

    # Ownership: only practice from your own knowledge.
    doc = document_repo.get(payload.document_id)
    if doc is None or doc["owner"] != user.username:
        raise HTTPException(status_code=404, detail="not found")

    try:
        from core.services.mcq_pipeline import generate_from_document
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
    )
    return {"ok": True, "session_key": session_key, "count": len(mcqs)}
