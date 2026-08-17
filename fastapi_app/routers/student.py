"""Learner-facing pages: dashboard, progress, library, and the quiz flow.

Port of core/routes/student.py. The one behavioural change is on ``/dashboard``:
it now embeds the full dashboard payload so the React screen paints immediately
instead of showing a spinner while it fetches ``/api/dashboard``. See
services/dashboard.py for the reasoning.

Session-cookie discipline (carried over from the CSRF fix): only ``session_key``
is stored in the cookie. The questions themselves are re-read from the database,
because a full MCQ list would blow past the ~4KB cookie limit.
"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

import models.explanation_engine as explanation_engine
from core.repositories import session_repo
from utils.session_manager import validate_session_key

from ..deps import CurrentUser, Db
from ..services.dashboard import build_dashboard, load_prefs
from ..templating import render

router = APIRouter(tags=["student"])

#: Fallback quiz duration in seconds when a session has none recorded. Not 60 --
#: a missing value used to silently give every quiz a one-minute limit.
DEFAULT_TIMER_SECONDS = 600

_OPTION_LABELS = ("A", "B", "C", "D")


def _already_attempted(conn: sqlite3.Connection, key: str, user_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM results WHERE session_key=? AND user_id=?", (key, user_id)
    ).fetchone() is not None


@router.get("/student")
async def student_alias():
    """Legacy alias -> canonical /dashboard."""
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/dashboard")
async def dashboard(request: Request, user: CurrentUser, conn: Db):
    data = build_dashboard(conn, user.id, user.username)
    # Embedded in data-bootstrap by the template, so the React island has
    # everything it needs on first paint -- no follow-up request, no spinner.
    return render(request, "student_dashboard.html", **data, bootstrap=data)


@router.get("/progress")
async def progress(request: Request, user: CurrentUser, conn: Db):
    """Read-only aggregation of results + weak topics + prefs."""
    rows = conn.execute(
        """
        SELECT r.score, r.total, r.submitted_at, r.time_spent, s.difficulty
        FROM results r LEFT JOIN sessions s ON r.session_key = s.session_key
        WHERE r.user_id=? ORDER BY r.submitted_at ASC
        """,
        (user.id,),
    ).fetchall()

    total_quizzes = len(rows)
    total_correct = sum(r["score"] or 0 for r in rows)
    total_answered = sum(r["total"] or 0 for r in rows)
    total_time = sum(r["time_spent"] or 0 for r in rows)
    accuracy = round(total_correct / total_answered * 100, 1) if total_answered else 0

    # Daily activity heatmap {YYYY-MM-DD: quiz_count} + monthly average bars.
    heatmap: dict[str, int] = {}
    monthly: dict[str, dict[str, float]] = {}
    for row in rows:
        submitted = row["submitted_at"] or ""
        day, month = submitted[:10], submitted[:7]
        if day:
            heatmap[day] = heatmap.get(day, 0) + 1
        if month:
            bucket = monthly.setdefault(month, {"sum": 0.0, "count": 0})
            bucket["sum"] += (row["score"] / row["total"] * 100) if row["total"] else 0
            bucket["count"] += 1
    weekly_series = [
        {
            "label": key,
            "avg": round(val["sum"] / val["count"], 1) if val["count"] else 0,
            "count": val["count"],
        }
        for key, val in sorted(monthly.items())
    ]

    weak_topics: list[dict] = []
    try:
        from core.services.learning_service import get_weak_topics
        weak_topics = get_weak_topics(user.id, limit=10)
    except Exception as exc:
        print(f"[WARNING] Progress weak-topics lookup skipped: {exc}")

    stored = load_prefs(conn, user.id) or {}
    prefs = (
        {
            "goal": stored.get("goal"),
            "style": stored.get("style"),
            "daily_minutes": stored.get("daily_minutes"),
        }
        if stored
        else None
    )
    # XP heuristic when prefs.xp isn't populated: 10 XP per correct answer.
    xp = stored.get("xp") or total_correct * 10
    streak = stored.get("streak") or 0

    return render(
        request, "progress.html",
        total_quizzes=total_quizzes, accuracy=accuracy, total_time=total_time,
        total_correct=total_correct, total_answered=total_answered,
        heatmap=heatmap, weekly=weekly_series, weak_topics=weak_topics,
        prefs=prefs, xp=xp, streak=streak,
    )


# --- React islands that fetch their own data from /api/* -------------------- #

@router.get("/knowledge")
async def knowledge(request: Request, user: CurrentUser):
    """Manages the user's uploaded study resources (data from /api/knowledge)."""
    return render(request, "knowledge.html")


@router.get("/journey")
async def journey(request: Request, user: CurrentUser):
    """Learning Journey: the saved-resource library. Generates nothing itself."""
    return render(request, "journey.html")


@router.get("/add_resource")
async def add_resource(request: Request, user: CurrentUser):
    """Store-only upload page. Posts to /ingest_resource; never makes a quiz."""
    return render(request, "add_resource.html")


@router.get("/practice")
async def practice(request: Request, user: CurrentUser, doc: str | None = None):
    """Practice configuration. Optional ?doc=<id> preselects a source."""
    return render(request, "practice.html", document_id=doc)


@router.get("/weak-topics")
async def weak_topics_page(request: Request, user: CurrentUser):
    """AI-tracked weak concepts with revision recommendations."""
    return render(request, "weak_topics.html")


@router.get("/achievements")
async def achievements(request: Request, user: CurrentUser):
    """Streak, XP, badges, milestones (data from /api/achievements)."""
    return render(request, "achievements.html")


# --- Quiz flow -------------------------------------------------------------- #

@router.get("/student_login")
async def student_login_form(request: Request, user: CurrentUser):
    return render(request, "student_dashboard.html")


@router.post("/student_login")
async def student_login(
    request: Request,
    user: CurrentUser,
    conn: Db,
    session_key: Annotated[str, Form()] = "",
):
    """Join a quiz by its session key."""
    key = session_key.strip()

    def reject(message: str):
        return render(request, "student_dashboard.html", error=message)

    if not key.isalnum():
        return reject("Invalid session key format")
    if not validate_session_key(key):
        return reject("Invalid session key")
    if _already_attempted(conn, key, user.id):
        return reject("You have already taken this test.")
    if conn.execute(
        "SELECT 1 FROM sessions WHERE session_key=?", (key,)
    ).fetchone() is None:
        return reject("Session not found")

    # Only the key goes in the cookie; mcqs + timer are re-read from the DB.
    request.session["session_key"] = key
    return RedirectResponse("/mcq_test", status_code=302)


@router.get("/mcq_test")
async def mcq_test(request: Request, user: CurrentUser, conn: Db):
    key = request.session.get("session_key")
    if not key:
        return RedirectResponse("/student_login", status_code=302)

    row = session_repo.get(key)
    if not row:
        return RedirectResponse("/student_login", status_code=302)
    if _already_attempted(conn, key, user.id):
        return render(request, "student_dashboard.html", error="You have already taken this test.")

    # Shuffle option ORDER only. Question order and answer_text are untouched,
    # which is what lets /submit rebuild the quiz from the DB and still score it.
    randomized = []
    for question in json.loads(row["mcqs_json"]):
        options = list(question["options"])
        random.shuffle(options)
        for index, option in enumerate(options):
            option["label"] = _OPTION_LABELS[index]
        randomized.append({
            "question": question["question"],
            "options": options,
            "answer_text": question["answer_text"],
        })

    return render(
        request, "mcq_test.html",
        mcqs=randomized, timer=row["timer"] or DEFAULT_TIMER_SECONDS,
    )


@router.post("/submit")
async def submit(request: Request, user: CurrentUser, conn: Db):
    key = request.session.get("session_key")
    if not key:
        return RedirectResponse("/student_login", status_code=302)

    # Rebuild from the DB, not the cookie. Scoring compares the submitted option
    # text against answer_text -- both stable across the render-time shuffle --
    # so the stored mcqs are the source of truth.
    stored = session_repo.get(key)
    if not stored:
        return RedirectResponse("/student_login", status_code=302)
    questions = [
        {"question": q["question"], "answer_text": q["answer_text"]}
        for q in json.loads(stored["mcqs_json"])
    ]

    if _already_attempted(conn, key, user.id):
        return render(request, "student_dashboard.html", error="You have already taken this test.")

    form = await request.form()
    student_name = (form.get("student_name") or "").strip() or user.username or "Student"
    try:
        time_spent = int(form.get("time_spent") or 0)
    except (TypeError, ValueError):
        time_spent = 0

    details: list[dict[str, Any]] = []
    score = 0
    for index, question in enumerate(questions):
        selected = form.get(f"q-{index}") or ""
        is_correct = selected == question["answer_text"]
        score += int(is_correct)
        details.append({
            "question": question["question"],
            "selected": selected,
            "correct": question["answer_text"],
            "is_correct": is_correct,
        })

    # Module-level call so tests can patch the source.
    explanations = explanation_engine.explain_answers(details)

    conn.execute(
        "INSERT INTO results (session_key, student_name, user_id, score, total, "
        "submitted_at, detail_json, time_spent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (key, student_name, user.id, score, len(questions),
         datetime.utcnow().isoformat(), json.dumps(details), time_spent),
    )
    conn.commit()

    # Weak-concept detection. Best-effort: wrapped so any failure here can never
    # affect the student's score or result page.
    try:
        difficulty_row = conn.execute(
            "SELECT difficulty FROM sessions WHERE session_key=?", (key,)
        ).fetchone()
        from core.services.learning_service import analyse_submission
        analyse_submission(
            details, user_id=user.id, session_key=key,
            difficulty=(difficulty_row["difficulty"] if difficulty_row else None) or "medium",
        )
    except Exception as exc:
        print(f"[WARNING] Learning analysis skipped: {exc}")

    # Gamification: 10 XP per correct answer + 5 completion bonus, and advance
    # the daily streak. Best-effort -- never affects scoring.
    try:
        from core.repositories import prefs_repo
        prefs_repo.record_activity(user.id, xp_gain=score * 10 + 5)
    except Exception as exc:
        print(f"[WARNING] XP/streak update skipped: {exc}")

    return render(
        request, "result.html",
        score=score, total=len(questions), details=details, explanations=explanations,
    )
