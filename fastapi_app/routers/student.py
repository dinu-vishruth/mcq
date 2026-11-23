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
import time
from contextlib import closing
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import config
import models.explanation_engine as explanation_engine
from core.llm.budget import remaining, time_budget
from core.models.db import get_db
from core.repositories import session_repo
from utils.session_manager import validate_session_key

from ..deps import CurrentApiUser, CurrentUser, Db
from ..services.dashboard import build_dashboard, load_prefs
from ..templating import render

router = APIRouter(tags=["student"])

#: Fallback quiz duration in seconds when a session has none recorded. Not 60 --
#: a missing value used to silently give every quiz a one-minute limit.
DEFAULT_TIMER_SECONDS = 600

_OPTION_LABELS = ("A", "B", "C", "D")
SUBMISSION_GRACE_SECONDS = 5  # Allow the automatic timeout form to reach the server.


def _dashboard_response(request, user, conn, error=None):
    data = build_dashboard(conn, user.id, user.username)
    data["error"] = error
    return render(request, "dashboard.html", bootstrap=data, error=error)


def _start_attempt(conn, key, user_id, duration):
    now = time.time()
    conn.execute(
        "INSERT INTO quiz_attempts (session_key, user_id, started_at, deadline) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (session_key, user_id) DO NOTHING",
        (key, user_id, now, now + max(1, duration or DEFAULT_TIMER_SECONDS)),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM quiz_attempts WHERE session_key=? AND user_id=?", (key, user_id)
    ).fetchone()


class QuizDraft(BaseModel):
    session_key: str
    answers: dict[str, str] = Field(default_factory=dict)
    confidence: dict[str, str] = Field(default_factory=dict)
    question_index: int = Field(default=0, ge=0)


def _clean_draft(questions, answers, confidence):
    clean_answers, clean_confidence = {}, {}
    for i, question in enumerate(questions):
        key = str(i)
        selected = answers.get(key)
        if selected in [option["text"] for option in question["options"]]:
            clean_answers[key] = selected
        if confidence.get(key) in ("know", "maybe", "no_idea"):
            clean_confidence[key] = confidence[key]
    return clean_answers, clean_confidence


@router.post("/api/quiz-attempt")
async def save_quiz_draft(draft: QuizDraft, user: CurrentApiUser, conn: Db):
    row = session_repo.get(draft.session_key)
    if not row or row["teacher_user_id"] != user.id:
        raise HTTPException(404, "Quiz not found")
    questions = json.loads(row["mcqs_json"])
    answers, confidence = _clean_draft(questions, draft.answers, draft.confidence)
    updated = conn.execute(
        "UPDATE quiz_attempts SET answers_json=?, confidence_json=?, question_index=? "
        "WHERE session_key=? AND user_id=? AND submitted=0 AND deadline>=? RETURNING user_id",
        (json.dumps(answers), json.dumps(confidence),
         min(draft.question_index, max(0, len(questions) - 1)),
         draft.session_key, user.id, time.time()),
    ).fetchone()
    conn.commit()
    if not updated:
        raise HTTPException(409, "Quiz has expired or is not active")
    return {"saved": True}


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
    if data["needs_onboarding"]:
        return RedirectResponse("/onboarding", status_code=302)
    # Embedded in data-bootstrap by the template, so the React island has
    # everything it needs on first paint -- no follow-up request, no spinner.
    return render(request, "dashboard.html", **data, bootstrap=data)


@router.get("/onboarding")
async def onboarding(request: Request, user: CurrentUser, conn: Db):
    prefs = load_prefs(conn, user.id)
    if prefs and prefs.get("onboarded"):
        return RedirectResponse("/dashboard", status_code=302)
    return render(request, "onboarding.html", prefs=prefs)


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
async def practice(request: Request, user: CurrentUser, doc: int | None = None):
    """Practice configuration. Optional ?doc=<id> preselects a source."""
    from core.repositories import document_repo
    selected = None
    error = ""
    if doc is not None:
        row = document_repo.get(doc)
        if row is None or row["owner_user_id"] != user.id:
            error = "This resource could not be loaded from your Library. Please return to the Library and try again."
        else:
            selected = {
                "id": row["id"], "title": row["title"],
                "indexed": row["status"] == "ready", "status": row["status"],
                "topic_count": row["chunk_count"] or 0,
            }
    return render(request, "practice.html", document_id=doc, selected_document=selected, error=error)


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
async def student_login_form(request: Request, user: CurrentUser, conn: Db):
    return _dashboard_response(request, user, conn)


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
        return _dashboard_response(request, user, conn, message)

    if not key.isalnum():
        return reject("Invalid session key format")
    if not validate_session_key(key):
        return reject("Invalid session key")
    quiz = session_repo.get(key)
    if quiz is None or quiz["teacher_user_id"] != user.id:
        return reject("Quiz not found in your account")
    if _already_attempted(conn, key, user.id):
        return RedirectResponse(f"/results/{key}", status_code=302)

    # Only the key goes in the cookie; mcqs + timer are re-read from the DB.
    request.session["session_key"] = key
    _start_attempt(conn, key, user.id, quiz["timer"])
    return RedirectResponse("/mcq_test", status_code=302)


@router.get("/mcq_test")
async def mcq_test(request: Request, user: CurrentUser, conn: Db):
    key = request.session.get("session_key")
    if not key:
        return RedirectResponse("/student_login", status_code=302)

    row = session_repo.get(key)
    if not row or row["teacher_user_id"] != user.id:
        return RedirectResponse("/student_login", status_code=302)
    if _already_attempted(conn, key, user.id):
        return RedirectResponse(f"/results/{key}", status_code=303)

    attempt = _start_attempt(conn, key, user.id, row["timer"])

    # Send only question and option text. Answers stay in the database until grading.
    randomized = []
    for question in json.loads(row["mcqs_json"]):
        options = [{"text": option["text"]} for option in question["options"]]
        random.shuffle(options)
        for index, option in enumerate(options):
            option["label"] = _OPTION_LABELS[index]
        randomized.append({
            "question": question["question"],
            "options": options,
        })

    now = time.time()
    return render(
        request, "mcq_test.html",
        mcqs=randomized, timer=max(0, attempt["deadline"] - now),
        deadline=attempt["deadline"], server_now=now, session_key=key,
        answers=json.loads(attempt["answers_json"]),
        confidence=json.loads(attempt["confidence_json"]),
        question_index=attempt["question_index"], user_id=user.id,
    )


@router.get("/results/{session_key}")
async def quiz_result(request: Request, session_key: str, user: CurrentUser, conn: Db):
    row = conn.execute(
        "SELECT score, total, detail_json, metadata_json, enrichment_status FROM results "
        "WHERE session_key=? AND user_id=? ORDER BY id LIMIT 1", (session_key, user.id),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Quiz result not found")
    details = json.loads(row["detail_json"])
    metadata = json.loads(row["metadata_json"] or "{}")
    explanations = [
        detail.get("explanation") or (
            f"Correct. The answer is {detail['correct']}." if detail["is_correct"] else
            f"{'Not answered' if not detail.get('selected') else 'Incorrect'}. "
            f"The correct answer is {detail['correct']}."
        ) for detail in details
    ]
    return render(request, "result.html", score=row["score"], total=row["total"],
                  details=details, explanations=explanations, xp_awarded=metadata.get("xp_awarded", 0),
                  session_key=session_key, enrichment_pending=row["enrichment_status"] in ("pending", "running"),
                  notice=("Time expired. Only answers saved before the deadline were scored."
                          if metadata.get("late") else None))


@router.post("/submit")
async def submit(request: Request, user: CurrentUser):
    form = await request.form()
    # Open the connection in the same worker that owns scoring and enrichment.
    # No SQLite connection is transferred across thread boundaries.
    return await run_in_threadpool(_submit_with_connection, request, user, form)


def _submit_with_connection(request, user, form):
    with closing(get_db()) as conn:
        return _submit(request, user, form, conn)


def _submit(request, user, form, conn):
    key = request.session.get("session_key")
    if not key:
        return RedirectResponse("/student_login", status_code=302)

    # Rebuild from the DB, not the cookie. Scoring compares the submitted option
    # text against answer_text -- both stable across the render-time shuffle --
    # so the stored mcqs are the source of truth.
    stored = session_repo.get(key)
    if not stored or stored["teacher_user_id"] != user.id:
        return RedirectResponse("/student_login", status_code=302)
    questions = json.loads(stored["mcqs_json"])

    if _already_attempted(conn, key, user.id):
        return RedirectResponse(f"/results/{key}", status_code=303)

    attempt = conn.execute(
        "SELECT * FROM quiz_attempts WHERE session_key=? AND user_id=?", (key, user.id)
    ).fetchone()
    if not attempt:
        return _dashboard_response(request, user, conn, "Please join the quiz before submitting.")

    student_name = (form.get("student_name") or "").strip() or user.username or "Student"
    now = time.time()
    time_spent = max(0, int(min(now, attempt["deadline"]) - attempt["started_at"]))
    # After the deadline only the last draft saved on time can be scored.
    late = now > attempt["deadline"] + SUBMISSION_GRACE_SECONDS
    saved_answers = json.loads(attempt["answers_json"])
    saved_confidence = json.loads(attempt["confidence_json"])

    details: list[dict[str, Any]] = []
    score = 0
    for index, question in enumerate(questions):
        selected = (saved_answers.get(str(index), "") if late
                    else form.get(f"q-{index}") or "")
        confidence = (saved_confidence.get(str(index)) if late
                      else form.get(f"confidence-{index}"))
        if confidence not in ("know", "maybe", "no_idea"):
            confidence = None
        is_correct = selected == question["answer_text"]
        score += int(is_correct)
        details.append({
            "question": question["question"],
            "selected": selected,
            "correct": question["answer_text"],
            "is_correct": is_correct,
            "confidence": confidence,
            **{field: question[field] for field in ("explanation", "sources", "verification") if field in question},
        })

    # Claim the attempt in the same transaction as the result, including when
    # two tabs submit at once. A failed insert rolls the claim back too.
    claimed = conn.execute(
        "UPDATE quiz_attempts SET submitted=1 WHERE session_key=? AND user_id=? AND submitted=0 "
        "RETURNING user_id",
        (key, user.id),
    ).fetchone()
    if not claimed:
        conn.rollback()
        return RedirectResponse(f"/results/{key}", status_code=303)

    conn.execute(
        "INSERT INTO results (session_key, student_name, user_id, score, total, "
        "submitted_at, detail_json, time_spent, metadata_json, enrichment_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
        (key, student_name, user.id, score, len(questions),
         datetime.utcnow().isoformat(), json.dumps(details), time_spent, json.dumps({"late": late})),
    )
    conn.commit()

    # Commit useful source-grounded feedback before optional provider work.
    # Generation metadata stays server-side until the quiz is graded.
    for detail in details:
        detail.setdefault("verification", "unverified")
        detail.setdefault("explanation", "Explanation is temporarily unavailable.")
    conn.execute("UPDATE results SET detail_json=? WHERE session_key=? AND user_id=?",
                 (json.dumps(details), key, user.id))
    conn.commit()
    # Gamification: 10 XP per correct answer + 5 completion bonus, and advance
    # the daily streak. Best-effort -- never affects scoring.
    xp_awarded = 0
    try:
        from core.repositories import prefs_repo
        prefs_repo.record_activity(user.id, xp_gain=score * 10 + 5)
        xp_awarded = score * 10 + 5
    except Exception as exc:
        print(f"[WARNING] XP/streak update skipped: {exc}")

    conn.execute("UPDATE results SET metadata_json=? WHERE session_key=? AND user_id=?",
                 (json.dumps({"late": late, "xp_awarded": xp_awarded}), key, user.id))
    conn.commit()

    return RedirectResponse(f"/results/{key}", status_code=303)


def _feedback_payload(row):
    details = json.loads(row["detail_json"])
    return {"details": details,
            "explanations": [detail.get("explanation") or "Explanation is temporarily unavailable." for detail in details],
            "pending": row["enrichment_status"] in ("pending", "running"),
            "failed": row["enrichment_status"] == "failed"}


@router.post("/api/results/{session_key}/enrich")
def enrich_result(session_key: str, user: CurrentApiUser):
    """Optional feedback starts only after the durable score page is visible.

    A conditional claim prevents double billing and repeated learning analysis
    across reloads/tabs. Crashed claims expire to a visible failure, not an
    uncontrolled automatic replay of partially completed analysis.
    """
    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM results WHERE session_key=? AND user_id=? ORDER BY id LIMIT 1",
                           (session_key, user.id)).fetchone()
        if row is None:
            raise HTTPException(404, "Quiz result not found")
        if row["enrichment_status"] == "running" and (row["enrichment_deadline"] or 0) <= time.time():
            conn.execute("UPDATE results SET enrichment_status='failed' WHERE id=? AND enrichment_status='running'",
                         (row["id"],))
            conn.commit()
            row = conn.execute("SELECT * FROM results WHERE id=?", (row["id"],)).fetchone()
        if row["enrichment_status"] != "pending":
            return _feedback_payload(row)
        claimed = conn.execute(
            "UPDATE results SET enrichment_status='running', enrichment_deadline=? "
            "WHERE id=? AND enrichment_status='pending' RETURNING id",
            (time.time() + max(1, config.RESULT_ENRICHMENT_DEADLINE_SECONDS) + 30, row["id"]),
        ).fetchone()
        conn.commit()
        if not claimed:
            return _feedback_payload(conn.execute("SELECT * FROM results WHERE id=?", (row["id"],)).fetchone())
        try:
            _enrich_feedback(conn, row, user, session_key)
            conn.execute("UPDATE results SET enrichment_status='done' WHERE id=?", (row["id"],))
            conn.commit()
        except Exception:
            conn.rollback()
            conn.execute("UPDATE results SET enrichment_status='failed' WHERE id=?", (row["id"],))
            conn.commit()
        return _feedback_payload(conn.execute("SELECT * FROM results WHERE id=?", (row["id"],)).fetchone())


def _enrich_feedback(conn, row, user, key):
    details = json.loads(row["detail_json"])
    stored = session_repo.get(key)
    # Old shared assessments may have a different source owner. Do not retrieve
    # someone else's Library context while enriching a historical result.
    owned = stored is not None and stored["teacher_user_id"] == user.id
    document_id = stored["document_id"] if owned else None
    difficulty = stored["difficulty"] if stored is not None else "medium"
    from core.services.personalization import learning_profile
    with time_budget(config.RESULT_ENRICHMENT_DEADLINE_SECONDS):
        missing = [detail for detail in details if detail["explanation"] == "Explanation is temporarily unavailable."]
        if missing:
            try:
                remaining()
                with learning_profile(user.id):
                    explanations = (explanation_engine.explain_answers(missing, document_id=document_id)
                                    if document_id else explanation_engine.explain_answers(missing))
                for detail, explanation in zip(missing, explanations):
                    if isinstance(explanation, str) and explanation.strip():
                        detail["explanation"] = explanation
            except Exception:
                pass  # The scored result and generated explanations already exist.
        conn.execute("UPDATE results SET detail_json=? WHERE session_key=? AND user_id=?",
                     (json.dumps(details), key, user.id))
        conn.commit()
        try:
            remaining()
            from core.services.learning_service import analyse_submission
            analyse_submission(details, user_id=user.id, session_key=key,
                               difficulty=difficulty)
        except Exception as exc:
            print(f"[WARNING] Learning analysis skipped: {type(exc).__name__}")
