"""
Student blueprint: dashboard (with analytics + AI weak-topics card),
session-key join, MCQ test rendering, and submission/scoring. Verbatim move
from app.py; url_for targets are blueprint-namespaced. Route paths unchanged.

explanation_engine is imported as a MODULE (not the function) so tests can patch
models.explanation_engine.explain_answers at the source and have it take effect.
"""
from __future__ import annotations

import json
import random
from datetime import datetime

from flask import (Blueprint, render_template, request, session, redirect, url_for)

from core.models.db import get_db
from utils.session_manager import validate_session_key
import models.explanation_engine as explanation_engine

student_bp = Blueprint("student", __name__)


@student_bp.route("/student")
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("auth.home"))

    conn = get_db()
    history_rows = conn.execute("""
        SELECT r.session_key, r.score, r.total, r.submitted_at, r.time_spent, s.difficulty
        FROM results r
        LEFT JOIN sessions s ON r.session_key = s.session_key
        WHERE r.user_id=?
        ORDER BY r.submitted_at DESC
    """, (session.get("user_id"),)).fetchall()
    conn.close()

    history = []
    total_quizzes = len(history_rows)
    total_score_pct = 0
    total_time = 0
    best_score_pct = 0

    difficulty_stats = {"easy": {"sum": 0, "count": 0}, "medium": {"sum": 0, "count": 0}, "hard": {"sum": 0, "count": 0}}

    chrono_history = list(reversed(history_rows))
    chart_dates = []
    chart_scores = []

    for row in history_rows:
        h_dict = dict(row)
        history.append(h_dict)

        pct = (row["score"] / row["total"] * 100) if (row["total"] and row["total"] > 0) else 0
        total_score_pct += pct
        total_time += row["time_spent"] or 0
        if pct > best_score_pct:
            best_score_pct = pct

        diff = (row["difficulty"] or "medium").lower()
        if diff in difficulty_stats:
            difficulty_stats[diff]["sum"] += pct
            difficulty_stats[diff]["count"] += 1

    for row in chrono_history:
        pct = (row["score"] / row["total"] * 100) if (row["total"] and row["total"] > 0) else 0
        dt = row["submitted_at"][:16].replace("T", " ")
        chart_dates.append(dt)
        chart_scores.append(round(pct, 1))

    avg_score = round(total_score_pct / total_quizzes, 1) if total_quizzes > 0 else 0
    best_score = round(best_score_pct, 1)

    avg_easy = round(difficulty_stats["easy"]["sum"] / difficulty_stats["easy"]["count"], 1) if difficulty_stats["easy"]["count"] > 0 else 0
    avg_medium = round(difficulty_stats["medium"]["sum"] / difficulty_stats["medium"]["count"], 1) if difficulty_stats["medium"]["count"] > 0 else 0
    avg_hard = round(difficulty_stats["hard"]["sum"] / difficulty_stats["hard"]["count"], 1) if difficulty_stats["hard"]["count"] > 0 else 0

    # Weak-topics (agentic learning intelligence). Best-effort: empty list on any
    # failure, and the template only renders the card when the list is non-empty.
    weak_topics = []
    try:
        from core.services.learning_service import get_weak_topics
        weak_topics = get_weak_topics(session.get("user_id"), limit=5)
    except Exception as e:
        print(f"[WARNING] Weak-topics lookup skipped: {e}")

    return render_template(
        "student_dashboard.html",
        history=history,
        total_quizzes=total_quizzes,
        avg_score=avg_score,
        best_score=best_score,
        total_time=total_time,
        chart_dates=chart_dates,
        chart_scores=chart_scores,
        difficulty_averages=[avg_easy, avg_medium, avg_hard],
        weak_topics=weak_topics
    )


@student_bp.route("/progress")
def progress():
    """Learning-progress page (React island). Additive read-only aggregation of
    the learner's existing results + weak topics + prefs. Does not touch or
    change any existing route; renders the progress.html mount shell."""
    if session.get("role") not in ("student", "teacher"):
        return redirect(url_for("auth.home"))

    user_id = session.get("user_id")
    conn = get_db()
    rows = conn.execute("""
        SELECT r.score, r.total, r.submitted_at, r.time_spent, s.difficulty
        FROM results r
        LEFT JOIN sessions s ON r.session_key = s.session_key
        WHERE r.user_id=?
        ORDER BY r.submitted_at ASC
    """, (user_id,)).fetchall()
    conn.close()

    total_quizzes = len(rows)
    total_correct = sum(r["score"] or 0 for r in rows)
    total_answered = sum(r["total"] or 0 for r in rows)
    total_time = sum(r["time_spent"] or 0 for r in rows)
    accuracy = round(total_correct / total_answered * 100, 1) if total_answered else 0

    # Daily activity heatmap: {YYYY-MM-DD: quiz_count}
    heatmap: dict[str, int] = {}
    weekly: dict[str, dict] = {}
    for r in rows:
        day = (r["submitted_at"] or "")[:10]
        if day:
            heatmap[day] = heatmap.get(day, 0) + 1
        week = (r["submitted_at"] or "")[:7]  # YYYY-MM bucket for the monthly bar
        if week:
            pct = (r["score"] / r["total"] * 100) if r["total"] else 0
            b = weekly.setdefault(week, {"sum": 0.0, "count": 0})
            b["sum"] += pct
            b["count"] += 1
    weekly_series = [
        {"label": k, "avg": round(v["sum"] / v["count"], 1) if v["count"] else 0, "count": v["count"]}
        for k, v in sorted(weekly.items())
    ]

    weak_topics = []
    try:
        from core.services.learning_service import get_weak_topics
        weak_topics = get_weak_topics(user_id, limit=10)
    except Exception as e:
        print(f"[WARNING] Progress weak-topics lookup skipped: {e}")

    prefs = None
    xp = 0
    streak = 0
    try:
        conn2 = get_db()
        prow = conn2.execute(
            "SELECT goal, style, daily_minutes, xp, streak FROM user_prefs WHERE user_id=?",
            (user_id,)).fetchone()
        conn2.close()
        if prow:
            prefs = {"goal": prow["goal"], "style": prow["style"], "daily_minutes": prow["daily_minutes"]}
            xp = prow["xp"] or 0
            streak = prow["streak"] or 0
    except Exception as e:
        print(f"[WARNING] Progress prefs lookup skipped: {e}")

    # XP heuristic when prefs.xp isn't populated: 10 XP per correct answer.
    if not xp:
        xp = total_correct * 10

    return render_template(
        "progress.html",
        total_quizzes=total_quizzes,
        accuracy=accuracy,
        total_time=total_time,
        total_correct=total_correct,
        total_answered=total_answered,
        heatmap=heatmap,
        weekly=weekly_series,
        weak_topics=weak_topics,
        prefs=prefs,
        xp=xp,
        streak=streak,
    )


@student_bp.route("/student_login", methods=["GET", "POST"])
def student_login():
    if session.get("role") != "student":
        return redirect(url_for("auth.home"))

    if request.method == "GET":
        return render_template("student_dashboard.html")

    key = request.form.get("session_key", "").strip()
    if not key.isalnum():
        return render_template("student_dashboard.html", error="Invalid session key format")

    if not validate_session_key(key):
        return render_template("student_dashboard.html", error="Invalid session key")

    conn = get_db()
    existing = conn.execute("SELECT 1 FROM results WHERE session_key=? AND user_id=?",
                            (key, session.get("user_id"))).fetchone()
    if existing:
        conn.close()
        return render_template("student_dashboard.html", error="You have already taken this test.")

    cur = conn.cursor()
    cur.execute("SELECT mcqs_json, timer FROM sessions WHERE session_key=?", (key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return render_template("student_dashboard.html", error="Session not found")

    mcqs = json.loads(row[0])
    session["mcqs"] = mcqs
    session["timer"] = row[1]
    session["session_key"] = key
    return redirect(url_for("student.mcq_test"))


@student_bp.route("/mcq_test", methods=["GET"])
def mcq_test():
    if session.get("role") != "student":
        return redirect(url_for("auth.home"))

    mcqs = session.get("mcqs")
    timer = session.get("timer", 60)
    key = session.get("session_key")
    if not mcqs or not key:
        return redirect(url_for("student.student_login"))

    conn = get_db()
    existing = conn.execute("SELECT 1 FROM results WHERE session_key=? AND user_id=?",
                            (key, session.get("user_id"))).fetchone()
    conn.close()
    if existing:
        return render_template("student_dashboard.html", error="You have already taken this test.")

    randomized = []
    for q in mcqs:
        opts = list(q["options"])
        random.shuffle(opts)
        for i, opt in enumerate(opts):
            opt["label"] = ["A", "B", "C", "D"][i]
        randomized.append({
            "question": q["question"],
            "options": opts,
            "answer_text": q["answer_text"],
        })

    session["mcqs_randomized"] = randomized
    return render_template("mcq_test.html", mcqs=randomized, timer=timer)


@student_bp.route("/submit", methods=["POST"])
def submit():
    if session.get("role") != "student":
        return redirect(url_for("auth.home"))

    randomized = session.get("mcqs_randomized")
    key = session.get("session_key")
    if not randomized or not key:
        return redirect(url_for("student.student_login"))

    conn = get_db()
    existing = conn.execute("SELECT 1 FROM results WHERE session_key=? AND user_id=?",
                            (key, session.get("user_id"))).fetchone()
    if existing:
        conn.close()
        return render_template("student_dashboard.html", error="You have already taken this test.")

    student_name = request.form.get("student_name", "").strip() or session.get("username", "Student")
    time_spent = 0
    try:
        time_spent = int(request.form.get("time_spent", 0))
    except ValueError:
        pass

    total = len(randomized)
    score = 0
    details = []

    for i, q in enumerate(randomized):
        sel = request.form.get(f"q-{i}", "")
        is_correct = sel == q["answer_text"]
        if is_correct:
            score += 1
        details.append({
            "question": q["question"],
            "selected": sel,
            "correct": q["answer_text"],
            "is_correct": is_correct,
        })

    # Generate explanations (module-level call so tests can patch the source).
    explanations = explanation_engine.explain_answers(details)

    conn.execute(
        "INSERT INTO results (session_key, student_name, user_id, score, total, submitted_at, detail_json, time_spent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (key, student_name, session.get("user_id"), score, total,
         datetime.utcnow().isoformat(), json.dumps(details), time_spent),
    )
    conn.commit()
    conn.close()

    # Learning analysis (weak-concept detection). Best-effort: wrapped so any
    # failure here can never affect the student's score or result page.
    try:
        conn2 = get_db()
        difficulty = "medium"
        drow = conn2.execute("SELECT difficulty FROM sessions WHERE session_key=?", (key,)).fetchone()
        if drow and drow["difficulty"]:
            difficulty = drow["difficulty"]
        conn2.close()
        from core.services.learning_service import analyse_submission
        analyse_submission(details, user_id=session.get("user_id"),
                           session_key=key, difficulty=difficulty)
    except Exception as e:
        print(f"[WARNING] Learning analysis skipped: {e}")

    return render_template(
        "result.html", score=score, total=total, details=details, explanations=explanations
    )
