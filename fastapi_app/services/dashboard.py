"""The dashboard read model.

This is the fix for "showing the dashboard takes time".

Before: ``GET /dashboard`` server-rendered a page whose React island then fired a
second request to ``/api/dashboard`` and showed a spinner until it landed. Two
sequential round-trips, and on Vercel the second one can hit a cold start — so
the user stared at a spinner even though the server already had every number.

After: this one function builds the payload, ``/dashboard`` embeds it in
``data-bootstrap``, and the React screen renders from it on first paint. The
``/api/dashboard`` endpoint calls the same function, so it stays available for
refreshes and the two can't drift apart.
"""
from __future__ import annotations

import sqlite3
from typing import Any


def level_for_xp(xp: int) -> int:
    """100 XP per level, starting at level 1."""
    return xp // 100 + 1


def load_prefs(conn: sqlite3.Connection, user_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT goal, style, daily_minutes, xp, streak, onboarded FROM user_prefs WHERE user_id=?",
        (user_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def study_recommendations(weak: list[dict], prefs: dict | None, total_quizzes: int) -> list[dict]:
    """Proactive coach nudges derived from the user's own data."""
    recs: list[dict[str, str]] = []
    if weak:
        top = weak[0]
        recs.append({
            "kind": "revision",
            "title": f"Revise {top['topic']}",
            "reason": f"Your weakest topic — {top['pct']}% miss rate.",
            "cta": "Quick Revision",
        })
    if total_quizzes == 0:
        recs.append({
            "kind": "start",
            "title": "Take your first quiz",
            "reason": "Upload a document or save a resource, then make a quiz to build your baseline.",
            "cta": "Make Quiz & Test",
        })
    else:
        recs.append({
            "kind": "practice",
            "title": "Adaptive practice session",
            "reason": "Mix in fresh questions to keep concepts sharp.",
            "cta": "Start Practice",
        })
    if (prefs or {}).get("goal") == "interview":
        recs.append({
            "kind": "interview",
            "title": "Interview drill",
            "reason": "Your goal is interview prep — practice rapid recall.",
            "cta": "Interview Prep",
        })
    return recs


def _weak_topics(user_id: int, limit: int) -> list[dict]:
    """Best-effort weak-topic lookup: an empty list is a valid answer.

    The learning service reaches into the agentic layer, which can fail for
    reasons that have nothing to do with the dashboard. Never let that break the
    page — the UI simply omits the card.
    """
    try:
        from core.services.learning_service import get_weak_topics
        return get_weak_topics(user_id, limit=limit)
    except Exception as exc:
        print(f"[WARNING] weak-topics lookup skipped: {exc}")
        return []


def build_dashboard(conn: sqlite3.Connection, user_id: int, username: str) -> dict[str, Any]:
    """Everything the dashboard screen needs, in one pass over the results table."""
    rows = conn.execute(
        """
        SELECT r.session_key, r.score, r.total, r.submitted_at, r.time_spent, s.difficulty
        FROM results r LEFT JOIN sessions s ON r.session_key = s.session_key
        WHERE r.user_id=? ORDER BY r.submitted_at DESC
        """,
        (user_id,),
    ).fetchall()

    try:
        doc_count = conn.execute(
            "SELECT COUNT(*) c FROM documents WHERE owner=?", (username,)
        ).fetchone()["c"]
    except sqlite3.OperationalError:
        doc_count = 0  # documents table not yet migrated

    total_quizzes = len(rows)
    total_correct = sum(r["score"] or 0 for r in rows)
    total_answered = sum(r["total"] or 0 for r in rows)
    total_time = sum(r["time_spent"] or 0 for r in rows)

    def pct_of(row: sqlite3.Row) -> float:
        return (row["score"] / row["total"] * 100) if row["total"] else 0.0

    percentages = [pct_of(r) for r in rows]
    avg_score = round(sum(percentages) / total_quizzes, 1) if total_quizzes else 0
    best_score = round(max(percentages), 1) if percentages else 0

    # Per-difficulty averages, ordered [easy, medium, hard] for the chart.
    buckets: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    for row, pct in zip(rows, percentages):
        bucket = (row["difficulty"] or "medium").lower()
        if bucket in buckets:
            buckets[bucket].append(pct)
    difficulty_averages = [
        round(sum(v) / len(v), 1) if v else 0 for v in (buckets["easy"], buckets["medium"], buckets["hard"])
    ]

    chrono = list(reversed(rows))
    chart = [
        {"date": (r["submitted_at"] or "")[:10], "score": round(p, 1)}
        for r, p in zip(chrono[-14:], list(reversed(percentages))[-14:])
    ]
    chart_dates = [(r["submitted_at"] or "")[:16].replace("T", " ") for r in chrono]
    chart_scores = [round(p, 1) for p in reversed(percentages)]

    prefs = load_prefs(conn, user_id) or {}
    weak = _weak_topics(user_id, limit=5)

    # XP falls back to a 10-per-correct-answer heuristic when prefs aren't seeded.
    xp = prefs.get("xp") or (total_correct * 10)
    streak = prefs.get("streak") or 0

    recent = [
        {
            "session_key": r["session_key"], "score": r["score"], "total": r["total"],
            "submitted_at": r["submitted_at"], "difficulty": r["difficulty"],
            "pct": round(p),
        }
        for r, p in zip(rows[:5], percentages[:5])
    ]

    return {
        "username": username or "there",
        "total_quizzes": total_quizzes,
        "avg_score": avg_score,
        "best_score": best_score,
        "total_time": total_time,
        "total_correct": total_correct,
        "total_answered": total_answered,
        "streak": streak,
        "xp": xp,
        "level": level_for_xp(xp),
        "knowledge_count": doc_count,
        "goal": prefs.get("goal", ""),
        "daily_minutes": prefs.get("daily_minutes", 30),
        "chart": chart,
        "chart_dates": chart_dates,
        "chart_scores": chart_scores,
        "difficulty_averages": difficulty_averages,
        "weak_topics": weak,
        "history": [dict(r) for r in rows],
        "recent": recent,
        "recommendations": study_recommendations(weak, prefs, total_quizzes),
        "needs_onboarding": not prefs.get("onboarded"),
        "prefs": prefs or None,
    }
