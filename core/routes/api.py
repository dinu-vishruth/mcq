"""
Additive JSON API blueprint for the React frontend.

Every route here is NEW. Nothing in this module edits or replaces an existing
route, template, or table — it only reads existing data and reads/writes the
additive user_prefs table (see core/models/migrations.py). This mirrors the
codebase's "additive, never destructive" philosophy: the server-rendered pages
keep working untouched, and React screens layer on top.

All routes require an authenticated session (cookie auth is unchanged) and are
CSRF-protected for mutations via the shared Flask-WTF setup.
"""
from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request, session

from core.models.db import get_db

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _require_user():
    return session.get("user_id")


def _load_prefs(conn, user_id):
    row = conn.execute(
        "SELECT goal, style, daily_minutes, xp, streak, onboarded FROM user_prefs WHERE user_id=?",
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


@api_bp.route("/prefs", methods=["GET"])
def get_prefs():
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401
    conn = get_db()
    try:
        prefs = _load_prefs(conn, user_id)
    finally:
        conn.close()
    return jsonify({
        "prefs": prefs,
        "needs_onboarding": not (prefs and prefs.get("onboarded")),
    })


@api_bp.route("/prefs", methods=["POST"])
def save_prefs():
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    goal = str(data.get("goal", ""))[:64]
    style = str(data.get("style", ""))[:32]
    try:
        daily = int(data.get("daily_minutes", 30))
    except (TypeError, ValueError):
        daily = 30
    daily = max(5, min(600, daily))
    now = datetime.utcnow().isoformat()

    conn = get_db()
    try:
        exists = conn.execute("SELECT 1 FROM user_prefs WHERE user_id=?", (user_id,)).fetchone()
        if exists:
            conn.execute(
                "UPDATE user_prefs SET goal=?, style=?, daily_minutes=?, onboarded=1, updated_at=? WHERE user_id=?",
                (goal, style, daily, now, user_id),
            )
        else:
            conn.execute(
                "INSERT INTO user_prefs (user_id, goal, style, daily_minutes, onboarded, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (user_id, goal, style, daily, now),
            )
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True})


@api_bp.route("/export", methods=["GET"])
def export_data():
    """Return the authenticated user's own learning data as a JSON download.
    Read-only; scoped to the current user_id so no one can export another's
    data. Additive — nothing else depends on it."""
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    conn = get_db()
    try:
        user = conn.execute("SELECT username, email, created_at FROM users WHERE id=?", (user_id,)).fetchone()
        results = conn.execute(
            "SELECT session_key, score, total, submitted_at, time_spent FROM results WHERE user_id=? ORDER BY submitted_at",
            (user_id,)).fetchall()
        prefs = _load_prefs(conn, user_id)
        weak = conn.execute(
            "SELECT topic, wrong_count, total_count FROM weak_topics WHERE user_id=? ORDER BY wrong_count DESC",
            (user_id,)).fetchall()
    finally:
        conn.close()

    payload = {
        "account": dict(user) if user else {},
        "prefs": prefs,
        "results": [dict(r) for r in results],
        "weak_topics": [dict(w) for w in weak],
        "exported_at": datetime.utcnow().isoformat(),
    }
    resp = jsonify(payload)
    resp.headers.set("Content-Disposition", "attachment", filename="mcq_generator_data.json")
    return resp
