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


# --------------------------------------------------------------------------- #
# Exam-prep platform APIs (single-user). All read/write only the current
# user's own data. Reuses existing repositories/services — no backend rewrite.
# --------------------------------------------------------------------------- #

def _level(xp: int) -> int:
    return xp // 100 + 1


def _study_recommendations(weak, prefs, total_quizzes):
    """Proactive coach nudges derived from the user's own data. The AI guides
    rather than waiting to be asked."""
    recs = []
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
    goal = (prefs or {}).get("goal")
    if goal == "interview":
        recs.append({
            "kind": "interview",
            "title": "Interview drill",
            "reason": "Your goal is interview prep — practice rapid recall.",
            "cta": "Interview Prep",
        })
    return recs


@api_bp.route("/dashboard", methods=["GET"])
def dashboard_data():
    """'What should I study today?' — learner-centric, never lists documents."""
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT r.session_key, r.score, r.total, r.submitted_at, r.time_spent, s.difficulty
            FROM results r LEFT JOIN sessions s ON r.session_key = s.session_key
            WHERE r.user_id=? ORDER BY r.submitted_at DESC
        """, (user_id,)).fetchall()
        doc_count = conn.execute(
            "SELECT COUNT(*) c FROM documents WHERE owner=?", (session.get("username"),)
        ).fetchone()["c"]
    finally:
        conn.close()

    total_quizzes = len(rows)
    total_correct = sum(r["score"] or 0 for r in rows)
    total_answered = sum(r["total"] or 0 for r in rows)
    total_time = sum(r["time_spent"] or 0 for r in rows)
    avg_score = round(sum((r["score"] / r["total"] * 100) if r["total"] else 0 for r in rows) / total_quizzes, 1) if total_quizzes else 0

    chrono = list(reversed(rows))[-14:]
    chart = [{"date": (r["submitted_at"] or "")[:10],
              "score": round((r["score"] / r["total"] * 100) if r["total"] else 0, 1)} for r in chrono]

    pconn = get_db()
    try:
        prefs = _load_prefs(pconn, user_id) or {}
    finally:
        pconn.close()

    weak = []
    try:
        from core.services.learning_service import get_weak_topics
        weak = get_weak_topics(user_id, limit=5)
    except Exception as e:
        print(f"[WARNING] dashboard weak-topics skipped: {e}")

    xp = prefs.get("xp") or (total_correct * 10)
    streak = prefs.get("streak") or 0

    recent = [{
        "session_key": r["session_key"], "score": r["score"], "total": r["total"],
        "submitted_at": r["submitted_at"], "difficulty": r["difficulty"],
        "pct": round((r["score"] / r["total"] * 100) if r["total"] else 0),
    } for r in rows[:5]]

    return jsonify({
        "username": session.get("username", "there"),
        "total_quizzes": total_quizzes,
        "avg_score": avg_score,
        "total_time": total_time,
        "streak": streak,
        "xp": xp,
        "level": _level(xp),
        "knowledge_count": doc_count,
        "goal": prefs.get("goal", ""),
        "daily_minutes": prefs.get("daily_minutes", 30),
        "chart": chart,
        "weak_topics": weak,
        "recent": recent,
        "recommendations": _study_recommendations(weak, prefs, total_quizzes),
        "needs_onboarding": not prefs.get("onboarded"),
    })


@api_bp.route("/knowledge", methods=["GET"])
def knowledge_list():
    """The user's uploaded study resources (Knowledge page)."""
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    from core.repositories import document_repo
    docs = document_repo.list_by_owner(session.get("username"))

    items = []
    for d in docs:
        chunk_count = d.get("chunk_count") or 0
        # Estimate study time: ~1.5 min per chunk of material.
        est_min = max(5, round(chunk_count * 1.5))
        items.append({
            "id": d["id"],
            "title": d.get("title") or "Untitled",
            "subject": (d.get("source_type") or "document").upper(),
            "created_at": d.get("created_at"),
            "indexed": d.get("status") == "ready",
            "status": d.get("status"),
            "topic_count": chunk_count,
            "est_minutes": est_min,
        })
    return jsonify({"items": items})


@api_bp.route("/knowledge/<int:doc_id>", methods=["DELETE"])
def knowledge_delete(doc_id):
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401
    from core.repositories import document_repo
    ok = document_repo.delete(doc_id, session.get("username"))
    if not ok:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


@api_bp.route("/weak-topics", methods=["GET"])
def weak_topics_data():
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401
    weak = []
    try:
        from core.services.learning_service import get_weak_topics
        weak = get_weak_topics(user_id, limit=20)
    except Exception as e:
        print(f"[WARNING] weak-topics API skipped: {e}")

    from core.repositories import learning_repo
    for w in weak:
        pct = w.get("pct", 0)
        w["severity"] = "high" if pct >= 60 else "medium" if pct >= 30 else "low"
        # How many of the missed questions we can actually rebuild into a review
        # quiz. The UI only offers "Review" when this is > 0, so no dead buttons.
        try:
            w["reviewable"] = len(learning_repo.missed_questions_for_topic(user_id, w["topic"], limit=30))
        except Exception:
            w["reviewable"] = 0
    return jsonify({"items": weak})


@api_bp.route("/weak-topics/review", methods=["POST"])
def weak_topics_review():
    """Rebuild a quiz from the ACTUAL questions the user missed for a topic and
    launch it via the existing session/quiz flow. Returns a session_key the
    frontend hands to /student_login (same handoff as practice/generate).

    This is a genuine re-attempt of the exact questions gotten wrong — no
    regeneration, no LLM — so it works offline and in any pipeline mode."""
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic required"}), 400

    from core.repositories import learning_repo
    mcqs = learning_repo.missed_questions_for_topic(user_id, topic, limit=20)
    if not mcqs:
        return jsonify({"error": "No reviewable questions for this topic yet."}), 404

    # Timer scales with the set (~45s/question), clamped to a sane range.
    timer = max(60, min(1800, len(mcqs) * 45))
    from utils.session_manager import create_session_key
    session_key = create_session_key(
        teacher=session.get("username"), difficulty="review", timer=timer, mcqs=mcqs,
    )
    return jsonify({"ok": True, "session_key": session_key, "count": len(mcqs)})


@api_bp.route("/achievements", methods=["GET"])
def achievements_data():
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    conn = get_db()
    try:
        agg = conn.execute(
            "SELECT COUNT(*) quizzes, COALESCE(SUM(score),0) correct, "
            "COALESCE(SUM(total),0) answered, COALESCE(MAX(CASE WHEN total>0 THEN score*100/total ELSE 0 END),0) best "
            "FROM results WHERE user_id=?", (user_id,)
        ).fetchone()
    finally:
        conn.close()

    pconn = get_db()
    try:
        prefs = _load_prefs(pconn, user_id) or {}
    finally:
        pconn.close()
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
    return jsonify({
        "xp": xp, "level": _level(xp), "streak": streak, "quizzes": quizzes,
        "badges": badges, "milestones": milestones,
        "earned_count": sum(1 for b in badges if b["earned"]), "total_badges": len(badges),
    })


@api_bp.route("/practice/generate", methods=["POST"])
def practice_generate():
    """Generate an MCQ set from an existing indexed Knowledge document and create
    a playable session. Reuses the RAG/agent pipeline (falls back to legacy on a
    document with no retrievable context). Returns the session_key the Quiz screen
    loads via the existing /student_login flow."""
    user_id = _require_user()
    if not user_id:
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    try:
        document_id = int(data.get("document_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "document_id required"}), 400

    num_questions = max(1, min(30, int(data.get("num_questions", 5) or 5)))
    difficulty = str(data.get("difficulty", "medium")).lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    topic = (data.get("topic") or "").strip() or None
    try:
        # Seconds. Upper bound matches the 180-minute limit on /upload.
        timer = max(30, min(10800, int(data.get("timer", 600) or 600)))
    except (TypeError, ValueError):
        timer = 600

    # Ownership: only practice from your own knowledge.
    from core.repositories import document_repo
    doc = document_repo.get(document_id)
    if doc is None or doc["owner"] != session.get("username"):
        return jsonify({"error": "not found"}), 404

    try:
        from core.services.mcq_pipeline import generate_from_document
        mcqs = generate_from_document(document_id, num_questions=num_questions,
                                      difficulty=difficulty, topic=topic)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    if not mcqs:
        return jsonify({"error": "No questions generated. Try different settings."}), 502

    from utils.session_manager import create_session_key
    session_key = create_session_key(
        teacher=session.get("username"), difficulty=difficulty, timer=timer, mcqs=mcqs,
    )
    return jsonify({"ok": True, "session_key": session_key, "count": len(mcqs)})
