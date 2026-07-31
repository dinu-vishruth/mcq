"""
Learning data access: learning_history + weak_topics (Phase 2 schema).

Powers the EvaluationAgent's weak-concept detection and the student dashboard's
(additive) weak-topics card. All writes are best-effort: a failure here must
never break the quiz submit flow, so callers wrap these in try/except.
"""
from __future__ import annotations

import json
from datetime import datetime

from core.models.db import get_db


def record_events(user_id, session_key, difficulty, events: list[dict]) -> None:
    """events: [{"topic","question","is_correct"}]."""
    if not user_id or not events:
        return
    conn = get_db()
    try:
        now = datetime.utcnow().isoformat()
        for e in events:
            conn.execute(
                "INSERT INTO learning_history (user_id, session_key, topic, question, "
                "is_correct, difficulty, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_key, e.get("topic", ""), e.get("question", ""),
                 1 if e.get("is_correct") else 0, difficulty, now),
            )
        conn.commit()
    finally:
        conn.close()


def upsert_weak_topics(user_id, topic_stats: dict) -> None:
    """topic_stats: {topic: {"wrong": int, "total": int}}."""
    if not user_id or not topic_stats:
        return
    conn = get_db()
    try:
        now = datetime.utcnow().isoformat()
        for topic, s in topic_stats.items():
            row = conn.execute(
                "SELECT id, wrong_count, total_count FROM weak_topics WHERE user_id=? AND topic=?",
                (user_id, topic),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO weak_topics (user_id, topic, wrong_count, total_count, last_seen, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, topic, s["wrong"], s["total"], now, now),
                )
            else:
                conn.execute(
                    "UPDATE weak_topics SET wrong_count=?, total_count=?, last_seen=?, updated_at=? WHERE id=?",
                    (row["wrong_count"] + s["wrong"], row["total_count"] + s["total"], now, now, row["id"]),
                )
        conn.commit()
    finally:
        conn.close()


def missed_questions_for_topic(user_id, topic: str, limit: int = 20) -> list[dict]:
    """Reconstruct the actual MCQs a user got wrong for a given topic.

    learning_history records each wrong answer's question text + the session it
    came from; the full question (options + correct answer) lives in that
    session's mcqs_json. We join the two by question text to rebuild playable
    MCQ dicts in the canonical {question, options:[{label,text}*4], answer_text}
    shape. Most-recent misses first; de-duplicated by question text so repeated
    misses of the same question appear once.

    Returns [] when nothing can be reconstructed (e.g. the source session was
    deleted), so callers can avoid offering a dead "review" action.
    """
    if not user_id or not topic:
        return []
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT lh.question, lh.session_key, MAX(lh.created_at) AS last_seen "
            "FROM learning_history lh "
            "WHERE lh.user_id=? AND lh.topic=? AND lh.is_correct=0 "
            "GROUP BY lh.question, lh.session_key "
            "ORDER BY last_seen DESC",
            (user_id, topic),
        ).fetchall()

        # Cache each session's mcqs_json so we parse it once per session.
        session_cache: dict[str, dict] = {}
        rebuilt, seen = [], set()
        for r in rows:
            q_text = (r["question"] or "").strip()
            key = q_text.lower()
            if not q_text or key in seen:
                continue
            skey = r["session_key"]
            if skey not in session_cache:
                srow = conn.execute("SELECT mcqs_json FROM sessions WHERE session_key=?", (skey,)).fetchone()
                index = {}
                if srow and srow["mcqs_json"]:
                    try:
                        for mcq in json.loads(srow["mcqs_json"]):
                            index[(mcq.get("question") or "").strip().lower()] = mcq
                    except (ValueError, TypeError):
                        pass
                session_cache[skey] = index
            mcq = session_cache[skey].get(key)
            if not mcq:
                continue  # source session gone or question text drifted
            rebuilt.append(mcq)
            seen.add(key)
            if len(rebuilt) >= limit:
                break
        return rebuilt
    finally:
        conn.close()


def top_weak_topics(user_id, limit=5) -> list[dict]:
    """Return worst-performing topics (highest wrong ratio, min 1 wrong)."""
    if not user_id:
        return []
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT topic, wrong_count, total_count FROM weak_topics "
            "WHERE user_id=? AND wrong_count > 0 "
            "ORDER BY (CAST(wrong_count AS REAL) / total_count) DESC, wrong_count DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {"topic": r["topic"], "wrong": r["wrong_count"], "total": r["total_count"],
             "pct": round(r["wrong_count"] / r["total_count"] * 100) if r["total_count"] else 0}
            for r in rows
        ]
    finally:
        conn.close()
