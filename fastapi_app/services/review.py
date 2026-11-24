"""Learning evidence derived from the signed-in user's saved answers."""
from __future__ import annotations

import json


def _decode(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except (TypeError, ValueError):
        return fallback


def build_review(conn, user_id: int, username: str) -> dict:
    rows = conn.execute(
        "SELECT r.id, r.session_key, r.detail_json, r.submitted_at, s.mcqs_json, "
        "d.id AS document_id, d.title AS document_title "
        "FROM results r LEFT JOIN sessions s ON s.session_key=r.session_key "
        "LEFT JOIN documents d ON d.id=s.document_id AND d.owner=? "
        "WHERE r.user_id=? ORDER BY r.submitted_at ASC, r.id ASC",
        (username, user_id),
    ).fetchall()
    questions = {}
    for row in rows:
        mcqs = _decode(row["mcqs_json"], [])
        if not isinstance(mcqs, list):
            mcqs = []
        originals = {
            (q.get("question", "").strip(), q.get("answer_text", "").strip()): q
            for q in mcqs if isinstance(q, dict)
        }
        details = _decode(row["detail_json"], [])
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict) or not detail.get("question"):
                continue
            key = (detail["question"].strip(), (detail.get("correct") or "").strip())
            previous = questions.get(key, {})
            correct = detail.get("is_correct") is True
            had_mistake = previous.get("had_mistake", False) or not correct
            confidence = detail.get("confidence")
            state = "corrected" if correct and had_mistake else "correct"
            if not correct:
                state = "misconception" if confidence == "know" else "revisit"
            elif confidence in ("maybe", "no_idea"):
                state = "uncertain"
            original = originals.get(key)
            questions[key] = {
                "question": detail["question"], "selected": detail.get("selected") or "",
                "correct": detail.get("correct") or "", "confidence": confidence,
                "state": state, "had_mistake": had_mistake, "is_correct": correct,
                "attempts": previous.get("attempts", 0) + 1,
                "last_seen": row["submitted_at"],
                "document_id": row["document_id"] or previous.get("document_id"),
                "document_title": row["document_title"] or previous.get("document_title"),
                "mcq": original or previous.get("mcq"),
            }

    items = sorted(questions.values(), key=lambda q: q["last_seen"] or "", reverse=True)
    pending = [q for q in items if not q["is_correct"]]
    uncertain = [q for q in items if q["state"] == "uncertain"]
    corrected = [q for q in items if q["had_mistake"] and q["is_correct"]]
    candidates = sorted(pending, key=lambda q: q["state"] != "misconception") + uncertain
    playable = [q["mcq"] for q in candidates if q["mcq"] and q["mcq"].get("options")]
    return {
        "summary": {
            "to_revisit": len(pending), "uncertain": len(uncertain),
            "corrected": len(corrected),
            "misconceptions": sum(q["state"] == "misconception" for q in pending),
            "reviewable": min(20, len(playable)),
        },
        "items": [{k: v for k, v in q.items() if k != "mcq"} for q in candidates[:50]],
        "improvements": [{k: v for k, v in q.items() if k != "mcq"} for q in corrected[:10]],
        "questions": playable[:20],
    }
