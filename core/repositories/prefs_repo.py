"""
User prefs + gamification (XP, streak) data access.

Centralizes the user_prefs table access that was previously inline SQL scattered
across routes. All writes are best-effort from the caller's perspective: a
failure here must never break the quiz submit flow.

Streak rule: a "study day" is any day the user completes activity. Completing
activity on a consecutive calendar day (yesterday -> today) increments the
streak; a gap of more than one day resets it to 1; same-day activity leaves the
streak unchanged. last_active_date stores the YYYY-MM-DD of the last study day.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from core.models.db import get_db

_FIELDS = "goal, style, daily_minutes, xp, streak, onboarded, last_active_date"


def load(user_id) -> dict | None:
    if not user_id:
        return None
    conn = get_db()
    try:
        row = conn.execute(
            f"SELECT {_FIELDS} FROM user_prefs WHERE user_id=?", (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _ensure_row(conn, user_id) -> None:
    exists = conn.execute("SELECT 1 FROM user_prefs WHERE user_id=?", (user_id,)).fetchone()
    if not exists:
        conn.execute("INSERT INTO user_prefs (user_id) VALUES (?)", (user_id,))


def save_onboarding(user_id, goal: str, style: str, daily_minutes: int) -> None:
    """Upsert the onboarding prefs and mark onboarded. Does not touch xp/streak."""
    if not user_id:
        return
    conn = get_db()
    try:
        now = datetime.utcnow().isoformat()
        _ensure_row(conn, user_id)
        conn.execute(
            "UPDATE user_prefs SET goal=?, style=?, daily_minutes=?, onboarded=1, updated_at=? "
            "WHERE user_id=?",
            (goal, style, daily_minutes, now, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def record_activity(user_id, xp_gain: int) -> dict:
    """Add XP and update the daily streak for one completed study activity.

    Returns {"xp", "streak", "leveled_up"} reflecting the new state, or an empty
    dict if there's no user. Idempotent within a day for the streak (same-day
    repeats don't re-increment), but XP always accrues.
    """
    if not user_id:
        return {}
    conn = get_db()
    try:
        _ensure_row(conn, user_id)
        row = conn.execute(
            "SELECT xp, streak, last_active_date FROM user_prefs WHERE user_id=?",
            (user_id,),
        ).fetchone()

        prev_xp = (row["xp"] if row else 0) or 0
        prev_streak = (row["streak"] if row else 0) or 0
        last_active = row["last_active_date"] if row else None

        today = date.today()
        today_str = today.isoformat()

        if last_active == today_str:
            new_streak = prev_streak or 1
        elif last_active == (today - timedelta(days=1)).isoformat():
            new_streak = prev_streak + 1
        else:
            new_streak = 1  # first activity ever, or streak broken

        new_xp = prev_xp + max(0, xp_gain)
        leveled_up = (new_xp // 100) > (prev_xp // 100)

        conn.execute(
            "UPDATE user_prefs SET xp=?, streak=?, last_active_date=?, updated_at=? WHERE user_id=?",
            (new_xp, new_streak, today_str, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        return {"xp": new_xp, "streak": new_streak, "leveled_up": leveled_up}
    finally:
        conn.close()
