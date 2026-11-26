"""Owner identity, access activity, and reversible application controls."""
import time
from datetime import datetime, timezone

import config
from core.models.db import get_db
from core.services.account_identity import identity_write_lock

CONTROLS = {"registration": "New account signup", "demo": "Demo workspaces", "generation": "Quiz generation"}
ACTIVITY_RETENTION_SECONDS = 30 * 24 * 3600


def is_owner(row) -> bool:
    if row is None:
        return False
    return bool(config.ADMIN_OWNER_EMAIL and row["email_verified"]
                and (row["email"] or "").strip().casefold() == config.ADMIN_OWNER_EMAIL
                and not (row["is_disabled"] if "is_disabled" in row.keys() else False)
                and row["auth_provider"] != "demo")


def enabled(name: str, conn=None) -> bool:
    if name not in CONTROLS:
        raise ValueError("Unknown application control")
    own = conn is None
    conn = get_db() if own else conn
    try:
        row = conn.execute("SELECT enabled FROM app_controls WHERE name=?", (name,)).fetchone()
        return bool(row and row[0])
    finally:
        if own:
            conn.close()


def record_access(conn, request, user, event="sign_in"):
    now = time.time()
    # Use the direct peer supplied by the ASGI server; do not trust arbitrary
    # forwarded headers. The deployment's trusted proxy config owns that choice.
    ip = request.client.host if request.client else None
    conn.execute("DELETE FROM access_events WHERE occurred_at<?", (now - ACTIVITY_RETENTION_SECONDS,))
    conn.execute("INSERT INTO access_events(user_id,username,event,ip,occurred_at) VALUES (?,?,?,?,?)",
                 (user["id"], user["username"], event, ip, now))
    conn.execute("UPDATE users SET last_seen_at=? WHERE id=?", (now, user["id"]))
    conn.commit()


def audit(conn, actor_email, action, target):
    conn.execute("INSERT INTO admin_audit(actor_email,action,target,occurred_at) VALUES (?,?,?,?)",
                 (actor_email, action, str(target), time.time()))


def update_control(conn, actor_id, name, value):
    if name not in CONTROLS:
        raise ValueError("Unknown application control")
    actor = _authorized_write(conn, actor_id)
    conn.execute("UPDATE app_controls SET enabled=? WHERE name=?", (int(value), name))
    audit(conn, actor["email"], "enable" if value else "pause", name)
    conn.commit()


def update_user_access(conn, actor_id, target_id, action):
    actor = _authorized_write(conn, actor_id)
    target = conn.execute("SELECT * FROM users WHERE id=?", (target_id,)).fetchone()
    if target is None:
        conn.rollback()
        raise ValueError("Account no longer exists.")
    if target_id == actor_id or is_owner(target):
        conn.rollback()
        raise ValueError("The owner account cannot be disabled or signed out here.")
    if action not in ("disable", "enable", "revoke"):
        conn.rollback()
        raise ValueError("Unknown account action")
    if action == "revoke":
        conn.execute("UPDATE users SET session_epoch=session_epoch+1 WHERE id=?", (target_id,))
    else:
        conn.execute("UPDATE users SET is_disabled=?, session_epoch=session_epoch+1 WHERE id=?",
                     (int(action == "disable"), target_id))
    if action == "disable":
        conn.execute("DELETE FROM account_tokens WHERE user_id=?", (target_id,))
        conn.execute("UPDATE generation_jobs SET status='failed',run_token=NULL,error=?,updated_at=? "
                     "WHERE user_id=? AND status IN ('queued','running')",
                     ("Account access was disabled.", datetime.now(timezone.utc).replace(tzinfo=None).isoformat(), target_id))
    audit(conn, actor["email"], action, target["username"])
    conn.commit()


def _authorized_write(conn, actor_id):
    identity_write_lock(conn)
    actor = conn.execute("SELECT * FROM users WHERE id=?", (actor_id,)).fetchone()
    if not is_owner(actor):
        conn.rollback()
        raise PermissionError("Owner access is required.")
    return actor
