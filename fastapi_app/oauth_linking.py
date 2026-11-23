"""Explicit account linking: prove the current account and the new provider identity."""
import sqlite3

from fastapi import HTTPException
from werkzeug.security import check_password_hash


def confirm_account(conn, user_id: int, password: str) -> None:
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(401, "Please sign in again.")
    if row["auth_provider"] in (None, "local"):
        if not password or not check_password_hash(row["password_hash"], password):
            raise HTTPException(400, "Confirm your current password to connect a sign-in method.")


def link_identity(conn, user_id: int, provider: str, subject: str) -> None:
    if provider not in ("google", "github") or not subject:
        raise ValueError("The provider did not return a valid identity.")
    column = provider + "_id"
    try:
        # A subject already attached to another account must never be moved.
        existing = conn.execute(f"SELECT id FROM users WHERE {column}=?", (subject,)).fetchone()
        if existing is not None and existing["id"] != user_id:
            raise ValueError("This sign-in method already belongs to another Examly account.")
        current = conn.execute(f"SELECT {column} FROM users WHERE id=?", (user_id,)).fetchone()
        if current is None:
            raise ValueError("Your account is no longer available. Please sign in again.")
        if current[column] and current[column] != subject:
            raise ValueError("A different account from this provider is already connected.")
        changed = conn.execute(
            f"UPDATE users SET {column}=? WHERE id=? AND ({column} IS NULL OR {column}=?) RETURNING id",
            (subject, user_id, subject),
        ).fetchone()
        if changed is None:
            raise ValueError("This sign-in method changed during linking. Please try again.")
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValueError("This sign-in method is already connected. Please sign in again.") from exc
    except ValueError:
        conn.rollback()
        raise
