"""Home, signup, login (with lockout), logout, profile, delete account.

Behaviour is a faithful port of core/routes/auth.py. Two FastAPI idioms worth
noting while reading:

  * form fields arrive as declared parameters (``Form(...)``) instead of reaching
    into a global ``request.form``
  * the ``Db`` dependency closes the connection for you, so the handlers no
    longer thread ``conn.close()`` through every branch
"""
from __future__ import annotations

import sqlite3
import unicodedata
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from werkzeug.security import check_password_hash, generate_password_hash

from core.services.account_identity import (
    SESSION_VERSION,
    establish_session,
    identity_write_lock,
    normalize_email,
    normalize_username,
    password_signin_allowed,
    rename_username_owned_rows,
    username_exists,
)

from ..deps import CurrentUser, Db
from ..templating import flash, render

router = APIRouter(tags=["auth"])

#: Failed logins allowed per IP/username before a temporary lockout.
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_WINDOW = timedelta(minutes=5)
MIN_PASSWORD_LENGTH = 8


def _has_current_session(request: Request) -> bool:
    if not request.session.get("user_id"):
        return False
    if request.session.get("identity_version") == SESSION_VERSION:
        return True
    request.session.clear()
    return False


@router.get("/")
async def home(request: Request):
    if _has_current_session(request):
        return RedirectResponse("/dashboard", status_code=302)
    return render(request, "login.html")


@router.get("/signup")
async def signup_form(request: Request):
    return render(request, "signup.html")


@router.post("/demo")
async def start_demo(request: Request, conn: Db):
    if _has_current_session(request):
        return RedirectResponse("/dashboard", status_code=302)
    from core.services.demo_service import create_demo_workspace
    try:
        user = create_demo_workspace(conn)
    except Exception:
        return render(request, "login.html", error="The demo could not start. Please try again.")
    establish_session(request, user)
    request.session["is_demo"] = True
    return RedirectResponse("/dashboard", status_code=302)


@router.post("/signup")
async def signup(
    request: Request,
    conn: Db,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
):
    username, password = username.strip(), password.strip()

    if not username or not password:
        return render(request, "signup.html", error="All fields are required.")
    if len(password) < MIN_PASSWORD_LENGTH:
        return render(
            request, "signup.html",
            error=f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.",
        )

    try:
        username = normalize_username(username)
        identity_write_lock(conn)
        if username_exists(conn, username):
            raise ValueError("Username already exists.")
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), "user"),
        )
        conn.commit()
    except ValueError as exc:
        conn.rollback()
        return render(request, "signup.html", error=str(exc))
    except sqlite3.IntegrityError:
        conn.rollback()
        return render(request, "signup.html", error="Username already exists.")

    return RedirectResponse("/", status_code=302)


@router.post("/login")
async def login(
    request: Request,
    conn: Db,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
):
    username, password = username.strip(), password.strip()
    ip = request.client.host if request.client else None

    window_start = (datetime.utcnow() - LOCKOUT_WINDOW).isoformat()
    failures = conn.execute(
        """
        SELECT COUNT(*) FROM login_attempts
        WHERE (ip=? OR username=?) AND success=0 AND attempt_time > ?
        """,
        (ip, username, window_start),
    ).fetchone()[0]

    if failures >= MAX_LOGIN_ATTEMPTS:
        return render(
            request, "login.html",
            error="Too many failed login attempts. Please wait 5 minutes.",
        )

    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    ok = bool(user and password_signin_allowed(user) and user["password_hash"]
              and check_password_hash(user["password_hash"], password))

    conn.execute(
        "INSERT INTO login_attempts (ip, username, attempt_time, success) VALUES (?, ?, ?, ?)",
        (ip, username, datetime.utcnow().isoformat(), 1 if ok else 0),
    )
    conn.commit()

    if not ok:
        return render(request, "login.html", error="Invalid username or password")

    establish_session(request, user)
    if "@" in unicodedata.normalize("NFKC", user["username"]):
        return RedirectResponse("/profile", status_code=302)
    from ..services.dashboard import load_prefs
    prefs = load_prefs(conn, user["id"]) or {}
    if not prefs.get("onboarded"):
        return RedirectResponse("/onboarding", status_code=302)
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)


@router.get("/profile")
async def profile_form(request: Request, user: CurrentUser, conn: Db):
    row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    return _profile_response(request, row)


def _profile_response(request, row, error=None):
    if row is None:
        request.session.clear()
        return RedirectResponse("/", status_code=302)
    password_enabled = password_signin_allowed(row)
    return render(request, "profile.html", user=row, error=error or "",
                  email_locked=not password_enabled or bool(row["email_verified"]),
                  password_enabled=password_enabled)


@router.post("/profile")
async def profile(
    request: Request,
    user: CurrentUser,
    conn: Db,
    username: Annotated[str, Form()] = "",
    email: Annotated[str | None, Form()] = None,
    password: Annotated[str, Form()] = "",
):
    # FastAPI applies the default to empty form strings. Preserve the difference
    # between clearing an optional contact address and leaving it unchanged.
    if email is None and "email" in await request.form():
        email = ""
    identity_write_lock(conn)
    row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    if row is None:
        conn.rollback()
        return _profile_response(request, row)

    new_username, new_password = username.strip(), password.strip()
    old_username, old_email = row["username"], row["email"]
    password_enabled = password_signin_allowed(row)
    email_locked = not password_enabled or bool(row["email_verified"])

    try:
        if new_username != old_username:
            new_username = normalize_username(new_username)
            if username_exists(conn, new_username, exclude_user_id=user.id):
                raise ValueError("Username already exists.")
        if new_password and not password_enabled:
            raise ValueError("Use your sign-in provider to manage your password.")
        if new_password and len(new_password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")

        new_email = old_email
        if email is not None:
            submitted_email = normalize_email(email)
            if email_locked and submitted_email != normalize_email(old_email or ""):
                raise ValueError("Your sign-in provider manages this email address.")
            if not email_locked:
                new_email = submitted_email
                if new_email and conn.execute(
                    "SELECT 1 FROM users WHERE LOWER(TRIM(email))=? AND id<>?",
                    (new_email, user.id),
                ).fetchone():
                    raise ValueError("This email is already in use.")

        password_hash = generate_password_hash(new_password) if new_password else row["password_hash"]
        conn.execute(
            "UPDATE users SET username=?, email=?, password_hash=? WHERE id=?",
            (new_username, new_email, password_hash, user.id),
        )
        if new_username != old_username:
            rename_username_owned_rows(conn, old_username, new_username)
        conn.commit()
    except ValueError as exc:
        conn.rollback()
        return _profile_response(request, row, error=str(exc))
    except sqlite3.IntegrityError:
        conn.rollback()
        return _profile_response(request, row, error="That username or email is already in use.")

    request.session["username"] = new_username
    flash(request, "Profile updated successfully!", "success")
    return RedirectResponse("/", status_code=302)


@router.post("/delete_account")
async def delete_account(request: Request, user: CurrentUser, conn: Db):
    identity_write_lock(conn)
    row = conn.execute("SELECT username FROM users WHERE id=?", (user.id,)).fetchone()
    if row is None:
        conn.rollback()
        request.session.clear()
        return RedirectResponse("/", status_code=302)
    username = row["username"]
    conn.execute("INSERT INTO retired_usernames (name) VALUES (lower(?)) ON CONFLICT (name) DO NOTHING",
                 (username,))
    # A single user may own generated sessions (by username) and quiz results
    # (by user_id), plus learning data. Remove all of it.
    conn.execute(
        "DELETE FROM results WHERE session_key IN "
        "(SELECT session_key FROM sessions WHERE teacher=?)",
        (username,),
    )
    conn.execute("DELETE FROM sessions WHERE teacher=?", (username,))
    for table in ("results", "learning_history", "weak_topics", "user_prefs", "quiz_uploads",
                  "identity_email_conflicts"):
        try:
            conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user.id,))
        except sqlite3.OperationalError:
            pass  # additive table not present on an older database
    conn.execute("DELETE FROM users WHERE id=?", (user.id,))
    conn.commit()

    request.session.clear()
    # flash() writes to the session, so it must come after clear().
    flash(request, "Your account has been deleted.", "success")
    return RedirectResponse("/", status_code=302)
