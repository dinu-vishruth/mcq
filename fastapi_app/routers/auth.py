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
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from werkzeug.security import check_password_hash, generate_password_hash

from ..deps import CurrentUser, Db
from ..templating import flash, render

router = APIRouter(tags=["auth"])

#: Failed logins allowed per IP/username before a temporary lockout.
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_WINDOW = timedelta(minutes=5)
MIN_PASSWORD_LENGTH = 8


@router.get("/")
async def home(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/dashboard", status_code=302)
    return render(request, "login.html")


@router.get("/signup")
async def signup_form(request: Request):
    return render(request, "signup.html")


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

    # Single unified "User" role. The DB role column has a legacy CHECK
    # constraint (teacher|student); we write 'student' to satisfy it but never
    # branch on it.
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), "student"),
        )
        conn.commit()
    except sqlite3.IntegrityError:
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
    ok = bool(user and user["password_hash"] and check_password_hash(user["password_hash"], password))

    conn.execute(
        "INSERT INTO login_attempts (ip, username, attempt_time, success) VALUES (?, ?, ?, ?)",
        (ip, username, datetime.utcnow().isoformat(), 1 if ok else 0),
    )
    conn.commit()

    if not ok:
        return render(request, "login.html", error="Invalid username or password")

    request.session["user_id"] = user["id"]
    request.session["username"] = user["username"]
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)


@router.get("/profile")
async def profile_form(request: Request, user: CurrentUser, conn: Db):
    row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    return render(request, "profile.html", user=row)


@router.post("/profile")
async def profile(
    request: Request,
    user: CurrentUser,
    conn: Db,
    username: Annotated[str, Form()] = "",
    email: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
):
    row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    new_username, new_email, new_password = username.strip(), email.strip(), password.strip()

    try:
        if not new_username:
            raise ValueError("Username is required.")
        if new_username != row["username"]:
            taken = conn.execute(
                "SELECT 1 FROM users WHERE username=?", (new_username,)
            ).fetchone()
            if taken:
                raise ValueError("Username already exists.")
        if new_password and len(new_password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    except ValueError as exc:
        return render(request, "profile.html", user=row, error=str(exc))

    if new_password:
        conn.execute(
            "UPDATE users SET username=?, email=?, password_hash=? WHERE id=?",
            (new_username, new_email, generate_password_hash(new_password), user.id),
        )
    else:
        conn.execute(
            "UPDATE users SET username=?, email=? WHERE id=?",
            (new_username, new_email, user.id),
        )
    conn.commit()

    request.session["username"] = new_username
    flash(request, "Profile updated successfully!", "success")
    return RedirectResponse("/", status_code=302)


@router.post("/delete_account")
async def delete_account(request: Request, user: CurrentUser, conn: Db):
    # A single user may own generated sessions (by username) and quiz results
    # (by user_id), plus learning data. Remove all of it.
    conn.execute(
        "DELETE FROM results WHERE session_key IN "
        "(SELECT session_key FROM sessions WHERE teacher=?)",
        (user.username,),
    )
    conn.execute("DELETE FROM sessions WHERE teacher=?", (user.username,))
    for table in ("results", "learning_history", "weak_topics", "user_prefs"):
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
