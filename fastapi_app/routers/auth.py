"""Home, signup, login (with lockout), logout, profile, delete account.

Behaviour is a faithful port of core/routes/auth.py. Two FastAPI idioms worth
noting while reading:

  * form fields arrive as declared parameters (``Form(...)``) instead of reaching
    into a global ``request.form``
  * the ``Db`` dependency closes the connection for you, so the handlers no
    longer thread ``conn.close()`` through every branch
"""
from __future__ import annotations

import logging
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
from core.services.admin_control import enabled, is_owner
from core.services.password_policy import (
    MAX_PASSWORD_LENGTH,
    needs_password_upgrade,
    validate_password,
)

from ..deps import CurrentUser, Db
from ..templating import flash, render

router = APIRouter(tags=["auth"])

#: Failed logins allowed per IP/username before a temporary lockout.
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_WINDOW = timedelta(minutes=5)


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
    if not enabled("demo", conn):
        return render(request, "login.html", error="The demo is currently paused. Please sign in or try again later.")
    from core.services.demo_service import create_demo_workspace
    try:
        user = create_demo_workspace(conn)
    except Exception:
        logging.getLogger(__name__).exception("Could not prepare demo workspace")
        return render(request, "login.html", error="The demo could not start. Please try again.")
    user = conn.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    establish_session(request, user, conn)
    request.session["is_demo"] = True
    return RedirectResponse("/dashboard", status_code=302)


@router.post("/signup")
async def signup(
    request: Request,
    conn: Db,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    email: Annotated[str, Form()] = "",
):
    username, email = username.strip(), email.strip()

    def invalid(message):
        return render(request, "signup.html", error=message,
                      signup_username=username, signup_email=email)

    if not enabled("registration", conn):
        return invalid("New account signup is currently paused. Existing accounts can still sign in.")
    if not username or not password or not email:
        return invalid("Username, email, and password are required.")
    try:
        username = normalize_username(username)
        email = normalize_email(email)
        validate_password(password, username=username, email=email)
        identity_write_lock(conn)
        if username_exists(conn, username):
            raise ValueError("Username already exists.")
        if conn.execute("SELECT 1 FROM users WHERE lower(trim(email))=?", (email,)).fetchone():
            raise ValueError("This email is already in use. Sign in or recover your existing account.")
        conn.execute(
            "INSERT INTO users (username, password_hash, role, email, email_verified) VALUES (?, ?, ?, ?, 0)",
            (username, generate_password_hash(password), "user", email),
        )
        conn.commit()
    except ValueError as exc:
        conn.rollback()
        return invalid(str(exc))
    except sqlite3.IntegrityError:
        conn.rollback()
        return invalid("This username or email address is already in use.")

    from core.services.account_recovery import deliver_account_token
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    delivery = await deliver_account_token(conn, row, "verify")
    if delivery == "sent":
        flash(request, "Account created. Check your email to verify your address before using password recovery.", "success")
    else:
        flash(request, "Account created. Sign in and open Account security to verify your email when email delivery is available.", "success")
    return RedirectResponse("/", status_code=302)


@router.post("/login")
async def login(
    request: Request,
    conn: Db,
    username: Annotated[str, Form(max_length=254)] = "",
    password: Annotated[str, Form(max_length=1024)] = "",
):
    username = username.strip()
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
    ok = bool(user and len(password) <= MAX_PASSWORD_LENGTH and password_signin_allowed(user) and user["password_hash"]
              and check_password_hash(user["password_hash"], password))

    conn.execute(
        "INSERT INTO login_attempts (ip, username, attempt_time, success) VALUES (?, ?, ?, ?)",
        (ip, username, datetime.utcnow().isoformat(), 1 if ok else 0),
    )
    conn.commit()

    if not ok:
        return render(request, "login.html", error="Invalid username or password")

    establish_session(request, user, conn)
    request.session["password_upgrade_required"] = needs_password_upgrade(
        password, username=user["username"], email=user["email"] or "")
    if request.session["password_upgrade_required"]:
        return RedirectResponse("/profile", status_code=302)
    if is_owner(user):
        return RedirectResponse("/admin", status_code=302)
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
                  password_enabled=password_enabled, is_admin=is_owner(row),
                  password_upgrade_required=bool(request.session.get("password_upgrade_required")))


@router.post("/profile")
async def profile(
    request: Request,
    user: CurrentUser,
    conn: Db,
    username: Annotated[str, Form()] = "",
    email: Annotated[str | None, Form()] = None,
    password: Annotated[str, Form()] = "",
    current_password: Annotated[str, Form()] = "",
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

    new_username, new_password = username.strip(), password
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
        if request.session.get("password_upgrade_required") and not new_password:
            raise ValueError("Set a stronger password before continuing.")
        if new_password:
            validate_password(new_password, username=new_username, email=row["email"] or "")
            if not check_password_hash(row["password_hash"], current_password):
                raise ValueError("Enter your current password to change it.")

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
        if new_password:
            conn.execute("UPDATE users SET session_epoch=session_epoch+1 WHERE id=?", (user.id,))
            conn.execute("DELETE FROM account_tokens WHERE user_id=?", (user.id,))
        conn.commit()
    except ValueError as exc:
        conn.rollback()
        return _profile_response(request, row, error=str(exc))
    except sqlite3.IntegrityError:
        conn.rollback()
        return _profile_response(request, row, error="That username or email is already in use.")

    request.session["username"] = new_username
    if new_password:
        request.session["session_epoch"] = row["session_epoch"] + 1
        request.session["password_upgrade_required"] = False
    flash(request, "Profile updated successfully!", "success")
    return RedirectResponse("/", status_code=302)


@router.post("/delete_account")
async def delete_account(request: Request, user: CurrentUser, conn: Db):
    from starlette.concurrency import run_in_threadpool

    from core.services.account_deletion import delete_account_data, process_deletions

    task_id = delete_account_data(conn, user.id)
    request.session.clear()
    if task_id:
        await run_in_threadpool(process_deletions, task_id)
    flash(request, "Your account has been deleted.", "success")
    return RedirectResponse("/", status_code=302)
