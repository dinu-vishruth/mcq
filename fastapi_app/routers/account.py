"""Email verification, password recovery, and explicit provider linking."""
import hashlib
import time
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool
from werkzeug.security import check_password_hash

from core.services.account_identity import identity_write_lock, password_signin_allowed
from core.services.account_recovery import consume_token, issue_token, mail_configured, send_token

from ..deps import CurrentUser, Db
from ..templating import render

router = APIRouter(prefix="/account", tags=["account"])
_GENERIC = "If an eligible account exists, a recovery link will be sent."


@router.get("")
async def account_page(request: Request, user: CurrentUser, conn: Db):
    row = conn.execute("SELECT email, email_verified FROM users WHERE id=?", (user.id,)).fetchone()
    return render(request, "account.html", contact=row, mail_available=mail_configured())


@router.get("/recover")
async def recover_form(request: Request):
    return render(request, "account_recover.html", mail_available=mail_configured())


@router.post("/recover")
async def recover(request: Request, conn: Db, email: Annotated[str, Form()] = ""):
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    digest = hashlib.sha256(ip.encode()).hexdigest()
    identity_write_lock(conn)
    conn.execute("DELETE FROM recovery_requests WHERE requested_at<?", (now - 300,))
    limited = conn.execute("SELECT count(*) FROM recovery_requests WHERE ip_hash=?", (digest,)).fetchone()[0] >= 5
    if not limited:
        conn.execute("INSERT INTO recovery_requests VALUES (?,?)", (digest, now))
    conn.commit()
    if limited:
        return render(request, "account_recover.html", notice=_GENERIC, mail_available=mail_configured())
    row = conn.execute("SELECT * FROM users WHERE lower(trim(email))=? AND email_verified=1", (email.strip().lower(),)).fetchone()
    if mail_configured() and row and password_signin_allowed(row):
        token = issue_token(conn, row, "reset")
        if token:
            try:
                await run_in_threadpool(send_token, row["email"], token, "reset")
            except Exception:
                pass
    return render(request, "account_recover.html", notice=_GENERIC, mail_available=mail_configured())


@router.post("/verify")
async def verify(request: Request, user: CurrentUser, conn: Db):
    if not mail_configured():
        raise HTTPException(503, "Email delivery is not configured. Contact the site owner.")
    row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    if row["email"] and not row["email_verified"]:
        token = issue_token(conn, row, "verify")
        if token:
            try:
                await run_in_threadpool(send_token, row["email"], token, "verify")
            except Exception:
                raise HTTPException(503, "Email could not be delivered. Please try later.") from None
    return RedirectResponse("/account", status_code=303)


@router.get("/confirm")
async def confirm_form(request: Request, token: str = ""):
    # Email scanners may follow GET links. Only the CSRF-protected POST uses it.
    response = render(request, "account_confirm.html", token=token)
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@router.post("/confirm")
async def confirm(request: Request, conn: Db, token: Annotated[str, Form()] = "",
                  password: Annotated[str, Form()] = ""):
    accepted = consume_token(conn, token, password)
    response = render(request, "account_confirm.html", token="",
                      notice="Account updated. You can sign in now." if accepted else "This link is invalid or expired, or the new password does not meet the password requirements.")
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@router.post("/link/{provider}")
async def start_link(request: Request, provider: str, user: CurrentUser, conn: Db,
                     password: Annotated[str, Form()] = ""):
    if provider not in ("google", "github"):
        raise HTTPException(404)
    row = conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    if time.time() - request.session.get("signed_in_at", 0) > 300:
        raise HTTPException(403, "Sign in again before linking an account.")
    if password_signin_allowed(row) and not check_password_hash(row["password_hash"], password):
        raise HTTPException(403, "Confirm your current password before linking.")
    request.session[f"_{provider}_link_target"] = {"user_id": user.id, "expires": time.time() + 300}
    return RedirectResponse(f"/auth/{provider}", status_code=303)
