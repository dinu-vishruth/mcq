"""Single-use email proof bound to an account and its current contact address."""
import hashlib
import logging
import secrets
import smtplib
import ssl
import time
from email.message import EmailMessage
from urllib.parse import urlencode, urlsplit

from werkzeug.security import generate_password_hash

import config
from core.services.account_identity import identity_write_lock, password_signin_allowed
from core.services.password_policy import validate_password


def mail_configured():
    url = urlsplit(config.PUBLIC_BASE_URL)
    return bool(config.SMTP_HOST and config.SMTP_FROM and url.scheme == "https" and url.netloc
                and not url.username and not url.query and not url.fragment
                and url.path in ('', '/') and config.SMTP_STARTTLS
                and bool(config.SMTP_USERNAME) == bool(config.SMTP_PASSWORD))


async def deliver_account_token(conn, user, purpose):
    """Deliver without logging secrets; a failed send must allow an immediate retry."""
    from starlette.concurrency import run_in_threadpool

    from core.observability import log_event
    if not mail_configured():
        return "unavailable"
    token = issue_token(conn, user, purpose)
    if not token:
        return "limited"
    try:
        await run_in_threadpool(send_token, user['email'], token, purpose)
    except Exception as exc:
        conn.execute("DELETE FROM account_tokens WHERE digest=?",
                     (hashlib.sha256(token.encode()).hexdigest(),))
        conn.commit()
        log_event("account.mail_failed", level=logging.ERROR, purpose=purpose, error_type=type(exc).__name__)
        return "failed"
    log_event("account.mail_sent", purpose=purpose)
    return "sent"


def issue_token(conn, user, purpose):
    if purpose not in ("verify", "reset"):
        raise ValueError("Unsupported account token")
    now = time.time()
    identity_write_lock(conn)
    recent = conn.execute("SELECT 1 FROM account_tokens WHERE user_id=? AND created_at>?",
                          (user["id"], now - 60)).fetchone()
    if recent:
        conn.rollback()
        return None
    conn.execute("DELETE FROM account_tokens WHERE user_id=? AND purpose=?", (user["id"], purpose))
    token = secrets.token_urlsafe(32)
    conn.execute("INSERT INTO account_tokens VALUES (?,?,?,?,?,?)",
                 (hashlib.sha256(token.encode()).hexdigest(), user["id"], purpose, user["email"], now + 1800, now))
    conn.commit()
    return token


def send_token(email, token, purpose):
    if not mail_configured():
        raise RuntimeError("Account mail is not configured")
    message = EmailMessage()
    message["From"], message["To"] = config.SMTP_FROM, email
    message["Subject"] = "Confirm your Examly email" if purpose == "verify" else "Reset your Examly password"
    url = config.PUBLIC_BASE_URL.rstrip("/") + "/account/confirm?" + urlencode({"token": token})
    message.set_content(f"Open this link to {purpose} your Examly account. It expires in 30 minutes.\n\n{url}\n\nIf you did not request this, ignore this message.")
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as smtp:
        if config.SMTP_STARTTLS:
            smtp.starttls(context=ssl.create_default_context())
        if config.SMTP_USERNAME:
            smtp.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
        smtp.send_message(message)


def consume_token(conn, token, password=None):
    identity_write_lock(conn)
    row = conn.execute("SELECT t.*, u.email_verified, u.google_id, u.github_id, u.auth_provider, "
                       "u.is_disabled, u.username, u.email AS current_email FROM account_tokens t JOIN users u ON u.id=t.user_id "
                       "WHERE t.digest=? AND t.expires_at>?",
                       (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
    if row is None or row["is_disabled"] or row["email"] != row["current_email"]:
        conn.rollback()
        return False
    if row["purpose"] == "reset":
        if not row["email_verified"] or not password_signin_allowed(row) or not password:
            conn.rollback()
            return False
        try:
            validate_password(password, username=row["username"], email=row["current_email"] or "")
        except ValueError:
            conn.rollback()
            return False
        conn.execute("UPDATE users SET password_hash=?, session_epoch=session_epoch+1 WHERE id=?",
                     (generate_password_hash(password), row["user_id"]))
    else:
        conn.execute("UPDATE users SET email_verified=1 WHERE id=?", (row["user_id"],))
    conn.execute("DELETE FROM account_tokens WHERE user_id=?", (row["user_id"],))
    conn.commit()
    return True
