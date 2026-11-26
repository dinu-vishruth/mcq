"""GitHub sign-in via the OAuth 2.0 authorization-code flow.

Mirrors fastapi_app/routers/google_auth.py: same state-based CSRF protection,
server-side code exchange, and upsert-into-local-account behaviour. GitHub's
token endpoint returns form-encoded (or JSON, depending on Accept) data, and
the user profile endpoint may not expose a verified email, so we fetch
/user/emails as a fallback.
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from werkzeug.security import generate_password_hash

import config
from core.observability import log_event
from core.services.account_identity import (
    claim_verified_email,
    establish_session,
    identity_write_lock,
    link_provider,
    normalize_email,
    unique_oauth_username,
)

from ..deps import Db
from ..templating import render

router = APIRouter(tags=["oauth"])

GITHUB_AUTH_ENDPOINT = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_ENDPOINT = "https://github.com/login/oauth/access_token"
GITHUB_USER_ENDPOINT = "https://api.github.com/user"
GITHUB_EMAILS_ENDPOINT = "https://api.github.com/user/emails"

#: Identity + (optionally) email access. `user:email` is required to read a
#: private email address through /user/emails.
GITHUB_SCOPES = "read:user user:email"

_STATE_SESSION_KEY = "_github_oauth_state"
CALLBACK_PATH = "/auth/github/callback"


def _redirect_uri(request: Request) -> str:
    if config.GITHUB_REDIRECT_URI:
        return config.GITHUB_REDIRECT_URI
    base = str(request.base_url).rstrip("/")
    if config.IS_DEPLOYED or request.headers.get("X-Forwarded-Proto") == "https":
        base = base.replace("http://", "https://", 1)
    return f"{base}{CALLBACK_PATH}"


@router.get("/auth/github")
async def github_login(request: Request):
    if not config.GITHUB_OAUTH_ENABLED:
        return render(
            request, "login.html",
            error="GitHub sign-in isn't configured on this server.",
        )

    redirect_uri = _redirect_uri(request)
    callback = urlsplit(redirect_uri)
    if callback.netloc.lower() != request.url.netloc.lower():
        # Start on the callback host so the session cookie returns with state.
        # This also moves Vercel preview visitors to the registered production URL.
        target = callback._replace(path="/auth/github", query="", fragment="")
        return RedirectResponse(target.geturl(), status_code=302)

    state = secrets.token_urlsafe(32)
    request.session[_STATE_SESSION_KEY] = state
    request.session[_STATE_SESSION_KEY + "_redirect_uri"] = redirect_uri

    query = urlencode({
        "client_id": config.GITHUB_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": GITHUB_SCOPES,
        "state": state,
    })
    return RedirectResponse(f"{GITHUB_AUTH_ENDPOINT}?{query}", status_code=302)


@router.get(CALLBACK_PATH)
async def github_callback(
    request: Request,
    conn: Db,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    expected_state = request.session.pop(_STATE_SESSION_KEY, None)
    redirect_uri = request.session.pop(_STATE_SESSION_KEY + "_redirect_uri", None)

    def fail(message: str):
        return render(request, "login.html", error=message)

    if error:
        return RedirectResponse("/", status_code=302)
    if not code:
        return fail("GitHub sign-in failed: no authorization code was returned.")
    if not state or not state.isascii() or not expected_state or not secrets.compare_digest(state, expected_state):
        return fail("GitHub sign-in failed: invalid state. Please try again.")

    try:
        profile = await _fetch_github_profile(code, redirect_uri or _redirect_uri(request))
    except (httpx.HTTPError, ValueError):
        return fail("Could not complete GitHub sign-in. Please try again.")

    if type(profile.get("id")) is not int or profile["id"] <= 0:
        return fail("GitHub did not return a complete account profile. Please try again.")
    try:
        linked = link_provider(conn, request, "github", str(profile["id"]),
                               profile.get("email") if profile.get("email_verified") is True else None)
        user = linked if linked is not None else _upsert_github_user(conn, profile)
    except sqlite3.IntegrityError:
        conn.rollback()
        log_event("oauth.github.failed", level=logging.WARNING, stage="account", reason="identity_conflict")
        return fail("This GitHub account could not be used to sign in. Try another sign-in method.")
    except ValueError as exc:
        conn.rollback()
        return fail(str(exc))
    user = conn.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    establish_session(request, user, conn)
    from core.services.admin_control import is_owner
    return RedirectResponse("/admin" if is_owner(user) else "/dashboard", status_code=302)


async def _fetch_github_profile(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            GITHUB_TOKEN_ENDPOINT,
            data={
                "client_id": config.GITHUB_CLIENT_ID,
                "client_secret": config.GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        if token_response.status_code != 200:
            raise ValueError("GitHub rejected the sign-in attempt. Please try again.")

        token_data = token_response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("GitHub did not return an access token.")

        profile_response = await client.get(
            GITHUB_USER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
        if profile_response.status_code != 200:
            raise ValueError("Could not read your GitHub profile.")
        profile = profile_response.json()

        if not isinstance(profile, dict) or type(profile.get("id")) is not int or profile["id"] <= 0:
            raise ValueError("GitHub did not return a complete account profile. Please try again.")
        # The public profile email has no verification status. A returning
        # account only needs the stable GitHub ID even when email is unavailable.
        profile["email"] = None
        profile["email_verified"] = False
        emails_response = await client.get(
            GITHUB_EMAILS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
        if emails_response.status_code == 200:
            try:
                entries = emails_response.json()
            except ValueError:
                entries = []
            if isinstance(entries, list):
                verified = (entry for entry in entries if isinstance(entry, dict) and entry.get("verified") is True
                            and isinstance(entry.get("email"), str) and entry["email"].strip())
                email_entry = max(verified, key=lambda entry: entry.get("primary") is True, default=None)
                if email_entry:
                    profile["email"] = email_entry["email"]
                    profile["email_verified"] = True
        return profile


def _upsert_github_user(conn, profile: dict) -> dict:
    github_id = str(profile["id"])
    identity_write_lock(conn)
    existing = conn.execute("SELECT id, username, email, email_verified FROM users WHERE github_id=?",
                            (github_id,)).fetchone()
    email = None
    raw_email = profile.get("email")
    if profile.get("email_verified") is True and isinstance(raw_email, str):
        try:
            email = normalize_email(raw_email)
        except ValueError:
            # GitHub email is optional and never identifies the signed-in account.
            pass
    if existing is not None:
        if (email and (existing["email"] != email or not existing["email_verified"])
                and claim_verified_email(conn, email, user_id=existing["id"])):
            conn.execute("UPDATE users SET email=?, email_verified=1 WHERE id=?", (email, existing["id"]))
        conn.commit()
        return {"id": existing["id"], "username": existing["username"]}

    from core.services.admin_control import enabled
    if not enabled("registration", conn):
        raise ValueError("New account signup is currently paused.")
    if email and not claim_verified_email(conn, email):
        raise sqlite3.IntegrityError("GitHub email already claimed")
    picture = profile.get("avatar_url")
    username = unique_oauth_username(conn, profile.get("name"), profile.get("login"))
    cursor = conn.execute(
        "INSERT INTO users (username, email, password_hash, role, github_id, picture, "
        "auth_provider, email_verified) VALUES (?, ?, ?, 'user', ?, ?, 'github', ?) RETURNING id",
        (username, email, generate_password_hash(secrets.token_urlsafe(32)),
         github_id, picture if isinstance(picture, str) else "", int(bool(email))),
    )
    user_id = cursor.fetchone()["id"]
    conn.commit()
    return {"id": user_id, "username": username}
