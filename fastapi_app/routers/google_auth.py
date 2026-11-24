"""Google sign-in via the OAuth 2.0 authorization-code flow.

Implemented directly against Google's endpoints with httpx rather than through an
OAuth library -- it's about 100 lines, adds no dependency, and the flow is worth
seeing plainly:

  1. /auth/google            -> redirect the user to Google with a random `state`
  2. Google authenticates them and redirects back with a one-time `code`
  3. /auth/google/callback   -> verify `state`, exchange `code` for tokens over
                                the back channel, read the profile, log them in

Why `state` matters: it's stored in the session before the redirect and compared
on return, which is what stops an attacker from feeding you a callback for *their*
Google account and silently signing you into it (login CSRF).

Why the code exchange is a server-side POST: it carries the client secret and
returns tokens, so it must never touch the browser.
"""
from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from werkzeug.security import generate_password_hash

import config
from core.observability import log_event

from ..deps import Db
from ..templating import render

router = APIRouter(tags=["oauth"])

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"

#: We only need identity, not access to any Google service.
GOOGLE_SCOPES = "openid email profile"

_STATE_SESSION_KEY = "_google_oauth_state"
CALLBACK_PATH = "/auth/google/callback"

_GOOGLE_ERROR_CODES = frozenset({
    "access_denied", "admin_policy_enforced", "deleted_client", "invalid_client",
    "invalid_grant", "invalid_request", "org_internal", "redirect_uri_mismatch",
    "server_error", "temporarily_unavailable", "unauthorized_client",
})
_UNAVAILABLE = "Google sign-in is temporarily unavailable. Try another sign-in method or try again later."
_RESTART = "This Google sign-in attempt could not be completed. Start Google sign-in again."


def _google_error_code(value) -> str:
    return value if isinstance(value, str) and value in _GOOGLE_ERROR_CODES else "unknown"


class GoogleExchangeError(Exception):
    def __init__(self, stage: str, reason: str, status: int):
        self.stage = stage
        self.reason = reason
        self.status = status
        super().__init__(reason)


def _google_json(response: httpx.Response, stage: str) -> dict:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if response.status_code != 200:
        reason = _google_error_code(payload.get("error")) if isinstance(payload, dict) else "unknown"
        raise GoogleExchangeError(stage, reason, response.status_code)
    if not isinstance(payload, dict):
        raise GoogleExchangeError(stage, "invalid_response", response.status_code)
    return payload


def _redirect_uri(request: Request) -> str:
    """The callback URL Google will send the user back to.

    Derived from the request unless explicitly configured; that URL must be
    registered with Google. Forced to https on Vercel, behind a proxy terminating
    TLS, so request.url.scheme is 'http' and Google rejects a mismatched URI.
    """
    if config.GOOGLE_REDIRECT_URI:
        return config.GOOGLE_REDIRECT_URI
    base = str(request.base_url).rstrip("/")
    if config.IS_DEPLOYED or request.headers.get("X-Forwarded-Proto") == "https":
        base = base.replace("http://", "https://", 1)
    return f"{base}{CALLBACK_PATH}"


@router.get("/auth/google")
async def google_login(request: Request):
    """Kick off the flow by redirecting to Google's consent screen."""
    if not config.GOOGLE_OAUTH_ENABLED:
        return render(
            request, "login.html",
            error="Google sign-in isn't configured on this server.",
        )

    redirect_uri = _redirect_uri(request)
    callback = urlsplit(redirect_uri)
    if callback.netloc.lower() != request.url.netloc.lower():
        # Start on the callback host so the session cookie returns with state.
        # This also moves Vercel preview visitors to the registered production URL.
        target = callback._replace(path="/auth/google", query="", fragment="")
        return RedirectResponse(target.geturl(), status_code=302)

    state = secrets.token_urlsafe(32)
    request.session[_STATE_SESSION_KEY] = state
    request.session[_STATE_SESSION_KEY + "_redirect_uri"] = redirect_uri

    query = urlencode({
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        # Ask for a fresh account choice rather than silently reusing whichever
        # Google session the browser happens to hold.
        "prompt": "select_account",
    })
    return RedirectResponse(f"{GOOGLE_AUTH_ENDPOINT}?{query}", status_code=302)


@router.get(CALLBACK_PATH)
async def google_callback(
    request: Request,
    conn: Db,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """Handle Google's redirect back: verify state, exchange code, sign in."""
    expected_state = request.session.pop(_STATE_SESSION_KEY, None)
    redirect_uri = request.session.pop(_STATE_SESSION_KEY + "_redirect_uri", None)

    def fail(message: str):
        return render(request, "login.html", error=message)

    if (not state or not state.isascii() or not expected_state
            or not secrets.compare_digest(state, expected_state) or not redirect_uri):
        log_event("oauth.google.failed", level=logging.WARNING, stage="callback", reason="invalid_state")
        return fail(_RESTART)
    if error:
        log_event("oauth.google.failed", level=logging.WARNING, stage="authorize", reason=_google_error_code(error))
        return fail("Google did not grant access. Try again or use another sign-in method.")
    if not code:
        log_event("oauth.google.failed", level=logging.WARNING, stage="callback", reason="missing_code")
        return fail(_RESTART)

    try:
        profile = await _fetch_google_profile(code, redirect_uri)
    except GoogleExchangeError as exc:
        log_event("oauth.google.failed", level=logging.WARNING, stage=exc.stage,
                  reason=exc.reason, provider_status=exc.status)
        return fail(_RESTART if exc.stage == "token" and exc.reason == "invalid_grant" else _UNAVAILABLE)
    except httpx.HTTPError:
        log_event("oauth.google.failed", level=logging.WARNING, stage="exchange", reason="transport_error")
        return fail(_UNAVAILABLE)

    if not profile.get("sub") or not profile.get("email"):
        return fail("Google did not return a complete account profile. Please try again.")
    if profile.get("email_verified") is not True:
        # An unverified email could belong to someone else, and we match accounts
        # by email below -- so refuse rather than risk linking the wrong user.
        return fail("Your Google account's email address isn't verified.")

    user = _upsert_google_user(conn, profile)
    request.session["user_id"] = user["id"]
    request.session["username"] = user["username"]
    return RedirectResponse("/dashboard", status_code=302)


async def _fetch_google_profile(code: str, redirect_uri: str) -> dict:
    """Exchange the one-time code for tokens, then read the user's profile."""
    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            GOOGLE_TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": config.GOOGLE_CLIENT_ID,
                "client_secret": config.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        access_token = _google_json(token_response, "token").get("access_token")
        if not isinstance(access_token, str) or not access_token or not access_token.isascii():
            raise GoogleExchangeError("token", "invalid_response", token_response.status_code)

        profile_response = await client.get(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        return _google_json(profile_response, "userinfo")


def _upsert_google_user(conn, profile: dict) -> dict:
    """Find or create the local account for this Google identity.

    Matching order matters:
      1. by google_id  -- a returning OAuth user
      2. by email      -- someone who signed up with a password first; link the
                          accounts instead of creating a confusing duplicate
      3. create        -- brand new user

    New rows get a random unusable password_hash: the column is NOT NULL, and a
    random value means the account cannot be logged into via /login (no password
    will ever hash to it) without adding a nullable-password code path.
    """
    google_id = str(profile.get("sub") or "")
    email = (profile.get("email") or "").strip()
    picture = profile.get("picture") or ""

    existing = conn.execute("SELECT * FROM users WHERE google_id=?", (google_id,)).fetchone()
    if existing is None and email:
        existing = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if existing is not None:
            conn.execute(
                "UPDATE users SET google_id=?, picture=?, auth_provider='google' WHERE id=?",
                (google_id, picture, existing["id"]),
            )
            conn.commit()

    if existing is not None:
        return {"id": existing["id"], "username": existing["username"]}

    username = _unique_username(conn, profile, email)
    cursor = conn.execute(
        "INSERT INTO users (username, email, password_hash, role, google_id, picture, "
        "auth_provider) VALUES (?, ?, ?, 'user', ?, ?, 'google') RETURNING id",
        (username, email, generate_password_hash(secrets.token_urlsafe(32)),
         google_id, picture),
    )
    user_id = cursor.fetchone()["id"]
    conn.commit()
    return {"id": user_id, "username": username}


def _unique_username(conn, profile: dict, email: str) -> str:
    """Derive a username from the Google profile, suffixing on collision."""
    base = (profile.get("name") or email.split("@")[0] or "user").strip()[:32] or "user"
    candidate = base
    suffix = 2
    while conn.execute("SELECT 1 FROM users WHERE username=?", (candidate,)).fetchone():
        candidate = f"{base}{suffix}"
        suffix += 1
    return candidate
