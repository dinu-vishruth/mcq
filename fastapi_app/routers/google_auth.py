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

import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from werkzeug.security import generate_password_hash

import config

from ..deps import Db
from ..templating import flash, render

router = APIRouter(tags=["oauth"])

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"

#: We only need identity, not access to any Google service.
GOOGLE_SCOPES = "openid email profile"

_STATE_SESSION_KEY = "_google_oauth_state"
CALLBACK_PATH = "/auth/google/callback"


def _redirect_uri(request: Request) -> str:
    """The callback URL Google will send the user back to.

    Derived from the request so preview deployments work without extra config,
    but forced to https on Vercel -- the app sits behind a proxy that terminates
    TLS, so request.url.scheme is 'http' and Google rejects a mismatched URI.
    """
    if config.GOOGLE_REDIRECT_URI:
        return config.GOOGLE_REDIRECT_URI
    base = str(request.base_url).rstrip("/")
    if request.headers.get("X-Forwarded-Proto") == "https":
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

    state = secrets.token_urlsafe(32)
    request.session[_STATE_SESSION_KEY] = state

    query = urlencode({
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": _redirect_uri(request),
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

    def fail(message: str):
        return render(request, "login.html", error=message)

    if error:
        # The user pressed "Cancel" on the consent screen -- not an error worth
        # shouting about, just send them back to the login page.
        return RedirectResponse("/", status_code=302)
    if not code:
        return fail("Google sign-in failed: no authorization code was returned.")
    if not state or not expected_state or not secrets.compare_digest(state, expected_state):
        return fail("Google sign-in failed: invalid state. Please try again.")

    try:
        profile = await _fetch_google_profile(code, _redirect_uri(request))
    except httpx.HTTPError as exc:
        return fail(f"Could not reach Google to complete sign-in: {exc}")
    except ValueError as exc:
        return fail(str(exc))

    if not profile.get("email_verified"):
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
        if token_response.status_code != 200:
            raise ValueError("Google rejected the sign-in attempt. Please try again.")

        access_token = token_response.json().get("access_token")
        if not access_token:
            raise ValueError("Google did not return an access token.")

        profile_response = await client.get(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if profile_response.status_code != 200:
            raise ValueError("Could not read your Google profile.")
        return profile_response.json()


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
        "auth_provider) VALUES (?, ?, ?, 'student', ?, ?, 'google')",
        (username, email, generate_password_hash(secrets.token_urlsafe(32)),
         google_id, picture),
    )
    conn.commit()
    return {"id": cursor.lastrowid, "username": username}


def _unique_username(conn, profile: dict, email: str) -> str:
    """Derive a username from the Google profile, suffixing on collision."""
    base = (profile.get("name") or email.split("@")[0] or "user").strip()[:32] or "user"
    candidate = base
    suffix = 2
    while conn.execute("SELECT 1 FROM users WHERE username=?", (candidate,)).fetchone():
        candidate = f"{base}{suffix}"
        suffix += 1
    return candidate
