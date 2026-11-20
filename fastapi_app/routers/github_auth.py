"""GitHub sign-in via the OAuth 2.0 authorization-code flow.

Mirrors fastapi_app/routers/google_auth.py: same state-based CSRF protection,
server-side code exchange, and upsert-into-local-account behaviour. GitHub's
token endpoint returns form-encoded (or JSON, depending on Accept) data, and
the user profile endpoint may not expose a verified email, so we fetch
/user/emails as a fallback.
"""
from __future__ import annotations

import secrets
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from werkzeug.security import generate_password_hash

import config

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
    if not state or not expected_state or not secrets.compare_digest(state, expected_state):
        return fail("GitHub sign-in failed: invalid state. Please try again.")

    try:
        profile = await _fetch_github_profile(code, redirect_uri or _redirect_uri(request))
    except httpx.HTTPError as exc:
        return fail(f"Could not reach GitHub to complete sign-in: {exc}")
    except ValueError as exc:
        return fail(str(exc))

    user = _upsert_github_user(conn, profile)
    request.session["user_id"] = user["id"]
    request.session["username"] = user["username"]
    return RedirectResponse("/dashboard", status_code=302)


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

        if not profile.get("id"):
            raise ValueError("GitHub did not return a complete account profile. Please try again.")
        # Only verified addresses may link an existing local account. The public
        # profile's email field does not include its verification status.
        profile["email"] = ""
        emails_response = await client.get(
            GITHUB_EMAILS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
        if emails_response.status_code == 200:
            for entry in emails_response.json():
                if entry.get("primary") and entry.get("verified"):
                    profile["email"] = entry.get("email")
                    break
            if not profile.get("email"):
                for entry in emails_response.json():
                    if entry.get("verified"):
                        profile["email"] = entry.get("email")
                        break
        return profile


def _upsert_github_user(conn, profile: dict) -> dict:
    github_id = str(profile.get("id") or "")
    login = (profile.get("login") or "").strip()
    name = (profile.get("name") or "").strip()
    email = (profile.get("email") or "").strip()
    picture = profile.get("avatar_url") or ""

    existing = conn.execute("SELECT * FROM users WHERE github_id=?", (github_id,)).fetchone()

    if existing is None and email:
        existing = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if existing is not None:
            conn.execute(
                "UPDATE users SET github_id=?, picture=?, auth_provider='github' WHERE id=?",
                (github_id, picture, existing["id"]),
            )
            conn.commit()

    if existing is not None:
        return {"id": existing["id"], "username": existing["username"]}

    username = _unique_username(conn, name or login or "github-user")
    cursor = conn.execute(
        "INSERT INTO users (username, email, password_hash, role, github_id, picture, "
        "auth_provider) VALUES (?, ?, ?, 'user', ?, ?, 'github') RETURNING id",
        (username, email, generate_password_hash(secrets.token_urlsafe(32)),
         github_id, picture),
    )
    user_id = cursor.fetchone()["id"]
    conn.commit()
    return {"id": user_id, "username": username}


def _unique_username(conn, base: str) -> str:
    base = (base or "user").strip()[:32] or "user"
    candidate = base
    suffix = 2
    while conn.execute("SELECT 1 FROM users WHERE username=?", (candidate,)).fetchone():
        candidate = f"{base}{suffix}"
        suffix += 1
    return candidate
