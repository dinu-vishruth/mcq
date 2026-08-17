"""CSRF protection and session helpers.

Sessions are handled by Starlette's ``SessionMiddleware`` (a signed cookie, same
idea as Flask's default session). That choice matters on Vercel: every request
can land on a different ephemeral instance, so anything stored server-side in
/tmp is effectively gone. A signed cookie travels with the client and verifies on
any instance.

CSRF uses the synchronizer-token pattern, matching what the existing templates
already expect:

  * ``csrf_token()`` is available as a Jinja global (see templating.py)
  * React pages read ``meta[name="csrf-token"]`` and send an ``X-CSRFToken``
    header on mutating fetches
  * classic form POSTs send a hidden ``csrf_token`` field

Both carriers are accepted so no template needs editing.
"""
from __future__ import annotations

import hmac
import secrets

from fastapi import Request
from fastapi.responses import HTMLResponse

#: Methods that never mutate state and therefore skip CSRF validation.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

#: Session key holding the per-session CSRF secret.
_CSRF_SESSION_KEY = "_csrf_token"

#: Paths exempt from CSRF validation. Kept empty by default -- every mutating
#: route in this app is either a same-origin form post or a fetch that already
#: sends the header. Add entries here (not decorators) so the exemption list
#: stays auditable in one place.
CSRF_EXEMPT_PATHS: frozenset[str] = frozenset()


def get_csrf_token(request: Request) -> str:
    """Return this session's CSRF token, minting one on first use.

    The token lives in the session, so it is bound to the client that will later
    submit it. An attacker's page can trigger a request but cannot read the
    victim's cookie to learn the token.
    """
    token = request.session.get(_CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        request.session[_CSRF_SESSION_KEY] = token
    return token


async def _submitted_token(request: Request) -> str | None:
    """Pull the token from the header, else the form body."""
    header = request.headers.get("X-CSRFToken") or request.headers.get("X-CSRF-Token")
    if header:
        return header

    content_type = request.headers.get("content-type", "")
    if content_type.startswith(("application/x-www-form-urlencoded", "multipart/form-data")):
        # This consumes the request stream. Safe only because CSRFMiddleware
        # buffers the body and replays it downstream -- see the note there.
        form = await request.form()
        value = form.get("csrf_token")
        if isinstance(value, str):
            return value
    return None


async def validate_csrf(request: Request) -> bool:
    """True when the request carries a token matching the session's."""
    expected = request.session.get(_CSRF_SESSION_KEY)
    if not expected:
        return False
    submitted = await _submitted_token(request)
    if not submitted:
        return False
    return hmac.compare_digest(str(submitted), str(expected))


def csrf_failure_response() -> HTMLResponse:
    """Mirror Flask-WTF's 400 so client behaviour is unchanged."""
    return HTMLResponse(
        "<h1>Bad Request</h1><p>The CSRF token is missing or invalid. "
        "Please reload the page and try again.</p>",
        status_code=400,
    )
