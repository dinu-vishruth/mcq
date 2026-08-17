"""Custom middleware: CSRF enforcement and HTTPS redirection.

Both were ``@app.before_request`` hooks in Flask. In Starlette the equivalent is
middleware, which runs before any route handler.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from .security import (CSRF_EXEMPT_PATHS, SAFE_METHODS, csrf_failure_response,
                       validate_csrf)

Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]


def _replayer(messages: list[Message]) -> Receive:
    """Return a fresh ``receive`` callable that replays buffered messages.

    Each call to this function yields an independent cursor over the same
    buffered body, so the body can be read more than once (validation, then the
    route handler).
    """
    iterator = iter(messages)

    async def receive() -> Message:
        try:
            return next(iterator)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}

    return receive


class CSRFMiddleware:
    """Reject mutating requests without a valid synchronizer token.

    Replaces Flask-WTF's CSRFProtect. Deliberately does NOT check the Referer
    header: flask-wtf's strict HTTPS Referer check was a source of spurious 400s
    (any client that omits Referer got rejected), and the session-bound token
    already provides the protection.

    Written as pure-ASGI rather than ``BaseHTTPMiddleware`` on purpose. Reading
    the token out of a form body means consuming the request stream, and
    Starlette caches a parsed body on the ``Request`` *instance* -- not on the
    scope. ``BaseHTTPMiddleware`` hands the route handler a different instance,
    so a form read here would leave the handler with an empty body (every field
    silently arriving as ""). Buffering the ASGI messages and replaying them
    downstream is what makes the double read safe.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Message, receive: Receive, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["method"] in SAFE_METHODS or scope["path"] in CSRF_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        # Drain the request body into memory so it can be read twice. Vercel caps
        # request bodies at 4.5MB, so the buffer is bounded.
        messages: list[Message] = []
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request" or not message.get("more_body", False):
                break

        if not await validate_csrf(Request(scope, receive=_replayer(messages))):
            await csrf_failure_response()(scope, _replayer(messages), send)
            return

        await self.app(scope, _replayer(messages), send)


class HTTPSRedirectOnVercelMiddleware(BaseHTTPMiddleware):
    """Force HTTPS behind Vercel's proxy.

    The app itself always sees plain HTTP from the proxy, so the original scheme
    has to come from the X-Forwarded-Proto header rather than request.url.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.headers.get("X-Forwarded-Proto", "http") != "https":
            return RedirectResponse(
                str(request.url).replace("http://", "https://", 1), status_code=301
            )
        return await call_next(request)
