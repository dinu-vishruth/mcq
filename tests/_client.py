"""Flask-test-client compatibility shim over FastAPI's TestClient.

The suite in test_regression.py / test_features.py was written against
``flask.Flask.test_client()``. The app is now FastAPI, whose TestClient (httpx)
has a slightly different surface. Rather than rewrite ~40 assertions, this shim
maps the handful of differences:

  ==========================  ====================================
  Flask test client           httpx / FastAPI TestClient
  ==========================  ====================================
  ``r.data`` -> bytes         ``r.content``
  ``r.get_json()``            ``r.json()``
  redirects NOT followed      redirects followed by default
  ``c.session_transaction()`` no equivalent -- decode the cookie
  ``WTF_CSRF_ENABLED=False``  no equivalent -- see ``_csrf`` below
  ==========================  ====================================

CSRF: the real app validates a session-bound synchronizer token. Instead of
disabling that (which would leave the protection untested and let a regression
through), the shim fetches a token and injects it into every mutating request,
exactly as a browser does. So these tests exercise the CSRF path rather than
bypassing it.
"""
from __future__ import annotations

import json
import re
from typing import Any

import itsdangerous
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

#: Matches either CSRF carrier the templates emit.
_TOKEN_RE = re.compile(
    r'name="csrf_token"[^>]*value="([^"]+)"|name="csrf-token"\s+content="([^"]+)"'
)

_SAFE_METHODS = {"get", "head", "options"}


class _Response:
    """Wraps an httpx response so ``.data`` and ``.get_json()`` keep working."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    @property
    def data(self) -> bytes:
        return self._raw.content

    def get_json(self) -> Any:
        return self._raw.json()

    def __getattr__(self, name: str) -> Any:
        # status_code, headers, text, json, cookies, ... pass straight through.
        return getattr(self._raw, name)


class FlaskLikeClient:
    """A per-test client that keeps cookies and speaks the Flask client's API."""

    def __init__(self, app: Any, secret_key: str) -> None:
        self._app = app
        self._secret_key = secret_key
        # follow_redirects=False matches Flask's default; tests assert on 302s.
        self._client = TestClient(app, follow_redirects=False)

    # -- CSRF ---------------------------------------------------------------
    def _csrf(self) -> str | None:
        """Return a token valid for this client's session.

        GET a page that renders one. Any page works -- the token is per-session,
        not per-form -- so start at ``/``.

        Redirects must be followed here: for a logged-in client ``/`` answers 302
        to /dashboard with an empty body, and a token would never be found.
        """
        match = _TOKEN_RE.search(self._client.get("/", follow_redirects=True).text)
        if not match:
            return None
        return match.group(1) or match.group(2)

    # -- request plumbing ---------------------------------------------------
    def open(self, path: str, method: str = "get", **kw: Any) -> _Response:
        method = method.lower()
        data = kw.pop("data", None)
        files = kw.pop("files", None)
        follow = kw.pop("follow_redirects", False)

        if method not in _SAFE_METHODS:
            token = self._csrf()
            if token is not None:
                # A form body can carry the hidden field. Anything else (a JSON
                # body, or no body at all) must use the header: adding `data`
                # alongside `json=` would replace the JSON payload and the route
                # would reject the request as malformed.
                if isinstance(data, dict) and "json" not in kw:
                    data = {**data, "csrf_token": token}
                else:
                    headers = dict(kw.get("headers") or {})
                    headers["X-CSRFToken"] = token
                    kw["headers"] = headers

        request_kw: dict[str, Any] = dict(kw)
        if data is not None:
            request_kw["data"] = data
        if files is not None:
            request_kw["files"] = files

        raw = self._client.request(
            method.upper(), path, follow_redirects=follow, **request_kw
        )
        return _Response(raw)

    def get(self, path: str, **kw: Any) -> _Response:
        return self.open(path, "get", **kw)

    def post(self, path: str, **kw: Any) -> _Response:
        return self.open(path, "post", **kw)

    def delete(self, path: str, **kw: Any) -> _Response:
        return self.open(path, "delete", **kw)

    # -- session inspection -------------------------------------------------
    def session_transaction(self) -> "_SessionView":
        return _SessionView(self._client, self._secret_key)


class _SessionView:
    """Read-only stand-in for Flask's ``session_transaction()`` context manager.

    Starlette signs the session cookie with itsdangerous and base64-encodes the
    JSON payload; decoding it here lets tests assert on session contents the way
    they did under Flask. Read-only is enough -- no test mutates the session.
    """

    def __init__(self, client: TestClient, secret_key: str) -> None:
        self._client = client
        self._secret_key = secret_key

    def __enter__(self) -> dict[str, Any]:
        cookie = self._client.cookies.get("session")
        if not cookie:
            return {}
        signer = itsdangerous.TimestampSigner(str(self._secret_key))
        payload = signer.unsign(cookie, max_age=14 * 24 * 3600)
        import base64

        return json.loads(base64.b64decode(payload))

    def __exit__(self, *exc: Any) -> None:
        return None


def make_client_factory(app: Any, secret_key: str):
    """Return a zero-arg callable that produces a fresh client per test."""

    def factory() -> FlaskLikeClient:
        return FlaskLikeClient(app, secret_key)

    return factory


__all__ = ["FlaskLikeClient", "make_client_factory", "SessionMiddleware"]
