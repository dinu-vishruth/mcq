"""FastAPI application factory.

Replaces app.py. Middleware ordering is the one thing here that is easy to get
wrong: ``add_middleware`` PREPENDS, so the middleware added *last* is outermost
and sees the request *first*. The adds below are therefore written in reverse of
the execution order:

    execution order            added
    1. HTTPSRedirect (Vercel)  last
    2. Session (decode cookie)  2nd
    3. CSRF (validate token)   first

CSRF has to run inside Session, because validating a token means comparing it to
one stored in the session -- reading ``request.session`` before SessionMiddleware
has run raises outright.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from config import SECRET_KEY, UPLOAD_FOLDER

from .database import init_db
from .deps import RedirectException
from .middleware import CSRFMiddleware, HTTPSRedirectOnVercelMiddleware
from .routers import api, auth, documents, google_auth, student, teacher
from .templating import BASE_DIR

IS_VERCEL = bool(os.environ.get("VERCEL"))

#: 24h, matching the Flask PERMANENT_SESSION_LIFETIME.
SESSION_MAX_AGE = 60 * 60 * 24


def _log_llm_config() -> None:
    """State the LLM credential situation once, at boot.

    Without this the only signal that no key was found arrived after a user had
    already picked a file and waited through an upload -- the failure surfaced as
    a mid-flow error on the quiz screen. Logging it at startup means the operator
    sees the problem before any learner does.
    """
    import config

    if config.llm_key_present():
        try:
            from core.llm import get_llm
            llm = get_llm()
            print(f"[startup] LLM ready: provider={llm.provider_id} model={llm.model}")
        except Exception as exc:  # never block boot on provider construction
            print(f"[startup] WARNING: LLM key present but provider setup failed: {exc}")
    else:
        print(f"[startup] WARNING: {config.missing_key_message()}")
        print("[startup] Quiz generation will fail until a key is set; the rest of the app works.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Examly",
        description="AI-powered MCQ generation and exam prep.",
        # The interactive docs are a FastAPI nicety worth keeping while learning;
        # they only expose the JSON API, which is session-guarded anyway.
        docs_url="/docs",
        redoc_url=None,
    )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    init_db()
    _log_llm_config()

    # --- Middleware -- added in REVERSE of execution order, see docstring ----
    app.add_middleware(CSRFMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=SECRET_KEY,
        max_age=SESSION_MAX_AGE,
        same_site="lax",
        https_only=IS_VERCEL,
        session_cookie="session",
    )
    if IS_VERCEL:
        app.add_middleware(HTTPSRedirectOnVercelMiddleware)

    # --- Static files -------------------------------------------------------
    # Mounted at the same prefix Flask served, so /static/dist/app.js and the
    # rest of the prebuilt React bundle resolve without touching any template.
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    # --- Routers (one per former Flask blueprint) ---------------------------
    app.include_router(auth.router)
    app.include_router(google_auth.router)
    app.include_router(student.router)
    app.include_router(documents.router)
    app.include_router(teacher.router)
    app.include_router(api.router)

    @app.exception_handler(RedirectException)
    async def _handle_redirect(request: Request, exc: RedirectException):
        """Turn a dependency's RedirectException into a real 302."""
        return RedirectResponse(exc.location, status_code=302)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(request: Request, exc: StarletteHTTPException):
        """Keep the Flask error shape on /api: ``{"error": ...}``, not ``{"detail": ...}``.

        frontend/src/bootstrap.ts reads ``data.error`` to surface a message, so
        FastAPI's default ``detail`` key would silently degrade every API error
        into a generic "failed: 500".
        """
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
        return await http_exception_handler(request, exc)

    return app


app = create_app()
