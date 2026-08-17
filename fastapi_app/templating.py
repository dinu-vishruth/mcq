"""Jinja environment that keeps the existing templates working untouched.

The templates were written against Flask, so they rely on three things that
FastAPI does not provide out of the box:

  ``session``               a mapping of the current session (``session.username``)
  ``csrf_token()``          callable returning the synchronizer token
  ``get_flashed_messages()`` Flask's one-shot message queue

Rather than rewrite 18 templates, this module supplies all three. Jinja resolves
``session.username`` on a plain dict by falling back to item access, and a missing
key yields Undefined -- which is exactly what ``| default(0)`` and ``or "Learner"``
in the templates already handle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse

import config

from .security import get_csrf_token

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"

#: Session key holding queued flash messages as (category, message) pairs.
_FLASH_KEY = "_flashes"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Available to every template without each route passing it: login.html and
# signup.html use it to decide whether to render the Google button as a working
# link or an inert placeholder.
templates.env.globals["google_oauth_enabled"] = config.GOOGLE_OAUTH_ENABLED


def flash(request: Request, message: str, category: str = "message") -> None:
    """Queue a message for the next rendered template (Flask's ``flash``)."""
    request.session.setdefault(_FLASH_KEY, []).append((category, message))


def _get_flashed_messages(request: Request, with_categories: bool = False) -> list[Any]:
    """Drain the flash queue. Consuming is the point -- messages show once."""
    queued = request.session.pop(_FLASH_KEY, [])
    if with_categories:
        return [(category, message) for category, message in queued]
    return [message for _category, message in queued]


def render(request: Request, template_name: str, /, **context: Any) -> HTMLResponse:
    """Render a template with the Flask-compatible globals injected.

    Use this everywhere instead of ``templates.TemplateResponse`` so no route can
    forget the session/CSRF globals and render a page whose forms then fail
    validation.
    """
    context.setdefault("request", request)
    context["session"] = request.session
    context["csrf_token"] = lambda: get_csrf_token(request)
    context["get_flashed_messages"] = (
        lambda with_categories=False: _get_flashed_messages(request, with_categories)
    )
    return templates.TemplateResponse(request, template_name, context)
