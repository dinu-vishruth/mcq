"""FastAPI implementation of the MCQ generator.

This package is a full replacement for the Flask app in ``app.py``. It keeps
every route path, template, static asset and form/JSON contract identical so the
prebuilt React bundle in ``static/dist`` works unchanged, while the plumbing
(sessions, CSRF, request parsing, dependency injection) is idiomatic FastAPI.

Layout:
    main.py         app factory + middleware wiring + startup DB init
    templating.py   Jinja2 environment with the Flask-compatible globals
                    (``session``, ``csrf_token()``, ``get_flashed_messages()``)
    security.py     signed-cookie session helpers + CSRF synchronizer token
    deps.py         reusable dependencies (current user, DB connection)
    routers/        one module per former Flask blueprint
"""
