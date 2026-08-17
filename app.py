# app.py
#
# Application entry point. This used to build a Flask app; it now exposes the
# FastAPI application, which fully replaces it.
#
# The module-level name `app` is kept deliberately: vercel.json targets app.py and
# @vercel/python detects an ASGI app under that name, so the deployment config
# needs no change. Everything real lives in fastapi_app/ -- see that package's
# __init__ for the layout.
#
# Run locally:
#     uvicorn app:app --reload
from fastapi_app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
