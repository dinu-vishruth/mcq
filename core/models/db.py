"""
Central SQLite access.

`get_db()` mirrors the connection app.py already uses (Row factory, same path).
It reads config.DB_PATH at call time (not import time) so tests that repoint the
DB continue to work. This module does NOT own schema creation for the legacy
tables — app.init_db() remains the source of truth for those — it adds the new
RAG/agentic tables via core.models.migrations, which is safe to call repeatedly.
"""
from __future__ import annotations

import sqlite3

import config


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
