"""PostgreSQL backend behind the same small surface the repositories use.

Every repository was written against ``sqlite3.Connection``:
``conn.execute(sql, params)`` with ``?`` placeholders, ``fetchone``/``fetchall``
returning Row-like ``row["column"]`` objects, ``executemany``, ``commit``,
``close``. Rather than rewrite a dozen repositories, this adapter presents that
same surface over a psycopg connection:

  * ``?`` placeholders are translated to psycopg's ``%s``.
  * Rows are wrapped so both ``row["col"]`` and ``dict(row)`` work like
    sqlite3.Row, and integer indexing (``row[0]``) maps to position.
  * ``executescript`` runs a whole multi-statement script in one cursor, which
    the migration runner relies on.

Scope note (deliberate simplification): this is a synchronous adapter used from
async handlers that are logged as threadpool-executed (the same constraint the
SQLite path already carries). Dialect differences (AUTOINCREMENT vs SERIAL,
PRAGMAs) are handled by the numbered SQL migrations in database/migrations/,
not here.
"""
from __future__ import annotations

import sqlite3

try:  # psycopg 3
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - only exercised without the dep
    psycopg = None
    dict_row = None

_DATABASE_URL: str = ""


class PgRow:
    """Row-like mapping: access by column name, by index, or via dict(row)."""

    def __init__(self, values: dict) -> None:
        self._values = dict(values)
        self._keys = list(self._values.keys())

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[self._keys[key]]
        return self._values[key]

    def get(self, key, default=None):
        return self._values.get(key, default)

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._values)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return repr(self._values)


def available() -> bool:
    """True when psycopg is importable, i.e. the Postgres path can be used."""
    return psycopg is not None


def open_postgres(database_url: str) -> "_PgConnection":
    """Build the connection. Raises promptly when psycopg is missing."""
    if psycopg is None:
        raise RuntimeError(
            "DATABASE_URL is set but psycopg is not installed. "
            "Install it with: pip install 'psycopg[binary]'"
        )
    import math

    from core.services.generation_budget import remaining
    seconds = remaining()
    cap = 30 if seconds is None else max(0.05, min(30, seconds))
    conn = psycopg.connect(database_url, row_factory=dict_row, autocommit=False,
                           connect_timeout=max(2, math.ceil(cap)))
    return _PgConnection(conn)


def _translate(sql: str) -> str:
    """Rewrite ``?`` placeholders to psycopg's ``%s``.

    Skips strings that use psycopg-style placeholders already, so adapter code
    can send either form. ``%%``-escaping is preserved as-is; the SQL this app
    writes never contains literal percent signs in statements with parameters.
    """
    if "%s" in sql or "(%(" in sql:
        return sql
    # Only a plain question mark surrounded by non-quote characters is a
    # placeholder; this cheap pass is safe for the SQL the app writes.
    return sql.replace("?", "%s")


class _Cursor:
    """Cursor facade over a real psycopg cursor with Row-like results."""

    def __init__(self, cur) -> None:
        self._cur = cur

    def fetchone(self):
        row = self._cur.fetchone()
        return PgRow(row) if row is not None else None

    def fetchall(self) -> list[PgRow]:
        return [PgRow(r) for r in self._cur.fetchall()]


class _PgConnection:
    """sqlite3.Connection-shaped facade over psycopg."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def _bound_statement(self, cur):
        # Transaction-local SQL works with managed transaction poolers. Startup
        # options are rejected by Neon, and session SET can leak across clients.
        from core.services.generation_budget import remaining
        seconds = remaining()
        if seconds is not None:
            cur.execute("SELECT set_config('statement_timeout', %s, true)",
                        (str(max(1, int(min(30, seconds) * 1000))),))

    def execute(self, sql: str, params: tuple = ()):
        self._cur = self._conn.cursor()
        self._bound_statement(self._cur)
        try:
            self._cur.execute(_translate(sql), params)
        except psycopg.IntegrityError as exc:
            raise sqlite3.IntegrityError(str(exc)) from exc
        return _Cursor(self._cur)

    def executescript(self, sql: str) -> None:
        """Run a script in the caller's transaction without committing it."""
        with self._conn.cursor() as cur:
            self._bound_statement(cur)
            cur.execute(sql)

    def executemany(self, sql: str, seq: list[tuple]) -> None:
        with self._conn.cursor() as cur:
            self._bound_statement(cur)
            cur.executemany(_translate(sql), seq)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()
