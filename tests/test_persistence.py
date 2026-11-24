"""Offline tests for the production-persistence layer.

The Postgres backend and pgvector store need a real server to talk to, and no
test in this suite may touch a network. What CAN be verified offline is the
adapter logic (placeholder translation, Row dual indexing) and the migration
runner's file discovery rules, which are the pieces most likely to break
silently in a drive-by edit.
"""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from core.models import pg_adapter
from core.models.pg_adapter import PgRow


class TestPgRow(unittest.TestCase):
    """sqlite3.Row contract: index by name, by position, and dict(row)."""

    def test_indexing_by_name_and_position(self):
        row = PgRow({"a": 1, "b": "x"})
        self.assertEqual(row["a"], 1)
        self.assertEqual(row["b"], "x")
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], "x")

    def test_dict_and_keys(self):
        row = PgRow({"a": 1, "b": "x"})
        self.assertEqual(dict(row), {"a": 1, "b": "x"})
        self.assertEqual(list(row.keys()), ["a", "b"])
        self.assertIsNone(row.get("missing"))

    def test_missing_key_raises(self):
        row = PgRow({"a": 1})
        with self.assertRaises(KeyError):
            row["nope"]


class TestPlaceholderTranslation(unittest.TestCase):
    def test_sqlite_style_placeholders_are_translated(self):
        sql = "SELECT * FROM users WHERE id=? AND owner=?"
        self.assertEqual(pg_adapter._translate(sql),
                         "SELECT * FROM users WHERE id=%s AND owner=%s")

    def test_psycopg_style_placeholders_pass_through(self):
        sql = "INSERT INTO t (a, b) VALUES (%s, %s)"
        self.assertEqual(pg_adapter._translate(sql), sql)


class TestMigrationRunnerDiscovery(unittest.TestCase):
    """The runner is file-driven; its discovery rules are all that runs offline."""

    @staticmethod
    def _runner_for(tmp_dir):
        import core.models.sql_migrations as runner

        orig_dir = runner.MIGRATIONS_DIR
        runner.MIGRATIONS_DIR = tmp_dir
        return runner, orig_dir

    def _teardown(self, orig_dir):
        import core.models.sql_migrations as runner

        runner.MIGRATIONS_DIR = orig_dir

    def test_unknown_backend_rejected(self):
        runner, orig = self._runner_for(tempfile.mkdtemp())
        try:
            conn = sqlite3.connect(":memory:")
            with self.assertRaises(ValueError):
                runner.run_sql_migrations(conn, "mysql")
        finally:
            self._teardown(orig)

    def test_migrations_only_apply_to_their_dialect(self):
        tmp = tempfile.mkdtemp()
        runner, orig = self._runner_for(tmp)
        try:
            conn = sqlite3.connect(":memory:")
            # postgres-only files are skipped for the sqlite backend
            with open(os.path.join(tmp, "001_postgres_init.sql"), "w") as fh:
                fh.write("CREATE TABLE should_not_apply (id INTEGER);")
            applied = runner.run_sql_migrations(conn, "sqlite")
            self.assertEqual(applied, [])
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            self.assertNotIn("should_not_apply", names)
        finally:
            self._teardown(orig)


def test_postgres_boot_serializes_migrations_until_one_final_commit(tmp_path: Path):
    from core.models import db, sql_migrations
    from fastapi_app.database import init_db

    (tmp_path / "001_postgres_init.sql").write_text("CREATE TABLE first (id integer);", encoding="utf-8")
    (tmp_path / "002_postgres_next.sql").write_text("CREATE TABLE second (id integer);", encoding="utf-8")
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.__enter__.return_value = cursor
    cursor.fetchall.return_value = []
    with patch.object(db, "is_postgres", return_value=True), \
            patch.object(db, "get_db", return_value=pg_adapter._PgConnection(connection)), \
            patch.object(sql_migrations, "MIGRATIONS_DIR", tmp_path):
        init_db()

    assert cursor.execute.call_args_list[0] == call("SELECT pg_advisory_xact_lock(7842301)", ())
    connection.assert_has_calls([
        call.cursor().execute("CREATE TABLE first (id integer);"),
        call.cursor().execute("INSERT INTO sql_migrations (name) VALUES (%s)", ("001_postgres_init.sql",)),
        call.cursor().execute("CREATE TABLE second (id integer);"),
        call.cursor().execute("INSERT INTO sql_migrations (name) VALUES (%s)", ("002_postgres_next.sql",)),
        call.commit(),
        call.close(),
    ], any_order=True)
    assert connection.mock_calls[-2:] == [call.commit(), call.close()]
    connection.commit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
