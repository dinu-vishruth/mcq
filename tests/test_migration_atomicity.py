"""DDL and migration tracking must succeed or roll back together."""
import sqlite3
from unittest.mock import patch

import pytest

from core.models.sql_migrations import run_sql_migrations


def test_failed_sqlite_migration_rolls_back_ddl_and_can_be_retried(tmp_path):
    migration = tmp_path / '001_sqlite_test.sql'
    migration.write_text('CREATE TABLE example(id INTEGER); INVALID SQL;', encoding='utf-8')
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    try:
        with patch('core.models.sql_migrations.MIGRATIONS_DIR', str(tmp_path)):
            with pytest.raises(sqlite3.OperationalError):
                run_sql_migrations(conn, 'sqlite')
            assert conn.execute("SELECT name FROM sqlite_master WHERE name='example'").fetchone() is None
            assert not conn.execute('SELECT * FROM sql_migrations').fetchall()
            migration.write_text('CREATE TABLE example(id INTEGER);', encoding='utf-8')
            assert run_sql_migrations(conn, 'sqlite') == [migration.name]
            assert run_sql_migrations(conn, 'sqlite') == []
    finally:
        conn.close()
