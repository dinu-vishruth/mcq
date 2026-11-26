"""Preserve old SQLite accounts while admitting the unified learner role."""
import re
import sqlite3

_ROLE_CHECK = re.compile(r"CHECK\s*\(\s*role\s+IN\s*\(([^)]+)\)\s*\)", re.IGNORECASE)
_USERS_TABLE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:users|"users"|`users`|\[users\])\s*\(',
    re.IGNORECASE,
)


def upgrade_legacy_user_roles(conn: sqlite3.Connection) -> None:
    """Expand a legacy role CHECK without changing any rows or their IDs.

    SQLite cannot change a CHECK in place. Rebuild only when the stored schema
    has the old restriction, keeping columns, indexes, triggers, and the ID
    sequence. Disable foreign-key actions for the swap and restore the setting.
    """
    def upgraded_schema():
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
        if row is None:
            return None
        ddl = row[0]
        check = _ROLE_CHECK.search(ddl)
        if check is None or re.search(r"['\"]user['\"]", check[1], re.IGNORECASE):
            return None
        ddl = ddl[:check.end(1)] + ", 'user'" + ddl[check.end(1):]
        ddl, replacements = _USERS_TABLE.subn('CREATE TABLE "users_role_upgrade" (', ddl, count=1)
        if replacements != 1:
            raise RuntimeError("Could not upgrade the legacy users table")
        return ddl

    if upgraded_schema() is None:
        return
    conn.commit()
    foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        ddl = upgraded_schema()  # Another boot may have completed the upgrade.
        if ddl is None:
            conn.commit()
            return
        objects = conn.execute(
            "SELECT sql FROM sqlite_master WHERE tbl_name='users' "
            "AND type IN ('index','trigger') AND sql IS NOT NULL"
        ).fetchall()
        columns = [row[1] for row in conn.execute("PRAGMA table_xinfo(users)").fetchall() if row[6] == 0]
        names = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
        has_sequence = conn.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'").fetchone()
        sequence = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='users'").fetchone() if has_sequence else None
        conn.execute(ddl)
        conn.execute(f'INSERT INTO users_role_upgrade ({names}) SELECT {names} FROM users')
        conn.execute("DROP TABLE users")
        conn.execute("ALTER TABLE users_role_upgrade RENAME TO users")
        for obj in objects:
            conn.execute(obj[0])
        if sequence:
            conn.execute("UPDATE sqlite_sequence SET seq=max(seq, ?) WHERE name='users'", (sequence[0],))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute(f"PRAGMA foreign_keys={int(foreign_keys)}")
