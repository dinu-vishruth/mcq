"""Report legacy ownership; apply only explicit mappings reviewed by an operator.

Run against a migrated database COPY before rollout. No schema changes here.
Default invocation is read-only; --mapping validates and rolls back unless --apply.
"""
import argparse
import json
from pathlib import Path

from core.models.db import get_db

TABLES = {
    'documents': ('owner_user_id', 'owner', 'doc_hash'),
    'sessions': ('teacher_user_id', 'teacher', 'session_key'),
}


def report(conn):
    return {
        table: [dict(row) for row in conn.execute(
            f'SELECT id, {label} AS legacy_label, {key} AS expected_key '
            f'FROM {table} WHERE {column} IS NULL ORDER BY id'
        ).fetchall()]
        for table, (column, label, key) in TABLES.items()
    }


def map_ownership(conn, mappings, *, apply=False):
    """Atomically validate mappings; never look up an owner by username."""
    seen = set()
    try:
        for entry in mappings:
            table = entry['table']
            column, label, key = TABLES[table]
            record_id, user_id = entry['record_id'], entry['user_id']
            if type(record_id) is not int or type(user_id) is not int:
                raise ValueError('Record and user IDs must be integers')
            if (table, record_id) in seen:
                raise ValueError('Duplicate mapping')
            seen.add((table, record_id))
            if conn.execute('SELECT id FROM users WHERE id=?', (user_id,)).fetchone() is None:
                raise ValueError('Mapped user does not exist')
            row = conn.execute(
                f'SELECT {column}, {label}, {key} FROM {table} WHERE id=?', (record_id,)
            ).fetchone()
            if row is None or row[column] is not None:
                raise ValueError('Record missing or already assigned')
            if row[label] != entry['legacy_label'] or row[key] != entry['expected_key']:
                raise ValueError('Record changed since review')
            changed = conn.execute(
                f'UPDATE {table} SET {column}=? WHERE id=? AND {column} IS NULL RETURNING id',
                (user_id, record_id),
            ).fetchone()
            if changed is None:
                raise ValueError('Concurrent ownership change; review again')
        if apply:
            conn.commit()
        else:
            conn.rollback()
        return len(seen)
    except Exception:
        conn.rollback()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, help='JSON array of reviewed record-to-user mappings')
    parser.add_argument('--apply', action='store_true', help='Commit the explicit mappings')
    args = parser.parse_args()
    if args.apply and not args.mapping:
        parser.error('--apply requires --mapping')
    conn = get_db()
    try:
        if args.mapping:
            count = map_ownership(conn, json.loads(args.mapping.read_text(encoding='utf-8')), apply=args.apply)
            print(json.dumps({'validated': count, 'applied': args.apply}))
        else:
            print(json.dumps(report(conn), indent=2))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
