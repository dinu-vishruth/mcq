import sqlite3

import pytest

from scripts.ownership_migration import map_ownership, report


@pytest.fixture
def legacy_db():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY);
        INSERT INTO users VALUES(7);
        CREATE TABLE documents(id INTEGER PRIMARY KEY, owner TEXT, doc_hash TEXT, owner_user_id INTEGER);
        INSERT INTO documents VALUES(1, 'reused-name', 'hash', NULL);
        CREATE TABLE sessions(id INTEGER PRIMARY KEY, teacher TEXT, session_key TEXT, teacher_user_id INTEGER);
    """)
    yield conn
    conn.close()


def mapping(**overrides):
    return {'table': 'documents', 'record_id': 1, 'user_id': 7,
            'legacy_label': 'reused-name', 'expected_key': 'hash', **overrides}


def test_report_and_dry_run_do_not_assign_ownership(legacy_db):
    before = report(legacy_db)
    assert map_ownership(legacy_db, [mapping()]) == 1
    assert report(legacy_db) == before
    assert map_ownership(legacy_db, [mapping()], apply=True) == 1
    assert report(legacy_db)['documents'] == []


@pytest.mark.parametrize('invalid', [mapping(user_id=99), mapping(expected_key='changed'),
                                     mapping(legacy_label='changed'), mapping(user_id='7')])
def test_invalid_mapping_never_writes(legacy_db, invalid):
    before = report(legacy_db)
    with pytest.raises(ValueError):
        map_ownership(legacy_db, [invalid], apply=True)
    assert report(legacy_db) == before


def test_partial_mapping_failure_rolls_back_every_assignment(legacy_db):
    before = report(legacy_db)
    with pytest.raises(ValueError):
        map_ownership(legacy_db, [mapping(), mapping(record_id=99)], apply=True)
    assert report(legacy_db) == before
