-- schema.sql
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT UNIQUE,
    teacher TEXT,
    created_at TEXT,
    difficulty TEXT,
    timer INTEGER,
    mcqs_json TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT,
    student_name TEXT,
    score INTEGER,
    total INTEGER,
    submitted_at TEXT,
    detail_json TEXT
);
