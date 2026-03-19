-- schema.sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('teacher', 'student')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

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
    user_id INTEGER,  -- Link to registered user (optional, nullable for guest students if we allowed them, but here we enforce login)
    score INTEGER,
    total INTEGER,
    submitted_at TEXT,
    detail_json TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
