CREATE TABLE IF NOT EXISTS quiz_attempts (
    session_key TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    started_at REAL NOT NULL,
    deadline REAL NOT NULL,
    answers_json TEXT NOT NULL DEFAULT '{}',
    confidence_json TEXT NOT NULL DEFAULT '{}',
    question_index INTEGER NOT NULL DEFAULT 0,
    submitted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_key, user_id)
);
