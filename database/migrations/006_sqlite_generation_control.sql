CREATE TABLE IF NOT EXISTS generation_control (id INTEGER PRIMARY KEY);
INSERT INTO generation_control(id) VALUES (1) ON CONFLICT(id) DO NOTHING;
CREATE TABLE IF NOT EXISTS generation_leases (
 scope TEXT PRIMARY KEY, token TEXT NOT NULL, expires_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS generation_admissions (
 token TEXT PRIMARY KEY, username TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generation_admissions_user ON generation_admissions(username, created_at);
