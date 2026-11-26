ALTER TABLE users ADD COLUMN IF NOT EXISTS is_disabled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_seen_at DOUBLE PRECISION;
CREATE TABLE app_controls (name TEXT PRIMARY KEY, enabled INTEGER NOT NULL CHECK(enabled IN (0,1)));
INSERT INTO app_controls VALUES ('registration',1),('demo',1),('generation',1);
CREATE TABLE access_events (
 id BIGSERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
 username TEXT NOT NULL, event TEXT NOT NULL, ip TEXT, occurred_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX idx_access_events_time ON access_events(occurred_at);
CREATE TABLE admin_audit (
 id BIGSERIAL PRIMARY KEY, actor_email TEXT NOT NULL, action TEXT NOT NULL,
 target TEXT NOT NULL, occurred_at DOUBLE PRECISION NOT NULL
);
