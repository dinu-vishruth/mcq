ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS identity_email_conflicts (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    former_email TEXT NOT NULL,
    PRIMARY KEY (user_id, former_email)
);
CREATE TABLE IF NOT EXISTS retired_usernames (name TEXT PRIMARY KEY);

-- Previous automatic linking left the local password valid on some accounts.
UPDATE users SET password_hash='!'
WHERE NULLIF(google_id, '') IS NOT NULL OR NULLIF(github_id, '') IS NOT NULL;

-- A stored legacy email was editable, so duplicates have no proven owner.
-- Preserve the old values for recovery and let a future verified sign-in claim
-- the address. User IDs, passwords, and owned content keep their original rows.
INSERT INTO identity_email_conflicts (user_id, former_email)
SELECT id, email FROM users
WHERE lower(trim(email)) IN (
    SELECT lower(trim(email)) FROM users WHERE trim(email) <> ''
    GROUP BY lower(trim(email)) HAVING count(*) > 1
)
ON CONFLICT (user_id, former_email) DO NOTHING;

UPDATE users SET email=NULL, email_verified=0
WHERE id IN (SELECT user_id FROM identity_email_conflicts);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_normalized_email
    ON users (lower(trim(email))) WHERE trim(email) <> '';
