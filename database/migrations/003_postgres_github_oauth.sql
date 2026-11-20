ALTER TABLE users ADD COLUMN IF NOT EXISTS github_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_github_id
    ON users(github_id) WHERE github_id IS NOT NULL;

-- OAuth and password registration now use the unified 'user' role.
-- Preserve existing teacher/student accounts while allowing new accounts.
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check
    CHECK (role IN ('user', 'teacher', 'student'));
