-- Preserve legacy labels, but never infer access rights from a reusable name.
-- Existing NULL ownership needs an explicit, reviewed mapping before release.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_user_id INTEGER REFERENCES users(id);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS teacher_user_id INTEGER REFERENCES users(id);
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_doc_hash_owner_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_owner_id ON documents(doc_hash, owner_user_id);
CREATE INDEX IF NOT EXISTS idx_documents_owner_id ON documents(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_teacher_id ON sessions(teacher_user_id);
ALTER TABLE results ADD COLUMN IF NOT EXISTS metadata_json TEXT DEFAULT '{}';
