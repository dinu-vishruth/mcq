-- SQLite cannot remove the old UNIQUE(doc_hash, owner) constraint in place.
-- Keep every existing row and its ID; legacy ownership remains unassigned.
CREATE TABLE documents_owned (
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT, owner TEXT, title TEXT,
    source_type TEXT, char_count INTEGER DEFAULT 0, chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    meta_json TEXT, owner_user_id INTEGER REFERENCES users(id),
    UNIQUE(doc_hash, owner_user_id)
);
INSERT INTO documents_owned(id,doc_hash,owner,title,source_type,char_count,chunk_count,status,created_at,meta_json)
SELECT id,doc_hash,owner,title,source_type,char_count,chunk_count,status,created_at,meta_json FROM documents;
DROP TABLE documents;
ALTER TABLE documents_owned RENAME TO documents;
ALTER TABLE sessions ADD COLUMN teacher_user_id INTEGER REFERENCES users(id);
CREATE INDEX idx_documents_owner_id ON documents(owner_user_id);
CREATE INDEX idx_sessions_teacher_id ON sessions(teacher_user_id);
ALTER TABLE results ADD COLUMN metadata_json TEXT DEFAULT '{}';
