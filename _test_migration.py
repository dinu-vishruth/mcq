import os, shutil, sqlite3, tempfile
tmp = tempfile.mkdtemp(prefix="mig_")
db = os.path.join(tmp, "mcq.db")
shutil.copy("database/mcq.db", db)   # real production data

def snap(path):
    c = sqlite3.connect(path); c.row_factory = sqlite3.Row
    docs = [dict(r) for r in c.execute("SELECT * FROM documents ORDER BY id")]
    chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    orphans = c.execute("SELECT COUNT(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)").fetchone()[0]
    ddl = " ".join(c.execute("SELECT sql FROM sqlite_master WHERE name='documents'").fetchone()[0].split())
    tables = sorted(r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    c.close()
    return docs, chunks, orphans, ddl, tables

before = snap(db)
print("BEFORE docs:", len(before[0]), "chunks:", before[1], "orphans:", before[2])
print("BEFORE ddl has global UNIQUE:", "doc_hash TEXT UNIQUE" in before[3])

import config; config.DB_PATH = db
from core.models.migrations import run_migrations
c = sqlite3.connect(db); run_migrations(c); c.close()

after = snap(db)
print("AFTER  docs:", len(after[0]), "chunks:", after[1], "orphans:", after[2])
print("AFTER  ddl:", after[3][:120], "...")
print("AFTER  has UNIQUE(doc_hash, owner):", "UNIQUE(doc_hash, owner)" in after[3])

assert before[0] == after[0], "DOCUMENT ROWS CHANGED!"
assert before[1] == after[1], "CHUNK COUNT CHANGED!"
assert after[2] == 0, "ORPHANED CHUNKS!"
assert before[4] == after[4], f"TABLE SET CHANGED: {set(before[4]) ^ set(after[4])}"
print("\n[OK] all rows + ids + chunks preserved, no orphans, no table loss")

# Idempotency: a second run must be a no-op.
c = sqlite3.connect(db); run_migrations(c); run_migrations(c); c.close()
assert snap(db)[0] == after[0], "NOT IDEMPOTENT"
print("[OK] idempotent across repeat runs")

# The new constraint actually holds, and cross-owner duplicates are now allowed.
c = sqlite3.connect(db)
c.execute("INSERT INTO documents (doc_hash, owner, title) VALUES ('h1','u1','a')"); c.commit()
c.execute("INSERT INTO documents (doc_hash, owner, title) VALUES ('h1','u2','b')"); c.commit()
print("[OK] same hash, different owners accepted")
try:
    c.execute("INSERT INTO documents (doc_hash, owner, title) VALUES ('h1','u1','dup')"); c.commit()
    print("[FAIL] duplicate (hash, owner) was accepted")
except sqlite3.IntegrityError:
    print("[OK] duplicate (hash, owner) still rejected")
c.close()
