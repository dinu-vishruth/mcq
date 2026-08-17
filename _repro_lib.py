import os, tempfile, sqlite3
_TMP = tempfile.mkdtemp(prefix="repro_lib_")
_DB = os.path.join(_TMP, "r.db"); _UP = os.path.join(_TMP, "uploads")
os.makedirs(_UP, exist_ok=True)
import config
config.DB_PATH = _DB; config.UPLOAD_FOLDER = _UP
config.EMBEDDING_BACKEND = "hashing"; config.VECTOR_STORE = "sqlite"
import app as app_module
import utils.session_manager as sm; sm.DB_PATH = _DB
from fastapi_app.database import init_db; init_db()
from config import SECRET_KEY
from tests._client import make_client_factory
factory = make_client_factory(app_module.app, SECRET_KEY)

TEXT = ("Normalization organizes tables to reduce redundancy. "
        "First normal form removes repeating groups. Second normal form removes partial "
        "dependencies. Third normal form removes transitive dependencies. ") * 12

def signup(c, u):
    r = c.post("/signup", data={"username": u, "email": f"{u}@e.com", "password": "Passw0rd!x", "confirm_password": "Passw0rd!x"})
    r2 = c.post("/login", data={"username": u, "password": "Passw0rd!x"})
    print(f"  signup {u}: {r.status_code}  login: {r2.status_code} -> {r2.headers.get('location')}")

def ingest(c, title, text=TEXT):
    r = c.post("/ingest_resource", data={"title": title, "extracted_text": text})
    print(f"  ingest '{title}': {r.status_code} -> {r.headers.get('location')}")
    return r

def knowledge(c):
    r = c.get("/api/knowledge")
    items = r.json().get("items", []) if r.status_code == 200 else None
    print(f"  /api/knowledge: {r.status_code} items={len(items) if items is not None else 'ERR'}")
    return items

print("=== USER A: first resource ===")
a = factory(); signup(a, "alice")
ingest(a, "DBMS Notes")
ia = knowledge(a)
print("   ", [i['title'] for i in (ia or [])])

print("=== USER B: SAME content (dedup path) ===")
b = factory(); signup(b, "bob")
ingest(b, "Bob Notes")
ib = knowledge(b)
print("   ", [i['title'] for i in (ib or [])])

print("=== USER A: re-add same content ===")
ingest(a, "DBMS Notes again")
print("   ", [i['title'] for i in (knowledge(a) or [])])

print("=== RAW documents table ===")
c = sqlite3.connect(_DB); c.row_factory = sqlite3.Row
for r in c.execute("SELECT id,owner,title,status,chunk_count FROM documents").fetchall():
    print("   ", dict(r))
