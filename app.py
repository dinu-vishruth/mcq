# app.py
#
# Application entry point. Kept intentionally thin after the Phase 6 blueprint
# split: it builds the Flask `app` instance (which vercel.json's app.py target
# and Procfile's `gunicorn app:app` both import), runs DB init/migrations, and
# registers the route blueprints from core/routes/. All request handlers live in
# core/routes/{auth,teacher,student,documents}.py.
import os
import sqlite3
from datetime import timedelta
from flask import Flask, request, session, redirect
from flask_wtf.csrf import CSRFProtect
from config import SECRET_KEY, DB_PATH, UPLOAD_FOLDER

app = Flask(__name__, static_folder="static")
app.secret_key = SECRET_KEY
# Use Flask's default signed-cookie sessions. Server-side filesystem sessions
# (flask_session) don't work on Vercel: each request may hit a different
# ephemeral instance with its own /tmp, so the CSRF token written on the login
# GET is missing on the POST. Signed cookies travel with the client and verify
# on any instance. Keep the session payload small (session_key only, never the
# full mcqs list) so it stays under the ~4KB cookie limit.
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("VERCEL"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
csrf = CSRFProtect(app)

@app.before_request
def handle_before_request():
    # Enforce SSL on Vercel
    if os.environ.get("VERCEL") and request.headers.get("X-Forwarded-Proto", "http") != "https":
        url = request.url.replace("http://", "https://", 1)
        return redirect(url, code=301)
    
    # Make sessions permanent
    session.permanent = True

# ---- DB helpers ----
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = get_db()
    
    # Check if schema file exists, else use fallback inline SQL
    try:
        if os.path.exists("database/schema.sql"):
            with open("database/schema.sql", "r") as f:
                conn.executescript(f.read())
        else:
            conn.executescript("""
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
                user_id INTEGER,
                score INTEGER,
                total INTEGER,
                submitted_at TEXT,
                detail_json TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """)
    except Exception as e:
        print(f"[WARNING] Schema execution error: {e}")
        
    # Dynamically alter tables to add new columns if they do not exist
    try:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN archived INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    try:
        conn.execute("ALTER TABLE results ADD COLUMN time_spent INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    # Create login_attempts table for rate limiting
    try:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip TEXT,
            username TEXT,
            attempt_time TEXT,
            success INTEGER
        )
        """)
        conn.commit()
    except Exception as e:
        print(f"[WARNING] Failed to create login_attempts table: {e}")
    
    # Seed default user accounts so they work out-of-the-box on ephemeral Vercel containers
    try:
        from werkzeug.security import generate_password_hash
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        if count == 0:
            teacher_hash = generate_password_hash("teacher123")
            student_hash = generate_password_hash("student123")
            cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                        ("teacher", teacher_hash, "teacher"))
            cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                        ("student", student_hash, "student"))
            conn.commit()
    except Exception as e:
        print(f"[WARNING] Failed to seed default users: {e}")

    # Additive agentic/RAG schema (documents, chunks, embeddings, chat_history,
    # learning_history, weak_topics + sessions.document_id). Idempotent; never
    # touches existing tables/columns. Wrapped so a failure here can never stop
    # the app from booting with its legacy tables intact.
    try:
        from core.models.migrations import run_migrations
        run_migrations(conn)
    except Exception as e:
        print(f"[WARNING] Agentic/RAG migrations skipped: {e}")

    conn.close()

# Execute table creations automatically when app boots on Vercel
init_db()

# ---- Routes ----
# Routes live in core/routes/ blueprints (Phase 6). app.py stays the entry point
# and keeps the module-level `app` instance so vercel.json (app.py) and Procfile
# (gunicorn app:app) are unchanged. Endpoints are blueprint-namespaced
# (auth.*, teacher.*, student.*, documents.*).
from core.routes import register_blueprints
register_blueprints(app)


# init & run
if __name__ == "__main__":
    app.run(debug=True)