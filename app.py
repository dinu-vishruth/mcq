# app.py
import os
import json
import uuid
import sqlite3
import random
from datetime import datetime, timedelta
from flask import Flask, render_template, request, session, redirect, url_for, send_file, flash
from flask_session import Session
from flask_wtf.csrf import CSRFProtect
from config import SECRET_KEY, DB_PATH, UPLOAD_FOLDER, ALLOWED_EXT
from models.pdf_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx
from models.mcq_generator import generate_mcqs, MCQGenerationError
from models.explanation_engine import explain_answers
from utils.session_manager import create_session_key, validate_session_key
from utils.text_cleaner import clean_text

app = Flask(__name__, static_folder="static")
app.secret_key = SECRET_KEY
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "/tmp/flask_session" if os.environ.get("VERCEL") else "flask_session"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
Session(app)
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
        
    conn.close()

from werkzeug.security import generate_password_hash, check_password_hash

# Execute table creations automatically when app boots on Vercel
init_db()

# ---- Routes ----
@app.route("/")
def home():
    if "user_id" in session:
        if session.get("role") == "teacher":
            return redirect(url_for("teacher_dashboard"))
        return redirect(url_for("student_dashboard"))
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    role = request.form.get("role")

    if not username or not password or role not in ("teacher", "student"):
        return render_template("signup.html", error="All fields are required.")

    if len(password) < 8:
        return render_template("signup.html", error="Password must be at least 8 characters long.")

    hashed_pw = generate_password_hash(password)

    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                     (username, hashed_pw, role))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return render_template("signup.html", error="Username already exists.")
    conn.close()

    return redirect(url_for("home"))

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    ip = request.remote_addr

    conn = get_db()
    
    # Check failed login attempts in last 5 minutes (for either username or IP)
    five_mins_ago = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    attempts = conn.execute("""
        SELECT COUNT(*) FROM login_attempts 
        WHERE (ip=? OR username=?) AND success=0 AND attempt_time > ?
    """, (ip, username, five_mins_ago)).fetchone()[0]

    if attempts >= 5:
        conn.close()
        return render_template("login.html", error="Too many failed login attempts. Please wait 5 minutes.")

    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

    if user and check_password_hash(user["password_hash"], password):
        # Log successful attempt
        conn.execute("INSERT INTO login_attempts (ip, username, attempt_time, success) VALUES (?, ?, ?, 1)",
                     (ip, username, datetime.utcnow().isoformat()))
        conn.commit()
        
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]
        conn.close()
        
        if user["role"] == "teacher":
            return redirect(url_for("teacher_dashboard"))
        return redirect(url_for("student_dashboard"))
    
    # Log failed attempt
    conn.execute("INSERT INTO login_attempts (ip, username, attempt_time, success) VALUES (?, ?, ?, 0)",
                 (ip, username, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    
    return render_template("login.html", error="Invalid username or password")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# Teacher dashboard
@app.route("/teacher")
def teacher_dashboard():
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT session_key, created_at, difficulty, timer, archived FROM sessions WHERE teacher=? ORDER BY created_at DESC",
        (session.get("username"),),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return render_template("teacher_dashboard.html", sessions=rows)

# Student dashboard
@app.route("/student")
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("home"))
    
    conn = get_db()
    # Fetch past results for this user with difficulty
    history_rows = conn.execute("""
        SELECT r.session_key, r.score, r.total, r.submitted_at, r.time_spent, s.difficulty
        FROM results r
        LEFT JOIN sessions s ON r.session_key = s.session_key
        WHERE r.user_id=? 
        ORDER BY r.submitted_at DESC
    """, (session.get("user_id"),)).fetchall()
    conn.close()

    history = []
    total_quizzes = len(history_rows)
    total_score_pct = 0
    total_time = 0
    best_score_pct = 0
    
    difficulty_stats = {"easy": {"sum": 0, "count": 0}, "medium": {"sum": 0, "count": 0}, "hard": {"sum": 0, "count": 0}}
    
    # Chronological history for chart
    chrono_history = list(reversed(history_rows))
    chart_dates = []
    chart_scores = []
    
    for row in history_rows:
        h_dict = dict(row)
        history.append(h_dict)
        
        pct = (row["score"] / row["total"] * 100) if (row["total"] and row["total"] > 0) else 0
        total_score_pct += pct
        total_time += row["time_spent"] or 0
        if pct > best_score_pct:
            best_score_pct = pct
            
        diff = (row["difficulty"] or "medium").lower()
        if diff in difficulty_stats:
            difficulty_stats[diff]["sum"] += pct
            difficulty_stats[diff]["count"] += 1

    for row in chrono_history:
        pct = (row["score"] / row["total"] * 100) if (row["total"] and row["total"] > 0) else 0
        dt = row["submitted_at"][:16].replace("T", " ")  # YYYY-MM-DD HH:MM
        chart_dates.append(dt)
        chart_scores.append(round(pct, 1))

    avg_score = round(total_score_pct / total_quizzes, 1) if total_quizzes > 0 else 0
    best_score = round(best_score_pct, 1)
    
    avg_easy = round(difficulty_stats["easy"]["sum"] / difficulty_stats["easy"]["count"], 1) if difficulty_stats["easy"]["count"] > 0 else 0
    avg_medium = round(difficulty_stats["medium"]["sum"] / difficulty_stats["medium"]["count"], 1) if difficulty_stats["medium"]["count"] > 0 else 0
    avg_hard = round(difficulty_stats["hard"]["sum"] / difficulty_stats["hard"]["count"], 1) if difficulty_stats["hard"]["count"] > 0 else 0

    return render_template(
        "student_dashboard.html",
        history=history,
        total_quizzes=total_quizzes,
        avg_score=avg_score,
        best_score=best_score,
        total_time=total_time,
        chart_dates=chart_dates,
        chart_scores=chart_scores,
        difficulty_averages=[avg_easy, avg_medium, avg_hard]
    )



@app.route("/upload", methods=["GET", "POST"])
def upload():
    if session.get("role") not in ("teacher", "student"):
        return redirect(url_for("home"))
    if request.method == "GET":
        return render_template("upload.html")

    # Form parameters validation
    try:
        num_questions = int(request.form.get("num_questions", 5))
        timer = int(request.form.get("timer", 60))
        difficulty = request.form.get("difficulty", "medium").strip().lower()
        
        if num_questions < 1 or num_questions > 30:
            raise ValueError("Number of questions must be between 1 and 30.")
        if timer < 10 or timer > 3600:
            raise ValueError("Timer must be between 10 and 3600 seconds.")
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("Invalid difficulty level selected.")
    except ValueError as ve:
        return render_template("upload.html", error=str(ve))

    # Rate limiting on MCQ Generation: 1 request/10s, 5 requests/5min
    conn = get_db()
    recent_sessions = conn.execute(
        "SELECT created_at FROM sessions WHERE teacher=? ORDER BY created_at DESC LIMIT 5",
        (session.get("username"),)
    ).fetchall()
    conn.close()

    if recent_sessions:
        now = datetime.utcnow()
        # 1. 10 second throttle
        most_recent = datetime.fromisoformat(recent_sessions[0]["created_at"])
        if (now - most_recent).total_seconds() < 10:
            return render_template("upload.html", error="Rate limit exceeded. Please wait 10 seconds between generation requests.")
        
        # 2. 5 requests per 5 minutes throttle
        if len(recent_sessions) >= 5:
            oldest_of_five = datetime.fromisoformat(recent_sessions[-1]["created_at"])
            if (now - oldest_of_five).total_seconds() < 300:
                return render_template("upload.html", error="Rate limit exceeded. You can only generate 5 MCQ sets every 5 minutes.")

    # Text extraction and processing
    extracted_text = request.form.get("extracted_text", "").strip()
    if extracted_text:
        text = extracted_text
    else:
        f = request.files.get("file")
        if not f or "." not in f.filename:
            return render_template("upload.html", error="Please upload a valid file.")
        
        # Check file size (Vercel payload limit)
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        if file_size > 4.5 * 1024 * 1024:
            return render_template("upload.html", error="File is too large! Vercel limits file uploads to 4.5 MB. Please compress or split your file.")

        ext = f.filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXT:
            return render_template("upload.html", error="Unsupported file type.")

        # Extract text with error handling
        try:
            if ext == "pdf":
                text = extract_text_from_pdf(f)
            elif ext == "docx":
                path = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.docx")
                f.save(path)
                try:
                    text = extract_text_from_docx(path)
                finally:
                    if os.path.exists(path):
                        os.remove(path)
            elif ext == "pptx":
                path = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.pptx")
                f.save(path)
                try:
                    text = extract_text_from_pptx(path)
                finally:
                    if os.path.exists(path):
                        os.remove(path)
            else:
                text = f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return render_template("upload.html", error=f"Failed to extract text from the file: {str(e)}. Please check if the file is corrupted.")

    text = clean_text(text)
    if not text or not text.strip():
        return render_template("upload.html", error="Could not extract any readable text. Please check the file contents.")

    # Generate MCQs using AI with detailed error propagation
    try:
        mcqs = generate_mcqs(text, num_questions=num_questions, difficulty=difficulty)
    except MCQGenerationError as mge:
        return render_template("upload.html", error=str(mge))
    except Exception as e:
        return render_template("upload.html", error=f"An unexpected error occurred during MCQ generation: {str(e)}")

    if not mcqs:
        return render_template("upload.html", error="No questions were generated. Please try again with a different text.")

    # Save session to DB
    session_key = create_session_key(
        teacher=session.get("username"),
        difficulty=difficulty,
        timer=timer,
        mcqs=mcqs,
    )
    return render_template("report_generated.html", session_key=session_key, mcqs=mcqs)

# Student login by session key
@app.route("/student_login", methods=["GET", "POST"])
def student_login():
    if session.get("role") != "student":
        return redirect(url_for("home"))
        
    if request.method == "GET":
        return render_template("student_dashboard.html")
        
    key = request.form.get("session_key", "").strip()
    if not key.isalnum():
        return render_template("student_dashboard.html", error="Invalid session key format")
        
    if not validate_session_key(key):
        return render_template("student_dashboard.html", error="Invalid session key")
        
    # Check if student has already taken this test
    conn = get_db()
    existing = conn.execute("SELECT 1 FROM results WHERE session_key=? AND user_id=?", 
                            (key, session.get("user_id"))).fetchone()
    if existing:
        conn.close()
        return render_template("student_dashboard.html", error="You have already taken this test.")
        
    # Load MCQs from DB
    cur = conn.cursor()
    cur.execute("SELECT mcqs_json, timer FROM sessions WHERE session_key=?", (key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return render_template("student_dashboard.html", error="Session not found")

    mcqs = json.loads(row[0])
    session["mcqs"] = mcqs
    session["timer"] = row[1]
    session["session_key"] = key
    return redirect(url_for("mcq_test"))

@app.route("/mcq_test", methods=["GET"])
def mcq_test():
    if session.get("role") != "student":
        return redirect(url_for("home"))
        
    mcqs = session.get("mcqs")
    timer = session.get("timer", 60)
    key = session.get("session_key")
    if not mcqs or not key:
        return redirect(url_for("student_login"))

    # Verify student hasn't taken it yet (direct URL access check)
    conn = get_db()
    existing = conn.execute("SELECT 1 FROM results WHERE session_key=? AND user_id=?", 
                            (key, session.get("user_id"))).fetchone()
    conn.close()
    if existing:
        return render_template("student_dashboard.html", error="You have already taken this test.")

    # Shuffle options within each question to reduce cheating
    randomized = []
    for q in mcqs:
        opts = list(q["options"])  # each opt is {"label": "A", "text": "..."}
        random.shuffle(opts)
        # Re-label after shuffling
        for i, opt in enumerate(opts):
            opt["label"] = ["A", "B", "C", "D"][i]
        randomized.append({
            "question": q["question"],
            "options": opts,
            "answer_text": q["answer_text"],
        })

    session["mcqs_randomized"] = randomized
    return render_template("mcq_test.html", mcqs=randomized, timer=timer)

@app.route("/submit", methods=["POST"])
def submit():
    if session.get("role") != "student":
        return redirect(url_for("home"))
        
    randomized = session.get("mcqs_randomized")
    key = session.get("session_key")
    if not randomized or not key:
        return redirect(url_for("student_login"))

    # Verify student hasn't taken it yet (double-submit protection)
    conn = get_db()
    existing = conn.execute("SELECT 1 FROM results WHERE session_key=? AND user_id=?", 
                            (key, session.get("user_id"))).fetchone()
    if existing:
        conn.close()
        return render_template("student_dashboard.html", error="You have already taken this test.")

    student_name = request.form.get("student_name", "").strip() or session.get("username", "Student")
    time_spent = 0
    try:
        time_spent = int(request.form.get("time_spent", 0))
    except ValueError:
        pass

    total = len(randomized)
    score = 0
    details = []

    for i, q in enumerate(randomized):
        sel = request.form.get(f"q-{i}", "")
        is_correct = sel == q["answer_text"]
        if is_correct:
            score += 1
        details.append({
            "question": q["question"],
            "selected": sel,
            "correct": q["answer_text"],
            "is_correct": is_correct,
        })

    # Generate explanations
    explanations = explain_answers(details)

    # Save results to DB
    conn.execute(
        "INSERT INTO results (session_key, student_name, user_id, score, total, submitted_at, detail_json, time_spent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            key,
            student_name,
            session.get("user_id"),
            score,
            total,
            datetime.utcnow().isoformat(),
            json.dumps(details),
            time_spent,
        ),
    )
    conn.commit()
    conn.close()
    return render_template(
        "result.html", score=score, total=total, details=details, explanations=explanations
    )

@app.route("/session_report/<session_key>")
def session_report(session_key):
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    
    conn = get_db()
    # Verify teacher owns this session
    s_chk = conn.execute("SELECT teacher FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    if not s_chk or s_chk["teacher"] != session.get("username"):
        conn.close()
        return "Access Denied", 403

    results = conn.execute("""
        SELECT r.*, u.username as registered_name 
        FROM results r
        LEFT JOIN users u ON r.user_id = u.id
        WHERE r.session_key=? 
        ORDER BY r.score DESC
    """, (session_key,)).fetchall()
    conn.close()
    
    return render_template("session_report.html", session_key=session_key, results=results)

@app.route("/download_report/<session_key>")
def download_report(session_key):
    """Generate a PDF report with the MCQ set for this session."""
    if not session_key.isalnum():
        return "Invalid session key format", 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT mcqs_json, teacher, difficulty, timer FROM sessions WHERE session_key=?",
        (session_key,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Invalid session key", 404

    mcqs = json.loads(row[0])

    from reportlab.pdfgen import canvas

    filename = f"reports/report_{session_key}.pdf"
    os.makedirs("reports", exist_ok=True)
    c = canvas.Canvas(filename)
    y = 800

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"MCQ Set - {session_key}")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Teacher: {row[1]} | Difficulty: {row[2]} | Timer: {row[3]}s")
    y -= 30

    c.setFont("Helvetica", 11)
    for idx, q in enumerate(mcqs):
        # Draw question (handle long text wrapping)
        q_text = f"Q{idx+1}. {q['question']}"
        # Simple word wrap
        max_width = 80
        while len(q_text) > max_width:
            wrap_at = q_text.rfind(" ", 0, max_width)
            if wrap_at == -1:
                wrap_at = max_width
            c.drawString(40, y, q_text[:wrap_at])
            q_text = q_text[wrap_at:].strip()
            y -= 14
        c.drawString(40, y, q_text)
        y -= 18

        # Draw options — options are dicts: {"label": "A", "text": "..."} 
        for opt in q["options"]:
            label = opt.get("label", "?")
            text = opt.get("text", "")
            c.drawString(60, y, f"{label}) {text}")
            y -= 14
        y -= 8

        if y < 80:
            c.showPage()
            y = 800

    c.save()
    return send_file(filename, as_attachment=True)

@app.route("/delete_session/<session_key>", methods=["POST"])
def delete_session(session_key):
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    if not session_key.isalnum():
        return "Invalid session key format", 400
    conn = get_db()
    s_chk = conn.execute("SELECT teacher FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    if not s_chk or s_chk["teacher"] != session.get("username"):
        conn.close()
        return "Access Denied", 403
    conn.execute("DELETE FROM sessions WHERE session_key=?", (session_key,))
    conn.execute("DELETE FROM results WHERE session_key=?", (session_key,))
    conn.commit()
    conn.close()
    flash("Session deleted successfully!", "success")
    return redirect(url_for("teacher_dashboard"))

@app.route("/archive_session/<session_key>", methods=["POST"])
def archive_session(session_key):
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    if not session_key.isalnum():
        return "Invalid session key format", 400
    conn = get_db()
    s_chk = conn.execute("SELECT teacher FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    if not s_chk or s_chk["teacher"] != session.get("username"):
        conn.close()
        return "Access Denied", 403
    conn.execute("UPDATE sessions SET archived=1 WHERE session_key=?", (session_key,))
    conn.commit()
    conn.close()
    flash("Session archived successfully!", "success")
    return redirect(url_for("teacher_dashboard"))

@app.route("/unarchive_session/<session_key>", methods=["POST"])
def unarchive_session(session_key):
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    if not session_key.isalnum():
        return "Invalid session key format", 400
    conn = get_db()
    s_chk = conn.execute("SELECT teacher FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    if not s_chk or s_chk["teacher"] != session.get("username"):
        conn.close()
        return "Access Denied", 403
    conn.execute("UPDATE sessions SET archived=0 WHERE session_key=?", (session_key,))
    conn.commit()
    conn.close()
    flash("Session unarchived successfully!", "success")
    return redirect(url_for("teacher_dashboard"))

@app.route("/clone_session/<session_key>", methods=["POST"])
def clone_session(session_key):
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    if not session_key.isalnum():
        return "Invalid session key format", 400
    conn = get_db()
    s = conn.execute("SELECT * FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    if not s:
        conn.close()
        return "Session not found", 404
    if s["teacher"] != session.get("username"):
        conn.close()
        return "Access Denied", 403
    new_key = str(uuid.uuid4())[:8]
    conn.execute(
        "INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, mcqs_json, archived) VALUES (?, ?, ?, ?, ?, ?, 0)",
        (new_key, session.get("username"), datetime.utcnow().isoformat(), s["difficulty"], s["timer"], s["mcqs_json"])
    )
    conn.commit()
    conn.close()
    flash(f"Session cloned successfully! New key: {new_key}", "success")
    return redirect(url_for("teacher_dashboard"))

@app.route("/edit_session/<session_key>", methods=["GET", "POST"])
def edit_session(session_key):
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    if not session_key.isalnum():
        return "Invalid session key format", 400
    
    conn = get_db()
    s = conn.execute("SELECT * FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    if not s:
        conn.close()
        return "Session not found", 404
    if s["teacher"] != session.get("username"):
        conn.close()
        return "Access Denied", 403
        
    if request.method == "GET":
        mcqs = json.loads(s["mcqs_json"])
        conn.close()
        return render_template("edit_session.html", session_key=session_key, timer=s["timer"], difficulty=s["difficulty"], mcqs=mcqs)
        
    # POST request: save modifications
    try:
        timer = int(request.form.get("timer", 60))
        if timer < 10 or timer > 3600:
            raise ValueError("Timer must be between 10 and 3600 seconds.")
        difficulty = request.form.get("difficulty", "medium").strip().lower()
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("Invalid difficulty value.")
            
        # Parse the edited questions
        mcqs = json.loads(s["mcqs_json"])
        updated_mcqs = []
        for idx in range(len(mcqs)):
            q_text = request.form.get(f"q_{idx}_text", "").strip()
            if not q_text:
                raise ValueError(f"Question {idx+1} cannot be empty.")
                
            options = []
            for label in ["A", "B", "C", "D"]:
                o_text = request.form.get(f"q_{idx}_opt_{label}", "").strip()
                if not o_text:
                    raise ValueError(f"Option {label} of Question {idx+1} cannot be empty.")
                options.append({"label": label, "text": o_text})
                
            ans_text = request.form.get(f"q_{idx}_answer", "").strip()
            opt_texts = [o["text"] for o in options]
            if ans_text not in opt_texts:
                if ans_text in ["A", "B", "C", "D"]:
                    ans_text = options[["A", "B", "C", "D"].index(ans_text)]["text"]
                else:
                    raise ValueError(f"Correct answer for Question {idx+1} must match one of the option texts.")
                    
            updated_mcqs.append({
                "question": q_text,
                "options": options,
                "answer_text": ans_text
            })
            
        conn.execute(
            "UPDATE sessions SET timer=?, difficulty=?, mcqs_json=? WHERE session_key=?",
            (timer, difficulty, json.dumps(updated_mcqs), session_key)
        )
        conn.commit()
        conn.close()
        flash("Session updated successfully!", "success")
        return redirect(url_for("teacher_dashboard"))
        
    except ValueError as e:
        conn.close()
        mcqs = json.loads(s["mcqs_json"])
        return render_template("edit_session.html", session_key=session_key, timer=s["timer"], difficulty=s["difficulty"], mcqs=mcqs, error=str(e))

@app.route("/export_results/<session_key>")
def export_results(session_key):
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    if not session_key.isalnum():
        return "Invalid session key format", 400
    
    conn = get_db()
    # Verify owner
    s_chk = conn.execute("SELECT teacher FROM sessions WHERE session_key=?", (session_key,)).fetchone()
    if not s_chk or s_chk["teacher"] != session.get("username"):
        conn.close()
        return "Access Denied", 403
    
    results = conn.execute("""
        SELECT r.student_name, u.username as registered_name, r.score, r.total, r.submitted_at, r.time_spent 
        FROM results r
        LEFT JOIN users u ON r.user_id = u.id
        WHERE r.session_key=? 
        ORDER BY r.score DESC
    """, (session_key,)).fetchall()
    conn.close()
    
    import csv
    import io
    from flask import Response
    
    def generate():
        data = io.StringIO()
        writer = csv.writer(data)
        writer.writerow(["Student Name", "Username", "Score", "Total Questions", "Percentage", "Time Spent (s)", "Submitted At"])
        yield data.getvalue()
        data.seek(0)
        data.truncate(0)
        
        for r in results:
            percentage = round((r["score"] / r["total"]) * 100, 2) if r["total"] > 0 else 0
            writer.writerow([
                r["student_name"],
                r["registered_name"] or "N/A",
                r["score"],
                r["total"],
                f"{percentage}%",
                r["time_spent"],
                r["submitted_at"]
            ])
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)
            
    response = Response(generate(), mimetype="text/csv")
    response.headers.set("Content-Disposition", "attachment", filename=f"results_{session_key}.csv")
    return response

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if not session.get("user_id"):
        return redirect(url_for("home"))
        
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (session.get("user_id"),)).fetchone()
    
    if request.method == "GET":
        conn.close()
        return render_template("profile.html", user=user)
        
    # POST request: update details
    try:
        new_username = request.form.get("username", "").strip()
        new_email = request.form.get("email", "").strip()
        new_password = request.form.get("password", "").strip()
        
        if not new_username:
            raise ValueError("Username is required.")
            
        # Check uniqueness of username
        if new_username != user["username"]:
            chk = conn.execute("SELECT 1 FROM users WHERE username=?", (new_username,)).fetchone()
            if chk:
                raise ValueError("Username already exists.")
                
        if new_password and len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters long.")
            
        if new_password:
            hashed_pw = generate_password_hash(new_password)
            conn.execute(
                "UPDATE users SET username=?, email=?, password_hash=? WHERE id=?",
                (new_username, new_email, hashed_pw, session.get("user_id"))
            )
        else:
            conn.execute(
                "UPDATE users SET username=?, email=? WHERE id=?",
                (new_username, new_email, session.get("user_id"))
            )
        conn.commit()
        session["username"] = new_username
        conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("home"))
        
    except ValueError as e:
        conn.close()
        return render_template("profile.html", user=user, error=str(e))

@app.route("/delete_account", methods=["POST"])
def delete_account():
    user_id = session.get("user_id")
    role = session.get("role")
    username = session.get("username")
    if not user_id:
        return redirect(url_for("home"))
        
    conn = get_db()
    if role == "teacher":
        # Delete sessions and associated results created by this teacher
        conn.execute("DELETE FROM results WHERE session_key IN (SELECT session_key FROM sessions WHERE teacher=?)", (username,))
        conn.execute("DELETE FROM sessions WHERE teacher=?", (username,))
    else:
        # Student: delete results they have submitted
        conn.execute("DELETE FROM results WHERE user_id=?", (user_id,))
        
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    session.clear()
    flash("Your account has been deleted.", "success")
    return redirect(url_for("home"))

# init & run
if __name__ == "__main__":
    app.run(debug=True)