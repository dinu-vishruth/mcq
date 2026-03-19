# app.py
import os
import json
import uuid
import sqlite3
import random
from datetime import datetime
from flask import Flask, render_template, request, session, redirect, url_for, send_file, flash
from flask_session import Session
from config import SECRET_KEY, DB_PATH, UPLOAD_FOLDER, ALLOWED_EXT
from models.pdf_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx
from models.mcq_generator import generate_mcqs
from models.explanation_engine import explain_answers
from utils.session_manager import create_session_key, validate_session_key
from utils.text_cleaner import clean_text

app = Flask(__name__, static_folder="static")
app.secret_key = SECRET_KEY
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "/tmp/flask_session" if os.environ.get("VERCEL") else "flask_session"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
Session(app)

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
    with open("database/schema.sql", "r") as f:
        conn.executescript(f.read())
    conn.commit()
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

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]
        
        if user["role"] == "teacher":
            return redirect(url_for("teacher_dashboard"))
        return redirect(url_for("student_dashboard"))
    
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
        "SELECT session_key, created_at, difficulty, timer FROM sessions WHERE teacher=? ORDER BY created_at DESC",
        (session.get("username"),),
    )
    rows = cur.fetchall()
    conn.close()
    return render_template("teacher_dashboard.html", sessions=rows)

# Student dashboard
@app.route("/student")
def student_dashboard():
    if session.get("role") != "student":
        return redirect(url_for("home"))
    
    conn = get_db()
    # Fetch past results for this user
    history = conn.execute("SELECT * FROM results WHERE user_id=? ORDER BY submitted_at DESC", 
                           (session.get("user_id"),)).fetchall()
    conn.close()
    
    return render_template("student_dashboard.html", history=history)



@app.route("/upload", methods=["GET", "POST"])
def upload():
    if session.get("role") not in ("teacher", "student"):
        return redirect(url_for("home"))
    if request.method == "GET":
        return render_template("upload.html")

    f = request.files.get("file")
    if not f or "." not in f.filename:
        return render_template("upload.html", error="Upload a valid file.")
    ext = f.filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("upload.html", error="Unsupported file type.")

    # Extract text from the uploaded file
    if ext == "pdf":
        text = extract_text_from_pdf(f)
    elif ext == "docx":
        path = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.docx")
        f.save(path)
        text = extract_text_from_docx(path)
        os.remove(path)
    elif ext == "pptx":
        path = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.pptx")
        f.save(path)
        text = extract_text_from_pptx(path)
        os.remove(path)
    else:
        text = f.read().decode("utf-8", errors="ignore")

    text = clean_text(text)

    num_questions = int(request.form.get("num_questions", 5))
    difficulty = request.form.get("difficulty", "medium")
    timer = int(request.form.get("timer", 60))

    # Generate MCQs using Gemini AI with difficulty-aware prompts
    mcqs = generate_mcqs(text, num_questions=num_questions, difficulty=difficulty)

    if not mcqs:
        return render_template("upload.html", error="Could not generate questions. Please check your API key or try a different file.")

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
    if request.method == "GET":
        return render_template("student_dashboard.html")
    key = request.form.get("session_key", "").strip()
    if not validate_session_key(key):
        return render_template("student_dashboard.html", error="Invalid session key")
    # Load MCQs from DB
    conn = get_db()
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
    mcqs = session.get("mcqs")
    timer = session.get("timer", 60)
    if not mcqs:
        return redirect(url_for("student_login"))

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
    randomized = session.get("mcqs_randomized")
    if not randomized:
        return redirect(url_for("student_login"))

    student_name = request.form.get("student_name", "").strip() or session.get("username", "Student")
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
    conn = get_db()
    cur = conn.cursor()
    conn.execute(
        "INSERT INTO results (session_key, student_name, user_id, score, total, submitted_at, detail_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            session.get("session_key", ""),
            student_name,
            session.get("user_id"),  # Save the logged-in user's ID
            score,
            total,
            datetime.utcnow().isoformat(),
            json.dumps(details),
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

# init & run
if __name__ == "__main__":
    app.run(debug=True)