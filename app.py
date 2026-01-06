# app.py
import os
import json
import uuid
import sqlite3
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
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
Session(app)

# ---- DB helpers ----
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists("database"):
        os.makedirs("database", exist_ok=True)
    conn = get_db()
    with open("database/schema.sql", "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

# ---- Routes ----
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    name = request.form.get("username", "").strip()
    role = request.form.get("role")
    if not name or role not in ("teacher", "student"):
        return render_template("login.html", error="Enter name and role.")
    session["username"] = name
    session["role"] = role
    if role == "teacher":
        return redirect(url_for("teacher_dashboard"))
    return redirect(url_for("student_login"))

# Teacher dashboard
@app.route("/teacher")
def teacher_dashboard():
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    # show teacher's sessions
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT session_key, created_at, difficulty, timer FROM sessions WHERE teacher=? ORDER BY created_at DESC", (session.get("username"),))
    rows = cur.fetchall()
    conn.close()
    return render_template("teacher_dashboard.html", sessions=rows)

@app.route("/upload", methods=["GET","POST"])
def upload():
    if session.get("role") != "teacher":
        return redirect(url_for("home"))
    if request.method == "GET":
        return render_template("upload.html")
    f = request.files.get("file")
    if not f or '.' not in f.filename:
        return render_template("upload.html", error="Upload a valid file.")
    ext = f.filename.rsplit(".",1)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("upload.html", error="Unsupported file type.")
    # extract text
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
    prefer_local_t5 = True  # using local AI in this project
    # Generate MCQs
    mcqs = generate_mcqs(text, num_questions=num_questions, difficulty=difficulty)
    # save session
    session_key = create_session_key(teacher=session.get("username"), difficulty=difficulty, timer=timer, mcqs=mcqs)
    return render_template("report_generated.html", session_key=session_key, mcqs=mcqs)

# Student login by session key
@app.route("/student_login", methods=["GET","POST"])
def student_login():
    if request.method=="GET":
        return render_template("student_dashboard.html")
    key = request.form.get("session_key","").strip()
    if not validate_session_key(key):
        return render_template("student_dashboard.html", error="Invalid session key")
    # load mcqs from DB
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT mcqs_json, timer FROM sessions WHERE session_key=?", (key,))
    row = cur.fetchone()
    conn.close()
    mcqs = json.loads(row[0])
    # store for the student session
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
    # shuffle questions and options to reduce cheating
    import random
    randomized = []
    for q in mcqs:
        opts = list(q["options"])
        random.shuffle(opts)
        randomized.append({"question": q["question"], "options": opts, "answer_text": q["answer_text"]})
    session["mcqs_randomized"] = randomized
    return render_template("mcq_test.html", mcqs=randomized, timer=timer)

@app.route("/submit", methods=["POST"])
def submit():
    randomized = session.get("mcqs_randomized")
    if not randomized:
        return redirect(url_for("student_login"))
    student_name = request.form.get("student_name", session.get("username","Student"))
    total = len(randomized)
    score = 0
    details = []
    for i, q in enumerate(randomized):
        sel = request.form.get(f"q-{i}", "")
        is_correct = (sel == q["answer_text"])
        if is_correct:
            score += 1
        details.append({"question": q["question"], "selected": sel, "correct": q["answer_text"], "is_correct": is_correct})
    # explanations (local)
    explanations = explain_answers(details)
    # save results
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO results (session_key, student_name, score, total, submitted_at, detail_json) VALUES (?, ?, ?, ?, ?, ?)",
                (session.get("session_key",""), student_name, score, total, datetime.utcnow().isoformat(), json.dumps(details)))
    conn.commit()
    conn.close()
    return render_template("result.html", score=score, total=total, details=details, explanations=explanations)

@app.route("/download_report/<session_key>")
def download_report(session_key):
    # generate a simple PDF report with results for this session (teacher's quick view)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT mcqs_json, teacher, difficulty, timer FROM sessions WHERE session_key=?", (session_key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return "Invalid session key", 404
    mcqs = json.loads(row[0])
    # make a text-based PDF using reportlab
    from reportlab.pdfgen import canvas
    filename = f"reports/report_{session_key}.pdf"
    os.makedirs("reports", exist_ok=True)
    c = canvas.Canvas(filename)
    y = 800
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"MCQ Set - {session_key}")
    y -= 30
    c.setFont("Helvetica", 11)
    for idx, q in enumerate(mcqs):
        c.drawString(40, y, f"Q{idx+1}. {q['question']}")
        y -= 18
        for lbl, opt in q["options"]:
            c.drawString(60, y, f"{lbl}) {opt}")
            y -= 14
        y -= 8
        if y < 80:
            c.showPage()
            y = 800
    c.save()
    return send_file(filename, as_attachment=True)

# init & run
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
 