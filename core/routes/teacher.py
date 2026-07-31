"""
Teacher blueprint: dashboard, session report, PDF download, CSV export, and the
session lifecycle actions (delete/archive/unarchive/clone/edit). Verbatim move
from app.py; url_for targets are blueprint-namespaced. Route paths unchanged.
"""
from __future__ import annotations

import os
import json
import uuid
from datetime import datetime

from flask import (Blueprint, render_template, request, session, redirect,
                   url_for, send_file, flash, Response)

from core.models.db import get_db

teacher_bp = Blueprint("teacher", __name__)


@teacher_bp.route("/teacher")
def teacher_dashboard():
    # Legacy teacher dashboard removed. Everyone is a single "User"; the unified
    # dashboard is the home. Kept as a redirect so old links never 404.
    return redirect(url_for("student.dashboard"))


@teacher_bp.route("/session_report/<session_key>")
def session_report(session_key):
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))

    conn = get_db()
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


@teacher_bp.route("/download_report/<session_key>")
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
        q_text = f"Q{idx+1}. {q['question']}"
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


@teacher_bp.route("/delete_session/<session_key>", methods=["POST"])
def delete_session(session_key):
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))
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
    return redirect(url_for("teacher.teacher_dashboard"))


@teacher_bp.route("/archive_session/<session_key>", methods=["POST"])
def archive_session(session_key):
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))
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
    return redirect(url_for("teacher.teacher_dashboard"))


@teacher_bp.route("/unarchive_session/<session_key>", methods=["POST"])
def unarchive_session(session_key):
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))
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
    return redirect(url_for("teacher.teacher_dashboard"))


@teacher_bp.route("/clone_session/<session_key>", methods=["POST"])
def clone_session(session_key):
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))
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
    return redirect(url_for("teacher.teacher_dashboard"))


@teacher_bp.route("/edit_session/<session_key>", methods=["GET", "POST"])
def edit_session(session_key):
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))
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

    try:
        timer = int(request.form.get("timer", 60))
        if timer < 10 or timer > 3600:
            raise ValueError("Timer must be between 10 and 3600 seconds.")
        difficulty = request.form.get("difficulty", "medium").strip().lower()
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("Invalid difficulty value.")

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

            updated_mcqs.append({"question": q_text, "options": options, "answer_text": ans_text})

        conn.execute(
            "UPDATE sessions SET timer=?, difficulty=?, mcqs_json=? WHERE session_key=?",
            (timer, difficulty, json.dumps(updated_mcqs), session_key)
        )
        conn.commit()
        conn.close()
        flash("Session updated successfully!", "success")
        return redirect(url_for("teacher.teacher_dashboard"))

    except ValueError as e:
        conn.close()
        mcqs = json.loads(s["mcqs_json"])
        return render_template("edit_session.html", session_key=session_key, timer=s["timer"], difficulty=s["difficulty"], mcqs=mcqs, error=str(e))


@teacher_bp.route("/export_results/<session_key>")
def export_results(session_key):
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))
    if not session_key.isalnum():
        return "Invalid session key format", 400

    conn = get_db()
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
