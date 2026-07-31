"""
Documents blueprint: the /upload route (shared by teachers and students).
Verbatim move from app.py; url_for targets are blueprint-namespaced.

mcq_generator is imported as a MODULE (not the function) so tests patching
models.mcq_generator.generate_mcqs at the source take effect here. This is also
where AI_PIPELINE (legacy vs rag) is honoured, since generate_mcqs dispatches.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

from flask import (Blueprint, render_template, request, session, redirect, url_for)

from config import UPLOAD_FOLDER, ALLOWED_EXT
from core.models.db import get_db
from models.pdf_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx
import models.mcq_generator as mcq_generator
from utils.session_manager import create_session_key
from utils.text_cleaner import clean_text

documents_bp = Blueprint("documents", __name__)


@documents_bp.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))
    if request.method == "GET":
        return render_template("upload.html")

    try:
        num_questions = int(request.form.get("num_questions", 10))
        # The form submits the time limit in MINUTES; sessions store seconds.
        timer_minutes = int(request.form.get("timer", 10))
        difficulty = request.form.get("difficulty", "medium").strip().lower()

        if num_questions < 1 or num_questions > 30:
            raise ValueError("Number of questions must be between 1 and 30.")
        if timer_minutes < 1 or timer_minutes > 180:
            raise ValueError("Time limit must be between 1 and 180 minutes.")
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("Invalid difficulty level selected.")

        timer_seconds = timer_minutes * 60
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
        most_recent = datetime.fromisoformat(recent_sessions[0]["created_at"])
        if (now - most_recent).total_seconds() < 10:
            return render_template("upload.html", error="Rate limit exceeded. Please wait 10 seconds between generation requests.")

        if len(recent_sessions) >= 5:
            oldest_of_five = datetime.fromisoformat(recent_sessions[-1]["created_at"])
            if (now - oldest_of_five).total_seconds() < 300:
                return render_template("upload.html", error="Rate limit exceeded. You can only generate 5 MCQ sets every 5 minutes.")

    extracted_text = request.form.get("extracted_text", "").strip()
    if extracted_text:
        text = extracted_text
    else:
        f = request.files.get("file")
        if not f or "." not in f.filename:
            return render_template("upload.html", error="Please upload a valid file.")

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        if file_size > 4.5 * 1024 * 1024:
            return render_template("upload.html", error="File is too large! Vercel limits file uploads to 4.5 MB. Please compress or split your file.")

        ext = f.filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXT:
            return render_template("upload.html", error="Unsupported file type.")

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

    try:
        mcqs = mcq_generator.generate_mcqs(text, num_questions=num_questions, difficulty=difficulty)
    except mcq_generator.MCQGenerationError as mge:
        return render_template("upload.html", error=str(mge))
    except Exception as e:
        return render_template("upload.html", error=f"An unexpected error occurred during MCQ generation: {str(e)}")

    if not mcqs:
        return render_template("upload.html", error="No questions were generated. Please try again with a different text.")

    session_key = create_session_key(
        teacher=session.get("username"),
        difficulty=difficulty,
        timer=timer_seconds,
        mcqs=mcqs,
    )
    # Skip preview page - go directly to quiz
    session["mcqs"] = mcqs
    session["timer"] = timer_seconds
    session["session_key"] = session_key
    return redirect(url_for("student.mcq_test"))
