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

import config
from config import UPLOAD_FOLDER, ALLOWED_EXT
from core.models.db import get_db
from models.pdf_processor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx
import models.mcq_generator as mcq_generator
from utils.session_manager import create_session_key
from utils.text_cleaner import clean_text

documents_bp = Blueprint("documents", __name__)


class _ExtractError(Exception):
    """Carries a user-safe message when text can't be pulled from an upload."""


def _extract_upload_text(form, files):
    """Resolve document text + title + source ext from an upload request.

    Prefers client-extracted `extracted_text` (browser PDF.js/Mammoth path,
    which avoids the 4.5 MB server cap); otherwise reads the posted file
    server-side (pdf/docx/pptx/txt). Shared by the quiz-upload and the
    store-only resource-upload routes so extraction behaves identically.

    Returns (clean_text, title, source_ext). Raises _ExtractError with a
    user-facing message on any problem.
    """
    source_title = (form.get("title") or "").strip()
    source_ext = "paste"

    extracted_text = (form.get("extracted_text") or "").strip()
    if extracted_text:
        text = extracted_text
    else:
        f = files.get("file")
        if not f or "." not in f.filename:
            raise _ExtractError("Please upload a valid file.")

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        if file_size > 4.5 * 1024 * 1024:
            raise _ExtractError("File is too large! Vercel limits file uploads to 4.5 MB. Please compress or split your file.")

        ext = f.filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXT:
            raise _ExtractError("Unsupported file type.")
        source_ext = ext
        if not source_title:
            source_title = f.filename.rsplit(".", 1)[0].strip() or f.filename

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
            raise _ExtractError(f"Failed to extract text from the file: {str(e)}. Please check if the file is corrupted.")

    text = clean_text(text)
    if not text or not text.strip():
        raise _ExtractError("Could not extract any readable text. Please check the file contents.")
    return text, (source_title or "Untitled source"), source_ext


@documents_bp.route("/ingest_resource", methods=["POST"])
def ingest_resource():
    """Store + index an uploaded document as a saved resource — WITHOUT
    generating a quiz. Backs the Learning Journey 'Add Resource' page. On
    success it redirects to /journey (the library), never to a quiz. Ingestion
    failure here is a real error (unlike /upload where it's best-effort), since
    storing the resource is the whole point of this route."""
    if not session.get("user_id"):
        return redirect(url_for("auth.home"))

    try:
        text, title, source_ext = _extract_upload_text(request.form, request.files)
    except _ExtractError as e:
        return render_template("add_resource.html", error=str(e))

    try:
        from core.services.ingestion_service import ingest_document
        ingest_document(text, owner=session.get("username", ""), title=title, source_type=source_ext)
    except Exception as e:
        return render_template("add_resource.html",
                               error=f"Could not save this resource: {str(e)}. Please try another file.")

    return redirect(url_for("student.journey"))


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

    # Rate limiting on MCQ Generation — protects the LLM from request bursts.
    # Only counts sessions that actually triggered generation: weak-topic REVIEW
    # sessions (difficulty='review') are rebuilt from stored data with no LLM
    # call, so they must not count against this limit. Configurable + lenient by
    # default so a single user iterating on their own material isn't blocked.
    cooldown = config.GENERATION_COOLDOWN_SECONDS
    max_per_window = config.GENERATION_MAX_PER_5MIN
    if cooldown > 0:
        conn = get_db()
        recent_sessions = conn.execute(
            "SELECT created_at FROM sessions WHERE teacher=? AND COALESCE(difficulty,'') != 'review' "
            "ORDER BY created_at DESC LIMIT ?",
            (session.get("username"), max_per_window)
        ).fetchall()
        conn.close()

        if recent_sessions:
            now = datetime.utcnow()
            most_recent = datetime.fromisoformat(recent_sessions[0]["created_at"])
            if (now - most_recent).total_seconds() < cooldown:
                return render_template("upload.html", error=f"Please wait a few seconds between generation requests.")

            if len(recent_sessions) >= max_per_window:
                oldest = datetime.fromisoformat(recent_sessions[-1]["created_at"])
                if (now - oldest).total_seconds() < 300:
                    return render_template("upload.html", error=f"You can only generate {max_per_window} quiz sets every 5 minutes. Please wait a moment.")

    try:
        text, source_title, source_ext = _extract_upload_text(request.form, request.files)
    except _ExtractError as e:
        return render_template("upload.html", error=str(e))

    # NOTE: the instant-quiz upload does NOT save a Learning Journey resource.
    # Only documents the user explicitly adds via /ingest_resource are stored in
    # their library. This keeps one-off quiz material out of the saved resources.

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
    # Skip preview page - go directly to quiz. Only the session_key goes in the
    # cookie; mcqs + timer are re-fetched from the DB (they're already persisted
    # by create_session_key) so the signed-cookie session stays under ~4KB.
    session["session_key"] = session_key
    return redirect(url_for("student.mcq_test"))
