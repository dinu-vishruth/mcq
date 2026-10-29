"""Document upload: /upload and /ingest_resource both store + index a document.

Port of core/routes/documents.py. Both routes share one extraction helper so
they behave identically on file handling; they differ only in where the user
lands afterwards -- /upload redirects to the library (/knowledge), where quiz
settings are chosen, /ingest_resource back to the journey (/journey).

mcq_generator is imported as a MODULE (not the function) so tests patching
models.mcq_generator.generate_mcqs at the source take effect here. This is also
where AI_PIPELINE (legacy vs rag) is honoured, since generate_mcqs dispatches.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import RedirectResponse

import config
import models.mcq_generator as mcq_generator
from config import ALLOWED_EXT, UPLOAD_FOLDER
from models.pdf_processor import (extract_text_from_docx, extract_text_from_pdf,
                                  extract_text_from_pptx)
from utils.session_manager import create_session_key
from utils.text_cleaner import clean_text

from ..deps import CurrentUser, Db
from ..templating import render
from ..uploads import FileStorageAdapter, size_of

router = APIRouter(tags=["documents"])

#: App-enforced upload ceiling. On Vercel the platform rejected larger bodies
#: before our code ran; on a persistent host nothing does, so this is what keeps
#: a request inside MAX_REQUEST_BODY_BYTES (which CSRFMiddleware buffers whole).
#: Staying under it also means the user gets a useful message instead of a raw 413.
MAX_UPLOAD_BYTES = int(4.5 * 1024 * 1024)


class ExtractError(Exception):
    """Carries a user-safe message when text can't be pulled from an upload."""


def extract_upload_text(
    upload: UploadFile | None, title: str = "", extracted_text: str = ""
) -> tuple[str, str, str]:
    """Resolve document text + title + source ext from an upload.

    Prefers client-extracted ``extracted_text`` (the browser PDF.js/Mammoth path,
    which sidesteps the 4.5 MB server cap); otherwise reads the posted file
    server-side.

    Returns ``(clean_text, title, source_ext)``; raises ExtractError with a
    user-facing message on any problem.
    """
    source_title = (title or "").strip()
    source_ext = "paste"

    client_text = (extracted_text or "").strip()
    if client_text:
        text = client_text
    else:
        if upload is None or not upload.filename or "." not in upload.filename:
            raise ExtractError("Please upload a valid file.")

        if size_of(upload) > MAX_UPLOAD_BYTES:
            raise ExtractError(
                "File is too large! Uploads are limited to 4.5 MB. "
                "Please compress or split your file."
            )

        ext = upload.filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXT:
            raise ExtractError("Unsupported file type.")
        source_ext = ext
        if not source_title:
            source_title = upload.filename.rsplit(".", 1)[0].strip() or upload.filename

        stored = FileStorageAdapter(upload)
        try:
            if ext == "pdf":
                text = extract_text_from_pdf(stored)
            elif ext in ("docx", "pptx"):
                # python-docx and python-pptx need a real path, not a stream.
                path = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.{ext}")
                stored.save(path)
                try:
                    text = (
                        extract_text_from_docx(path) if ext == "docx"
                        else extract_text_from_pptx(path)
                    )
                finally:
                    if os.path.exists(path):
                        os.remove(path)
            else:
                text = stored.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            raise ExtractError(
                f"Failed to extract text from the file: {exc}. "
                "Please check if the file is corrupted."
            )

    text = clean_text(text)
    if not text or not text.strip():
        raise ExtractError("Could not extract any readable text. Please check the file contents.")
    return text, (source_title or "Untitled source"), source_ext


@router.post("/ingest_resource")
def ingest_resource(
    request: Request,
    user: CurrentUser,
    file: UploadFile | None = None,
    title: Annotated[str, Form()] = "",
    extracted_text: Annotated[str, Form()] = "",
):
    """Store + index a document as a saved resource, then return to the library."""
    try:
        text, resolved_title, source_ext = extract_upload_text(file, title, extracted_text)
    except ExtractError as exc:
        return render(request, "add_resource.html", error=str(exc))

    # Indexing runs inline. It used to be handed to a daemon thread so the redirect
    # could return before chunking + embedding finished, but nothing waits for that
    # thread: on a serverless host the function is frozen the moment the response
    # is sent, so the thread died mid-write and the library listed a resource that
    # had never been stored -- which later failed to generate a quiz. Being sync
    # also means a real failure shows up as an error here instead of as a phantom
    # entry. This handler is a plain def, so the work runs in the threadpool.
    from core.services.ingestion_service import ingest_document

    try:
        ingest_document(text, owner=user.username, title=resolved_title, source_type=source_ext)
    except Exception as exc:
        return render(
            request, "add_resource.html",
            error=f"Could not save this resource: {exc}. Please try another file.",
        )

    return RedirectResponse("/journey", status_code=302)


@router.get("/upload")
async def upload_form(request: Request, user: CurrentUser):
    return render(request, "upload.html")


@router.post("/upload")
def upload(
    request: Request,
    user: CurrentUser,
    file: UploadFile | None = None,
    title: Annotated[str, Form()] = "",
    extracted_text: Annotated[str, Form()] = "",
):
    """Upload a document, save it to the library, then redirect to MakeQuiz
    where the user configures difficulty, question count, and timer before generating."""
    def reject(message: str):
        return render(request, "upload.html", error=message)

    try:
        text, resolved_title, source_ext = extract_upload_text(file, title, extracted_text)
    except ExtractError as exc:
        return reject(str(exc))

    doc_id = None
    try:
        from core.services.ingestion_service import ingest_document
        doc_res = ingest_document(text, owner=user.username, title=resolved_title, source_type=source_ext)
        doc_id = doc_res.get("document_id") if isinstance(doc_res, dict) else None
    except Exception as exc:
        return reject(f"Could not save this resource: {exc}. Please try another file.")

    if doc_id:
        return RedirectResponse(f"/knowledge?doc={doc_id}", status_code=302)
    return RedirectResponse("/knowledge", status_code=302)


def _check_generation_throttle(conn, username: str) -> str | None:
    """Rate-limit MCQ generation; returns an error message or None.

    Protects the LLM from request bursts. Only counts sessions that actually
    triggered generation: weak-topic REVIEW sessions (difficulty='review') are
    rebuilt from stored data with no LLM call, so they must not count. Lenient by
    default so a single user iterating on their own material isn't blocked.
    """
    cooldown = config.GENERATION_COOLDOWN_SECONDS
    max_per_window = config.GENERATION_MAX_PER_5MIN
    if cooldown <= 0:
        return None

    recent = conn.execute(
        "SELECT created_at FROM sessions WHERE teacher=? AND COALESCE(difficulty,'') != 'review' "
        "ORDER BY created_at DESC LIMIT ?",
        (username, max_per_window),
    ).fetchall()
    if not recent:
        return None

    now = datetime.utcnow()
    if (now - datetime.fromisoformat(recent[0]["created_at"])).total_seconds() < cooldown:
        return "Please wait a few seconds between generation requests."

    if len(recent) >= max_per_window:
        oldest = datetime.fromisoformat(recent[-1]["created_at"])
        if (now - oldest).total_seconds() < 300:
            return (
                f"You can only generate {max_per_window} quiz sets every 5 minutes. "
                "Please wait a moment."
            )
    return None
