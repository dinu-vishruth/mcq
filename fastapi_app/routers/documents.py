"""Validated uploads with separate Library and temporary quiz-input paths.

Both routes share extraction and validation. /ingest_resource explicitly saves
and indexes a Library resource; /upload prepares temporary text for quiz settings
without saving an original file or indexing a Library document.
"""
from __future__ import annotations

import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import RedirectResponse

from config import ALLOWED_EXT, UPLOAD_FOLDER
from models.pdf_processor import (
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_pptx,
)
from utils.text_cleaner import clean_text

from ..deps import CurrentUser
from ..templating import render
from ..uploads import FileStorageAdapter, UploadValidationError, size_of, validate_upload

router = APIRouter(tags=["documents"])


def persist_original_file(user: str, upload: UploadFile, source_ext: str) -> None:
    """Keep the original upload in the object store (best effort, never fatal).

    Text extraction works straight out of the request stream, so without this
    step the file bytes are gone after the request ends -- on serverless hosts
    the filesystem is wiped right after. Saving via core.storage keeps the bytes
    reachable from later code (re-extraction, debugging) and becomes durable in
    production when S3_UPLOAD_BUCKET is set.

    Deliberately non-fatal: extraction already succeeded, so a storage hiccup
    must not turn a completed upload into an error for the user. Local dev keeps
    a plain disk copy under UPLOAD_FOLDER.
    """
    if upload is None or not upload.filename:
        return
    try:
        from core.storage import get_upload_store

        key = f"{user}/{uuid.uuid4().hex}-{os.path.basename(upload.filename)}"
        upload.file.seek(0)
        get_upload_store().save(key, upload.file, size_hint=size_of(upload))
    except Exception as exc:
        print(f"[WARNING] Could not persist original upload: {type(exc).__name__}")

#: App-enforced upload ceiling lives in config.MAX_UPLOAD_BYTES so the limit and
#: the error message can't drift apart. Enforced by uploads.validate_upload.
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

        ext = upload.filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXT:
            raise ExtractError("Unsupported file type.")
        # Size, magic bytes, PDF page count / encryption -- all before parsing.
        try:
            validate_upload(upload, ext)
        except UploadValidationError as exc:
            raise ExtractError(str(exc)) from exc
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
        except Exception:
            raise ExtractError(
                "Failed to extract text from the file. "
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
    persist_original_file(user.username, file, source_ext)
    from core.services.ingestion_service import ingest_document

    try:
        ingest_document(text, owner=user.username, title=resolved_title, source_type=source_ext)
    except Exception:
        return render(
            request, "add_resource.html",
            error="Could not save this resource. Please try another file.",
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
    """Prepare temporary quiz input without adding it to the Library."""
    def reject(message: str):
        return render(request, "upload.html", error=message)

    try:
        text, resolved_title, source_ext = extract_upload_text(file, title, extracted_text)
    except ExtractError as exc:
        return reject(str(exc))

    try:
        from core.repositories import quiz_upload_repo
        upload_id = quiz_upload_repo.create(user.id, resolved_title, text)
    except Exception:
        return reject("Could not prepare this quiz material. Please try again.")

    return RedirectResponse(f"/knowledge?upload={upload_id}", status_code=302)


def _check_generation_throttle(conn, username: str) -> str | None:
    """Deprecated alias kept for the legacy core/routes module.

    The FastAPI app uses fastapi_app.rate_limits.check_generation_throttle, which
    is actually wired into /api/practice/generate; this shim keeps the old
    import path working without duplicating the window logic.
    """
    from ..rate_limits import check_generation_throttle

    return check_generation_throttle(conn, username)
