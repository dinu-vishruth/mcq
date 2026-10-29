"""Adapter that makes FastAPI's ``UploadFile`` look like Flask's ``FileStorage``.

``models/pdf_processor.py`` was written against Werkzeug's FileStorage: it reaches
for ``.stream``, calls ``.save(path)``, and seeks on the object itself. Rather
than fork those extractors (they handle tables, slide notes and group shapes --
real logic worth not duplicating), this wraps the FastAPI upload in the small
surface they actually use.

``UploadFile.file`` is a ``SpooledTemporaryFile``, i.e. a genuine synchronous
binary file object, so every call here is a thin delegation rather than a copy.
"""
from __future__ import annotations

import shutil
from typing import BinaryIO

from fastapi import UploadFile

import config


class UploadValidationError(Exception):
    """Carries a user-safe message when an upload is rejected before parsing."""


#: Leading bytes every real file of the given extension starts with. A .pdf that
#: doesn't begin with %PDF- is a renamed .exe/text file; parsing it would only
#: produce a confusing "corrupted file" error after wasted work.
_MAGIC = {
    "pdf": (b"%PDF-",),
    "docx": (b"PK\x03\x04",),
    "pptx": (b"PK\x03\x04",),
}


class FileStorageAdapter:
    """Minimal FileStorage-compatible view over an UploadFile."""

    def __init__(self, upload: UploadFile) -> None:
        self._upload = upload
        self.filename = upload.filename or ""

    @property
    def stream(self) -> BinaryIO:
        return self._upload.file

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._upload.file.seek(offset, whence)

    def tell(self) -> int:
        return self._upload.file.tell()

    def read(self, size: int = -1) -> bytes:
        return self._upload.file.read(size)

    def save(self, path: str) -> None:
        """Write the upload to disk, as FileStorage.save does."""
        self._upload.file.seek(0)
        with open(path, "wb") as dest:
            shutil.copyfileobj(self._upload.file, dest)
        self._upload.file.seek(0)


def size_of(upload: UploadFile) -> int:
    """Byte length of the upload, restoring the stream position afterwards."""
    upload.file.seek(0, 2)
    size = upload.file.tell()
    upload.file.seek(0)
    return size


def validate_upload(upload: UploadFile, ext: str) -> None:
    """Reject an upload before any parser touches it.

    Checks, cheapest first:
      * non-empty and within ``config.MAX_UPLOAD_BYTES``
      * the leading bytes match the claimed type (catches renamed binaries)
      * text files contain no NUL bytes (catches binary-to-.txt uploads)
      * PDFs are readable, not password-protected, and within
        ``config.MAX_PDF_PAGES`` -- a 10 MB PDF can still expand into thousands
        of pages of parsing work.

    Raises ``UploadValidationError`` with a message that is safe to show.
    """
    limit = config.MAX_UPLOAD_BYTES
    size = size_of(upload)
    if size == 0:
        raise UploadValidationError("The uploaded file is empty.")
    if size > limit:
        limit_mb = limit / (1024 * 1024)
        raise UploadValidationError(
            f"File is too large! Uploads are limited to {limit_mb:.1f} MB. "
            "Please compress or split your file."
        )

    head = upload.file.read(4096)
    upload.file.seek(0)

    expected = _MAGIC.get(ext)
    if expected and not any(head.startswith(marker) for marker in expected):
        raise UploadValidationError(
            f"This doesn't look like a valid .{ext} file. "
            "Please check the file type and try again."
        )
    if ext == "txt" and b"\x00" in head:
        raise UploadValidationError(
            "This looks like a binary file, not text. Please upload a plain .txt file."
        )
    if ext == "pdf":
        _validate_pdf(upload)


def _validate_pdf(upload: UploadFile) -> None:
    """Open the PDF just far enough to count pages / detect encryption."""
    try:
        import pdfplumber
    except ImportError:  # extraction will report the missing dependency
        return

    try:
        with pdfplumber.open(FileStorageAdapter(upload)) as pdf:
            page_count = len(pdf.pages)
    except Exception as exc:
        if _is_password_error(exc):
            raise UploadValidationError(
                "This PDF is password-protected. Remove the password and try again."
            ) from exc
        raise UploadValidationError(
            "Could not read this PDF. Please check that it isn't corrupted."
        ) from exc
    finally:
        upload.file.seek(0)

    if page_count > config.MAX_PDF_PAGES:
        raise UploadValidationError(
            f"This PDF has {page_count} pages; the limit is {config.MAX_PDF_PAGES}. "
            "Please split it into smaller files."
        )


def _is_password_error(exc: Exception) -> bool:
    """True when the failure was "this PDF needs a password".

    pdfplumber wraps pdfminer's ``PDFPasswordIncorrect`` in a ``PdfminerException``
    with an empty message but keeps the original in ``args`` (and as
    ``__context__``), so the type has to be inspected rather than the text.
    """
    candidates = [exc, *exc.args, getattr(exc, "__context__", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        if "password" in candidate.__class__.__name__.lower():
            return True
        if "password" in str(candidate).lower():
            return True
    return False
