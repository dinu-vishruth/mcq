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
