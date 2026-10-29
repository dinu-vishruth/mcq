"""Uploaded-file storage behind one tiny interface.

Uploaded documents used to be consumed straight out of the request stream and
then vanish with the /tmp filesystem on serverless hosts. This module lets any
caller persist the original file without knowing where it went:

    store = get_upload_store()
    store.save(key, fileobj)   # returns the storage key
    store.open(key)            # readable binary file object
    store.exists(key) / store.delete(key)

Backends: ``LocalUploadStore`` (UPLOAD_FOLDER on disk, the dev/single-host
default) and ``S3UploadStore`` (any S3-compatible service; needs the optional
boto3 dependency and S3_UPLOAD_BUCKET). The registry picks "s3" automatically
when a bucket is configured, so dev stays on disk with zero config.
"""
from __future__ import annotations

import os
from typing import BinaryIO, Protocol

import config


class UploadStore(Protocol):
    def save(self, key: str, fileobj: BinaryIO, size_hint: int | None = None) -> str: ...
    def open(self, key: str): ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


class LocalUploadStore:
    """Store files under config.UPLOAD_FOLDER, keys as path components."""

    def __init__(self, folder: str | None = None) -> None:
        self.folder = folder or config.UPLOAD_FOLDER

    def _path(self, key: str) -> str:
        if os.path.isabs(key):
            return key
        return os.path.join(self.folder, key)

    def save(self, key: str, fileobj: BinaryIO, size_hint: int | None = None) -> str:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fileobj.seek(0)
        with open(path, "wb") as dest:
            dest.write(fileobj.read())
        fileobj.seek(0)
        return key

    def open(self, key: str):
        return open(self._path(key), "rb")

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def delete(self, key: str) -> None:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)


class S3UploadStore:
    """Boto3-based store for any S3-compatible endpoint."""

    _client = None  # cached boto3 client; built lazily so import stays optional

    def __init__(self, bucket: str | None = None, prefix: str | None = None) -> None:
        self.bucket = bucket or config.S3_UPLOAD_BUCKET
        self.prefix = (prefix or config.S3_UPLOAD_PREFIX).strip("/")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _get_client(self):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "UPLOAD_BACKEND=s3 requires the optional boto3 dependency: "
                "pip install boto3"
            ) from exc
        if S3UploadStore._client is None:
            kwargs = {}
            if config.S3_UPLOAD_ENDPOINT:
                kwargs["endpoint_url"] = config.S3_UPLOAD_ENDPOINT
            S3UploadStore._client = boto3.client("s3", **kwargs)
        return S3UploadStore._client

    def save(self, key: str, fileobj: BinaryIO, size_hint: int | None = None) -> str:
        extra = {"ContentLength": size_hint} if size_hint else None
        fileobj.seek(0)
        self._get_client().upload_fileobj(fileobj, self.bucket, self._full_key(key),
                                          ExtraArgs=extra)
        fileobj.seek(0)
        return key

    def open(self, key: str):
        from io import BytesIO

        buf = BytesIO()
        self._get_client().download_fileobj(self.bucket, self._full_key(key), buf)
        buf.seek(0)
        return buf

    def exists(self, key: str) -> bool:
        try:
            self._get_client().head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self._get_client().delete_object(Bucket=self.bucket, Key=self._full_key(key))


_cached: UploadStore | None = None


def build_upload_store() -> UploadStore:
    backend = config.UPLOAD_BACKEND
    if backend == "auto":
        backend = "s3" if config.S3_UPLOAD_BUCKET else "local"
    if backend == "s3":
        return S3UploadStore()
    if backend != "local":
        raise RuntimeError(f"Unknown UPLOAD_BACKEND {backend!r}; use local, s3 or auto.")
    return LocalUploadStore()


def get_upload_store() -> UploadStore:
    global _cached
    if _cached is None:
        _cached = build_upload_store()
    return _cached


def reset_upload_store() -> None:
    global _cached
    _cached = None
