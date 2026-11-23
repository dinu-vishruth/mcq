"""Security regression tests.

Covers the pieces that are easy to break silently:
  * demo accounts are not seeded on a hosted deployment unless opt-in
  * uploads are checked by size, magic bytes, PDF page count and encryption
  * generation is throttled per user (cooldown + rolling window)
  * CSRF rejects a tokenless mutating request
  * the session cookie is HttpOnly and SameSite=Lax
  * a deployed boot without SECRET_KEY fails instead of using a random key

Offline + deterministic: temp database, no network, no LLM calls (the throttle
route returns 429 before generation starts).
"""
import io
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="mcq_security_")
_DB = os.path.join(_TMP, "security.db")
os.environ.setdefault("SEED_DEMO_USERS", "true")

import config

config.DB_PATH = _DB
config.EMBEDDING_BACKEND = "hashing"
config.VECTOR_STORE = "sqlite"

import app as app_module  # noqa: E402
from fastapi_app.database import init_db  # noqa: E402
from fastapi_app.uploads import UploadValidationError, validate_upload  # noqa: E402

init_db()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _upload(data: bytes, filename: str):
    from fastapi import UploadFile

    return UploadFile(file=io.BytesIO(data), filename=filename)


def _pdf_bytes(pages: int = 1, encrypt=None) -> bytes:
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, encrypt=encrypt)
    for index in range(pages):
        pdf.drawString(72, 720, f"Page {index + 1}")
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


def _count_users(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    finally:
        conn.close()


class TestDemoSeeding(unittest.TestCase):
    def setUp(self):
        self._db = os.path.join(tempfile.mkdtemp(prefix="mcq_seed_"), "seed.db")
        self._saved = {name: getattr(config, name) for name in
                       ("DB_PATH", "IS_DEPLOYED", "SEED_DEMO_USERS")}
        config.DB_PATH = self._db

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(config, name, value)

    def test_deployed_without_opt_in_seeds_nothing(self):
        config.IS_DEPLOYED = True
        config.SEED_DEMO_USERS = False
        init_db()
        self.assertEqual(_count_users(self._db), 0)

    def test_deployed_with_opt_in_seeds_demo_accounts(self):
        config.IS_DEPLOYED = True
        config.SEED_DEMO_USERS = True
        init_db()
        self.assertEqual(_count_users(self._db), 1)

    def test_local_development_keeps_demo_accounts(self):
        config.IS_DEPLOYED = False
        config.SEED_DEMO_USERS = False
        init_db()
        self.assertEqual(_count_users(self._db), 1)


class TestUploadValidation(unittest.TestCase):
    def test_valid_pdf_passes(self):
        validate_upload(_upload(_pdf_bytes(1), "notes.pdf"), "pdf")

    def test_renamed_file_is_rejected(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload(_upload(b"#!/bin/sh\necho hi\n", "notes.pdf"), "pdf")
        self.assertIn("valid .pdf", str(ctx.exception))

    def test_docx_magic_bytes(self):
        validate_upload(_upload(b"PK\x03\x04" + b"\x00" * 32, "notes.docx"), "docx")
        with self.assertRaises(UploadValidationError):
            validate_upload(_upload(b"not a zip", "notes.docx"), "docx")

    def test_empty_file_is_rejected(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload(_upload(b"", "notes.txt"), "txt")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_oversize_file_is_rejected(self):
        with patch.multiple(config, MAX_UPLOAD_BYTES=64):
            with self.assertRaises(UploadValidationError) as ctx:
                validate_upload(_upload(b"a" * 65, "notes.txt"), "txt")
        self.assertIn("too large", str(ctx.exception).lower())

    def test_binary_text_file_is_rejected(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload(_upload(b"hello\x00world", "notes.txt"), "txt")
        self.assertIn("binary", str(ctx.exception).lower())

    def test_pdf_page_limit(self):
        with patch.multiple(config, MAX_PDF_PAGES=1):
            with self.assertRaises(UploadValidationError) as ctx:
                validate_upload(_upload(_pdf_bytes(2), "notes.pdf"), "pdf")
        self.assertIn("2 pages", str(ctx.exception))

    def test_encrypted_pdf_has_actionable_message(self):
        from reportlab.lib.pdfencrypt import StandardEncryption

        encrypted = _pdf_bytes(1, StandardEncryption("user", "owner", canPrint=0))
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload(_upload(encrypted, "notes.pdf"), "pdf")
        self.assertIn("password-protected", str(ctx.exception))


class TestRateLimits(unittest.TestCase):
    def setUp(self):
        config.DB_PATH = _DB
        self._saved = {name: getattr(config, name) for name in
                       ("GENERATION_COOLDOWN_SECONDS", "GENERATION_MAX_PER_5MIN")}
        self.conn = sqlite3.connect(_DB)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("DELETE FROM sessions")
        self.conn.commit()

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(config, name, value)
        self.conn.close()

    def _session(self, username: str, when: datetime, difficulty="medium"):
        self.conn.execute(
            "INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, mcqs_json, teacher_user_id) "
            "VALUES (?, ?, ?, ?, 300, '[]', NULL)",
            (f"k-{when.timestamp()}-{difficulty}", username, when.isoformat(), difficulty),
        )
        self.conn.commit()

    def test_cooldown_blocks_immediate_regeneration(self):
        from fastapi_app.rate_limits import check_generation_throttle

        config.GENERATION_COOLDOWN_SECONDS = 60
        config.GENERATION_MAX_PER_5MIN = 0
        self._session("alice", datetime.utcnow())
        self.assertIn("wait", check_generation_throttle(self.conn, "alice").lower())

    def test_cooldown_is_per_user(self):
        from fastapi_app.rate_limits import check_generation_throttle

        config.GENERATION_COOLDOWN_SECONDS = 60
        config.GENERATION_MAX_PER_5MIN = 0
        self._session("alice", datetime.utcnow())
        self.assertIsNone(check_generation_throttle(self.conn, "bob"))

    def test_window_limit(self):
        from fastapi_app.rate_limits import check_generation_throttle

        config.GENERATION_COOLDOWN_SECONDS = 0
        config.GENERATION_MAX_PER_5MIN = 2
        now = datetime.utcnow()
        self._session("carla", now - timedelta(seconds=30))
        self._session("carla", now - timedelta(seconds=60))
        self.assertIn("5 minutes", check_generation_throttle(self.conn, "carla"))

    def test_old_sessions_do_not_count(self):
        from fastapi_app.rate_limits import check_generation_throttle

        config.GENERATION_COOLDOWN_SECONDS = 0
        config.GENERATION_MAX_PER_5MIN = 1
        self._session("dave", datetime.utcnow() - timedelta(minutes=10))
        self.assertIsNone(check_generation_throttle(self.conn, "dave"))

    def test_review_sessions_are_exempt(self):
        from fastapi_app.rate_limits import check_generation_throttle

        config.GENERATION_COOLDOWN_SECONDS = 60
        config.GENERATION_MAX_PER_5MIN = 1
        self._session("erin", datetime.utcnow(), difficulty="review")
        self.assertIsNone(check_generation_throttle(self.conn, "erin"))


class _AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests._client import make_client_factory

        cls._client_factory = staticmethod(make_client_factory(app_module.app, config.SECRET_KEY))

    def setUp(self):
        config.DB_PATH = _DB
        config.GENERATION_COOLDOWN_SECONDS = 3
        config.GENERATION_MAX_PER_5MIN = 10
        conn = sqlite3.connect(_DB)
        conn.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()

    def _user(self, username: str) -> int:
        from werkzeug.security import generate_password_hash

        conn = sqlite3.connect(_DB)
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'teacher')",
                (username, generate_password_hash("password123")),
            )
            conn.commit()
            return conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()[0]
        finally:
            conn.close()

    def _login(self, client, username: str):
        user_id = self._user(username)
        response = client.post("/login", data={"username": username, "password": "password123"})
        self.assertEqual(response.status_code, 302)
        return user_id, response

    def _document(self, owner: str) -> int:
        from core.services.ingestion_service import ingest_document

        return ingest_document(
            ("Indexes speed reads while transactions keep writes atomic and durable. ") * 20,
            owner=owner, title="Security doc", source_type="paste",
        )["document_id"]


class TestRouteSecurity(_AppTest):
    def test_generation_returns_429_when_throttled(self):
        client = self._client_factory()
        user_id, _ = self._login(client, "throttled")
        document_id = self._document("throttled")

        conn = sqlite3.connect(_DB)
        conn.execute(
            "INSERT INTO sessions (session_key, teacher, created_at, difficulty, timer, mcqs_json, teacher_user_id) "
            "VALUES ('recent', 'throttled', ?, 'medium', 300, '[]', ?)",
            (datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        conn.close()

        response = client.post("/api/practice/generate",
                               json={"document_id": document_id, "num_questions": 1})
        self.assertEqual(response.status_code, 429)
        self.assertIn("wait", response.json()["error"].lower())
        self.assertTrue(user_id)

    def test_upload_rejects_renamed_file(self):
        client = self._client_factory()
        self._login(client, "uploader")
        response = client.post(
            "/upload",
            files={"file": ("fake.pdf", b"this is not a pdf", "application/pdf")},
            data={"title": "Fake"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("valid .pdf", response.text)

    def test_csrf_token_is_required(self):
        from fastapi.testclient import TestClient

        raw = TestClient(app_module.app)
        response = raw.post("/login", data={"username": "nobody", "password": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("CSRF", response.text)

    def test_session_cookie_flags(self):
        client = self._client_factory()
        _, response = self._login(client, "cookieuser")
        cookie = response.headers.get("set-cookie", "").lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)


class TestDeployedSecretKey(unittest.TestCase):
    def test_deployed_boot_without_secret_key_fails(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("SECRET_KEY", "RENDER", "VERCEL")}
        # SECRET_KEY="" blocks any value from .env (load_dotenv does not override
        # an existing variable, even an empty one).
        env.update({"VERCEL": "1", "SECRET_KEY": ""})
        result = subprocess.run(
            [sys.executable, "-c", "import config"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SECRET_KEY", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
