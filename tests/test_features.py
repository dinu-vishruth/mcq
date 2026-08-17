"""
Feature tests for the knowledge/weak-topic fixes:
  - /upload ingests + indexes a document so it becomes a Knowledge source
  - /api/knowledge lists it
  - weak-topic review reconstructs the ACTUAL missed questions and launches a
    playable session

Offline + deterministic: hashing embedder, sqlite vector store, stubbed MCQ
generation. Redirects DB/upload dirs to temp locations BEFORE importing app.

Run:  python -m unittest tests.test_features -v
"""
import os
import json
import sqlite3
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="mcq_feat_")
_DB = os.path.join(_TMP, "feat.db")
_UP = os.path.join(_TMP, "uploads")
os.makedirs(_UP, exist_ok=True)

import config
config.DB_PATH = _DB
config.UPLOAD_FOLDER = _UP
config.EMBEDDING_BACKEND = "hashing"
config.VECTOR_STORE = "sqlite"

import app as app_module
import utils.session_manager as sm
sm.DB_PATH = _DB

from fastapi_app.database import init_db
init_db()

# Deterministic, offline MCQ generation.
FIXED_MCQS = [
    {"question": "What does WHERE filter?",
     "options": [{"label": "A", "text": "rows before grouping"}, {"label": "B", "text": "grouped results"},
                 {"label": "C", "text": "columns"}, {"label": "D", "text": "indexes"}],
     "answer_text": "rows before grouping"},
    {"question": "What does HAVING filter?",
     "options": [{"label": "A", "text": "rows before grouping"}, {"label": "B", "text": "aggregated groups"},
                 {"label": "C", "text": "columns"}, {"label": "D", "text": "tables"}],
     "answer_text": "aggregated groups"},
]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # FastAPI app behind a Flask-test-client-shaped shim (see tests/_client.py).
        # CSRF is exercised, not disabled: the shim supplies a real token.
        from config import SECRET_KEY
        from tests._client import make_client_factory
        cls._client_factory = staticmethod(make_client_factory(app_module.app, SECRET_KEY))
        import models.mcq_generator as _mcq
        _mcq.generate_mcqs = lambda text, num_questions=5, difficulty="medium": FIXED_MCQS[:num_questions]

    def setUp(self):
        config.DB_PATH = _DB
        sm.DB_PATH = _DB
        conn = sqlite3.connect(_DB)
        conn.execute("DELETE FROM login_attempts")
        conn.commit()
        conn.close()

    def _seed_and_login(self, c, username):
        from werkzeug.security import generate_password_hash
        conn = sqlite3.connect(_DB)
        try:
            conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'student')",
                         (username, generate_password_hash("password123")))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()
        c.post("/login", data={"username": username, "password": "password123"})


class TestInstantUploadDoesNotSaveResource(Base):
    """Instant-quiz upload generates a quiz but must NOT store a Learning Journey
    resource. Only /ingest_resource (explicit 'Add Resource') saves to the library."""

    def test_upload_generates_quiz_without_saving_resource(self):
        c = self._client_factory()
        self._seed_and_login(c, "ku1")
        r = c.post("/upload", data={
            "extracted_text": "Deadlocks occur when two transactions each wait for the other. " * 20,
            "title": "One-off Quiz Material", "num_questions": "2", "difficulty": "medium", "timer": "10",
        })
        self.assertEqual(r.status_code, 302)
        self.assertIn("/mcq_test", r.headers["Location"])

        # The one-off material must NOT appear as a saved resource.
        items = c.get("/api/knowledge").get_json()["items"]
        self.assertFalse(any(it["title"] == "One-off Quiz Material" for it in items),
                         f"instant-quiz upload should not be saved: {items}")


class TestStoreOnlyResourceUpload(Base):
    """Learning Journey's Add Resource flow: /ingest_resource stores + indexes a
    document WITHOUT generating a quiz, and redirects to the library."""

    def _count_sessions(self):
        conn = sqlite3.connect(_DB)
        try:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            conn.close()

    def test_ingest_resource_stores_without_quiz(self):
        c = self._client_factory()
        self._seed_and_login(c, "rs1")
        before = self._count_sessions()

        r = c.post("/ingest_resource", data={
            "extracted_text": "Normalization reduces redundancy across relational tables. " * 20,
            "title": "DBMS Normalization",
        })
        # Redirects to the library, NOT to a quiz.
        self.assertEqual(r.status_code, 302)
        self.assertIn("/journey", r.headers["Location"])
        self.assertNotIn("/mcq_test", r.headers["Location"])

        # No quiz session was created by storing a resource.
        self.assertEqual(self._count_sessions(), before)

        # But it now appears as a saved resource.
        items = c.get("/api/knowledge").get_json()["items"]
        self.assertTrue(any(it["title"] == "DBMS Normalization" for it in items))

    def test_ingest_resource_requires_login(self):
        r = self._client_factory().post("/ingest_resource", data={"extracted_text": "x"})
        self.assertEqual(r.status_code, 302)
        self.assertNotIn("/journey", r.headers["Location"])  # bounced to home

    def test_same_content_saved_separately_for_each_owner(self):
        """Dedup is per owner, so the 2nd uploader still gets their own resource.

        Regression: doc_hash was globally UNIQUE and find_by_hash ignored owner,
        so the second user to add identical material got the first user's row
        back, had no row created, and landed on an empty library.
        """
        shared = "Indexes speed up lookups by avoiding a full table scan. " * 20

        first = self._client_factory()
        self._seed_and_login(first, "dup_a")
        r1 = first.post("/ingest_resource", data={"extracted_text": shared, "title": "A Copy"})
        self.assertEqual(r1.status_code, 302)
        self.assertIn("/journey", r1.headers["Location"])

        second = self._client_factory()
        self._seed_and_login(second, "dup_b")
        r2 = second.post("/ingest_resource", data={"extracted_text": shared, "title": "B Copy"})
        self.assertEqual(r2.status_code, 302)
        self.assertIn("/journey", r2.headers["Location"])

        # The heart of the bug: this list came back empty.
        b_items = second.get("/api/knowledge").get_json()["items"]
        self.assertTrue(any(it["title"] == "B Copy" for it in b_items),
                        f"second owner's library should not be empty: {b_items}")
        self.assertTrue(all(it["indexed"] for it in b_items),
                        f"second owner's copy should be indexed and quiz-ready: {b_items}")

        # Each library shows only its own copy.
        a_titles = {it["title"] for it in first.get("/api/knowledge").get_json()["items"]}
        b_titles = {it["title"] for it in b_items}
        self.assertIn("A Copy", a_titles)
        self.assertNotIn("B Copy", a_titles)
        self.assertNotIn("A Copy", b_titles)

    def test_re_adding_own_content_does_not_duplicate(self):
        """Per-owner dedup still applies: the same user re-adding gets one row."""
        text = "Transactions must satisfy atomicity, consistency, isolation, durability. " * 20
        c = self._client_factory()
        self._seed_and_login(c, "dup_c")

        c.post("/ingest_resource", data={"extracted_text": text, "title": "ACID"})
        c.post("/ingest_resource", data={"extracted_text": text, "title": "ACID again"})

        items = c.get("/api/knowledge").get_json()["items"]
        self.assertEqual(len(items), 1, f"re-adding own content should reuse the row: {items}")


class TestWeakTopicReview(Base):
    def _uid(self, username):
        conn = sqlite3.connect(_DB)
        try:
            row = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def test_review_reconstructs_missed_questions(self):
        c = self._client_factory()
        self._seed_and_login(c, "wt1")
        uid = self._uid("wt1")

        from utils.session_manager import create_session_key
        key = create_session_key("wt1", "medium", 120, FIXED_MCQS)

        from core.repositories import learning_repo
        learning_repo.record_events(uid, key, "medium", [
            {"topic": "SQL Clauses", "question": "What does WHERE filter?", "is_correct": True},
            {"topic": "SQL Clauses", "question": "What does HAVING filter?", "is_correct": False},
        ])
        learning_repo.upsert_weak_topics(uid, {"SQL Clauses": {"wrong": 1, "total": 2}})

        # The weak-topics API reports how many questions are reviewable.
        wr = c.get("/api/weak-topics").get_json()["items"]
        sql = next(w for w in wr if w["topic"] == "SQL Clauses")
        self.assertEqual(sql["reviewable"], 1)

        # Starting a review returns a real session_key with the missed question.
        rev = c.post("/api/weak-topics/review", json={"topic": "SQL Clauses"})
        self.assertEqual(rev.status_code, 200)
        body = rev.get_json()
        self.assertEqual(body["count"], 1)

        conn = sqlite3.connect(_DB)
        mj = conn.execute("SELECT mcqs_json FROM sessions WHERE session_key=?", (body["session_key"],)).fetchone()[0]
        conn.close()
        rebuilt = json.loads(mj)
        self.assertEqual(len(rebuilt), 1)
        self.assertEqual(rebuilt[0]["question"], "What does HAVING filter?")
        self.assertEqual(len(rebuilt[0]["options"]), 4)

    def test_review_404_when_nothing_reconstructable(self):
        c = self._client_factory()
        self._seed_and_login(c, "wt2")
        r = c.post("/api/weak-topics/review", json={"topic": "Nonexistent Topic"})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
