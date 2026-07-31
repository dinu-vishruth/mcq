"""
Phase 0 regression net. Locks current behaviour BEFORE the agentic/RAG migration.

Run:  python -m unittest tests.test_regression -v

Design notes:
- Redirects the DB to a throwaway temp file so the real database/mcq.db is never touched.
  config.DB_PATH is patched *before* `import app`, because app.py binds DB_PATH at import
  time and calls init_db() on import.
- Disables CSRF and stubs the two network-calling functions (generate_mcqs, explain_answers)
  so the suite runs fully offline and deterministically.
- Asserts the data contracts the whole app depends on: the MCQ dict shape
  {question, options:[{label,text}*4], answer_text} and the explanations list[str].
"""
import os
import json
import sqlite3
import tempfile
import unittest

# --- Redirect DB + upload dir to temp locations BEFORE importing app ---
_TMP = tempfile.mkdtemp(prefix="mcq_test_")
_DB = os.path.join(_TMP, "mcq_test.db")
_UP = os.path.join(_TMP, "uploads")
os.makedirs(_UP, exist_ok=True)

import config
config.DB_PATH = _DB
config.UPLOAD_FOLDER = _UP

import app as app_module
# app.py bound these names at import via `from config import ...`; repoint them too.
app_module.DB_PATH = _DB
app_module.UPLOAD_FOLDER = _UP
# session_manager also did `from config import DB_PATH`.
import utils.session_manager as sm
sm.DB_PATH = _DB

app_module.init_db()

FIXED_MCQS = [
    {
        "question": "What is 2 + 2?",
        "options": [
            {"label": "A", "text": "3"},
            {"label": "B", "text": "4"},
            {"label": "C", "text": "5"},
            {"label": "D", "text": "6"},
        ],
        "answer_text": "4",
    },
    {
        "question": "Capital of France?",
        "options": [
            {"label": "A", "text": "Berlin"},
            {"label": "B", "text": "Madrid"},
            {"label": "C", "text": "Paris"},
            {"label": "D", "text": "Rome"},
        ],
        "answer_text": "Paris",
    },
]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flask = app_module.app
        cls.flask.config["TESTING"] = True
        cls.flask.config["WTF_CSRF_ENABLED"] = False
        # Stub the network-calling functions at their SOURCE modules. The
        # blueprints import these modules (not the names), so patching the
        # module attribute takes effect in the routes.
        import models.mcq_generator as _mcq
        import models.explanation_engine as _exp
        _mcq.generate_mcqs = lambda text, num_questions=5, difficulty="medium": FIXED_MCQS[:num_questions] if num_questions <= len(FIXED_MCQS) else FIXED_MCQS
        _exp.explain_answers = lambda details, document_id=None: [
            ("Correct" if d["is_correct"] else "Wrong: " + d["correct"]) for d in details
        ]

    def setUp(self):
        # Re-pin config in case another test module (e.g. test_rag) mutated these
        # globals at import time. session_repo reads config.DB_PATH live, so this
        # keeps route reads and repo writes pointed at the same DB.
        config.DB_PATH = _DB
        config.UPLOAD_FOLDER = _UP
        app_module.DB_PATH = _DB
        sm.DB_PATH = _DB
        # The login lockout is keyed by IP; all test clients share 127.0.0.1,
        # so clear attempts between tests to keep them isolated.
        conn = sqlite3.connect(_DB)
        conn.execute("DELETE FROM login_attempts")
        conn.commit()
        conn.close()

    def client(self):
        return self.flask.test_client()

    def _seed_user(self, username, password, role):
        from werkzeug.security import generate_password_hash
        conn = sqlite3.connect(_DB)
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), role),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()

    def _login(self, c, username, password):
        return c.post("/login", data={"username": username, "password": password}, follow_redirects=False)


class TestPublicRoutes(Base):
    def test_home_shows_login_when_logged_out(self):
        r = self.client().get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"MCQ Generator", r.data)

    def test_signup_get(self):
        self.assertEqual(self.client().get("/signup").status_code, 200)

    def test_signup_rejects_short_password(self):
        r = self.client().post("/signup", data={"username": "x1", "password": "short", "role": "teacher"})
        self.assertIn(b"at least 8 characters", r.data)

    def test_signup_then_login(self):
        # Single unified "User" role: signup no longer takes a role and login
        # always lands on the unified dashboard.
        c = self.client()
        c.post("/signup", data={"username": "newuser", "password": "password123"})
        r = self._login(c, "newuser", "password123")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/dashboard", r.headers["Location"])

    def test_login_bad_credentials(self):
        r = self._login(self.client(), "nobody", "wrongpass")
        self.assertIn(b"Invalid username or password", r.data)

    def test_login_lockout_after_5_failures(self):
        c = self.client()
        self._seed_user("lockme", "password123", "student")
        for _ in range(5):
            self._login(c, "lockme", "wrongpass")
        r = self._login(c, "lockme", "wrongpass")
        self.assertIn(b"Too many failed login attempts", r.data)


class TestAuthGating(Base):
    """Single unified "User" role. Pages require a logged-in user (user_id),
    not a specific role. The legacy /teacher path redirects to the dashboard."""

    def test_dashboard_requires_login(self):
        r = self.client().get("/dashboard")
        self.assertEqual(r.status_code, 302)  # redirected to home

    def test_student_alias_requires_login(self):
        r = self.client().get("/student")
        self.assertEqual(r.status_code, 302)

    def test_legacy_teacher_path_redirects_to_dashboard(self):
        c = self.client()
        self._seed_user("teach2", "password123", "student")
        self._login(c, "teach2", "password123")
        r = c.get("/teacher")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/dashboard", r.headers["Location"])

    def test_user_reaches_dashboard(self):
        c = self.client()
        self._seed_user("stud2", "password123", "student")
        self._login(c, "stud2", "password123")
        r = c.get("/dashboard")
        self.assertEqual(r.status_code, 200)
        # Dashboard is a React island; assert the mount shell is served.
        self.assertIn(b'data-page="dashboard"', r.data)

    def test_new_pages_require_login(self):
        for path in ("/knowledge", "/journey", "/practice", "/weak-topics", "/achievements"):
            r = self.client().get(path)
            self.assertEqual(r.status_code, 302, f"{path} should redirect when logged out")


class TestGenerationAndQuizFlow(Base):
    """The core contract: generate -> session_key -> take -> submit -> score."""

    def _make_session(self):
        from utils.session_manager import create_session_key
        return create_session_key(teacher="teach3", difficulty="medium", timer=120, mcqs=FIXED_MCQS)

    def test_upload_generates_session(self):
        c = self.client()
        self._seed_user("uploader", "password123", "teacher")
        self._login(c, "uploader", "password123")
        # `timer` is submitted in MINUTES and stored as seconds.
        r = c.post("/upload", data={"extracted_text": "Some study material about math and geography.",
                                    "num_questions": "2", "difficulty": "medium", "timer": "10"})
        # There is no answer-revealing preview any more: upload hands straight
        # off to the quiz so the questions are never shown with their answers.
        self.assertEqual(r.status_code, 302)
        self.assertIn("/mcq_test", r.headers["Location"])
        with c.session_transaction() as sess:
            self.assertEqual(sess["timer"], 600)
            self.assertTrue(sess["session_key"])

    def test_upload_rejects_out_of_range_timer(self):
        c = self.client()
        self._seed_user("uploader2", "password123", "teacher")
        self._login(c, "uploader2", "password123")
        r = c.post("/upload", data={"extracted_text": "Some study material.",
                                    "num_questions": "2", "difficulty": "medium", "timer": "999"})
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"between 1 and 180 minutes", r.data)

    def test_full_student_quiz_scoring(self):
        c = self.client()
        self._seed_user("stud3", "password123", "student")
        self._login(c, "stud3", "password123")
        key = self._make_session()

        # Join
        r = c.post("/student_login", data={"session_key": key}, follow_redirects=False)
        self.assertEqual(r.status_code, 302)
        self.assertIn("/mcq_test", r.headers["Location"])

        # Render test (this populates mcqs_randomized in session)
        r = c.get("/mcq_test")
        self.assertEqual(r.status_code, 200)

        # Read randomized answers from the server session to submit correct ones.
        with c.session_transaction() as sess:
            randomized = sess["mcqs_randomized"]
        form = {"student_name": "stud3", "time_spent": "42"}
        for i, q in enumerate(randomized):
            form[f"q-{i}"] = q["answer_text"]  # all correct

        r = c.post("/submit", data=form)
        self.assertEqual(r.status_code, 200)
        # Result is now a React island; score is carried in the bootstrap JSON
        # payload rather than server-rendered HTML.
        self.assertIn(b'data-page="result"', r.data)
        self.assertIn(b'"score": 2', r.data)
        self.assertIn(b'"total": 2', r.data)

        # Result persisted with correct score
        conn = sqlite3.connect(_DB)
        row = conn.execute("SELECT score, total, time_spent FROM results WHERE session_key=?", (key,)).fetchone()
        conn.close()
        self.assertEqual(row[0], 2)
        self.assertEqual(row[1], 2)
        self.assertEqual(row[2], 42)

    def test_duplicate_attempt_blocked(self):
        c = self.client()
        self._seed_user("stud4", "password123", "student")
        self._login(c, "stud4", "password123")
        key = self._make_session()
        c.post("/student_login", data={"session_key": key})
        c.get("/mcq_test")
        with c.session_transaction() as sess:
            randomized = sess["mcqs_randomized"]
        form = {f"q-{i}": q["answer_text"] for i, q in enumerate(randomized)}
        c.post("/submit", data=form)
        # Second join attempt must be blocked
        r = c.post("/student_login", data={"session_key": key})
        self.assertIn(b"already taken", r.data)


class TestTeacherSessionManagement(Base):
    def _login_teacher_with_session(self, c, name="teach5"):
        from utils.session_manager import create_session_key
        self._seed_user(name, "password123", "teacher")
        self._login(c, name, "password123")
        return create_session_key(teacher=name, difficulty="hard", timer=90, mcqs=FIXED_MCQS)

    def test_download_report_pdf(self):
        c = self.client()
        key = self._login_teacher_with_session(c)
        r = c.get(f"/download_report/{key}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["Content-Type"], "application/pdf")

    def test_session_report_page(self):
        c = self.client()
        key = self._login_teacher_with_session(c, "teach6")
        r = c.get(f"/session_report/{key}")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Session Report", r.data)

    def test_export_results_csv(self):
        c = self.client()
        key = self._login_teacher_with_session(c, "teach7")
        r = c.get(f"/export_results/{key}")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/csv", r.headers["Content-Type"])

    def test_archive_unarchive_clone_delete(self):
        c = self.client()
        key = self._login_teacher_with_session(c, "teach8")
        self.assertEqual(c.post(f"/archive_session/{key}", follow_redirects=False).status_code, 302)
        self.assertEqual(c.post(f"/unarchive_session/{key}", follow_redirects=False).status_code, 302)
        self.assertEqual(c.post(f"/clone_session/{key}", follow_redirects=False).status_code, 302)
        self.assertEqual(c.post(f"/delete_session/{key}", follow_redirects=False).status_code, 302)

    def test_access_denied_for_other_teachers_session(self):
        c = self.client()
        key = self._login_teacher_with_session(c, "teach9")
        # Log in as a different teacher
        c.get("/logout")
        self._seed_user("teach10", "password123", "teacher")
        self._login(c, "teach10", "password123")
        r = c.get(f"/session_report/{key}")
        self.assertEqual(r.status_code, 403)


class TestDataContracts(Base):
    """These shapes are consumed by templates and /submit; they must never drift."""

    def test_mcq_dict_shape(self):
        for q in FIXED_MCQS:
            self.assertIn("question", q)
            self.assertIn("options", q)
            self.assertIn("answer_text", q)
            self.assertEqual(len(q["options"]), 4)
            texts = []
            for o in q["options"]:
                self.assertIn("label", o)
                self.assertIn("text", o)
                texts.append(o["text"])
            self.assertIn(q["answer_text"], texts)  # answer must match an option

    def test_explanations_contract(self):
        details = [
            {"question": "Q", "selected": "4", "correct": "4", "is_correct": True},
            {"question": "Q2", "selected": "Berlin", "correct": "Paris", "is_correct": False},
        ]
        import models.explanation_engine as _exp
        result = _exp.explain_answers(details)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(details))
        self.assertTrue(all(isinstance(x, str) for x in result))


class TestTruncatedJSONSalvage(unittest.TestCase):
    """A large batch can exceed the provider's output cap and be cut mid-array.
    The complete objects must survive so callers can top up the shortfall."""

    def _parse(self, raw):
        from core.llm.base import LLMProvider
        return LLMProvider._parse_json(raw)

    @staticmethod
    def _q(n):
        return (
            '{"question": "Q%d?", "options": ['
            '{"label": "A", "text": "a%d"}, {"label": "B", "text": "b%d"}, '
            '{"label": "C", "text": "c%d"}, {"label": "D", "text": "d%d"}], '
            '"answer_text": "a%d"}' % (n, n, n, n, n, n)
        )

    def test_complete_json_still_parses(self):
        raw = '{"questions": [%s, %s]}' % (self._q(1), self._q(2))
        self.assertEqual(len(self._parse(raw)["questions"]), 2)

    def test_truncated_array_salvages_complete_objects(self):
        # Two whole questions, then the response is cut off mid-third.
        raw = ('{"questions": [' + self._q(1) + ', ' + self._q(2)
               + ', {"question": "Q3?", "options": [{"label": "A", "tex')
        got = self._parse(raw)["questions"]
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["question"], "Q1?")
        self.assertEqual(got[1]["answer_text"], "a2")

    def test_braces_inside_strings_do_not_break_scanning(self):
        raw = ('{"questions": [{"question": "Is {a: 1} a dict?", "options": ['
               '{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}, '
               '{"label": "C", "text": "maybe"}, {"label": "D", "text": "n/a"}], '
               '"answer_text": "yes"}, {"question": "cut off her')
        got = self._parse(raw)["questions"]
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["question"], "Is {a: 1} a dict?")

    def test_unsalvageable_response_still_raises(self):
        from core.llm.base import LLMError
        with self.assertRaises(LLMError):
            self._parse("I'm sorry, I cannot help with that request.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
