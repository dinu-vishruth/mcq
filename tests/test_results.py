import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import config
from core.models.db import get_db
from tests._client import make_client_factory
from tests.test_regression import FIXED_MCQS, Base, app_module
from utils.session_manager import create_session_key


class TestSavedResults(Base):
    @classmethod
    def setUpClass(cls):
        cls._client_factory = staticmethod(make_client_factory(app_module.app, config.SECRET_KEY))

    def setUp(self):
        super().setUp()
        self.c = self.client()
        self._seed_user("results_learner", "password123", "student")
        self._login(self.c, "results_learner", "password123")
        self.key = create_session_key("results_learner", "easy", 600, FIXED_MCQS)
        self.c.post("/student_login", data={"session_key": self.key})
        analyse = patch("core.services.learning_service.analyse_submission")
        self.analyse = analyse.start()
        self.addCleanup(analyse.stop)

    def payload(self, response):
        return json.loads(html.unescape(re.search(rb"data-bootstrap='([^']+)'", response.data).group(1).decode()))

    def test_duplicate_submission_returns_original_score_and_explanations(self):
        with patch("models.explanation_engine.explain_answers", return_value=["Explanation one", "Explanation two"]) as explain:
            first = self.c.post("/submit", data={"q-0": "4", "q-1": ""})
            second = self.c.post("/submit", data={"q-0": "wrong", "q-1": "Paris"})
            self.assertEqual(first.status_code, 303)
            self.assertEqual(first.headers["Location"], second.headers["Location"])
            result = self.payload(self.c.get(first.headers["Location"]))
            self.assertEqual((result["score"], result["total"]), (1, 2))
            self.assertEqual(result["explanations"], ["Explanation one", "Explanation two"])
            self.assertEqual(self.payload(self.c.get(first.headers["Location"])), result)
            explain.assert_called_once()
            self.analyse.assert_called_once()
        self.assertEqual(self.c.get("/mcq_test").headers["Location"], f"/results/{self.key}")

    def test_concurrent_submissions_create_one_result(self):
        other = self.client()
        self._login(other, "results_learner", "password123")
        other.post("/student_login", data={"session_key": self.key})
        def explain(details):
            time.sleep(0.1)  # Keep the first submission active as the other arrives.
            return ["Saved explanation"] * len(details)
        with patch("models.explanation_engine.explain_answers", side_effect=explain) as explanation:
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda c: c.post("/submit", data={"q-0": "4"}), [self.c, other]))
            self.assertTrue(all(r.status_code == 303 for r in responses))
            explanation.assert_called_once()
        db = get_db()
        self.assertEqual(db.execute("SELECT COUNT(*) FROM results WHERE session_key=?", (self.key,)).fetchone()[0], 1)
        db.close()

    def test_explanation_failure_still_saves_score(self):
        with patch("models.explanation_engine.explain_answers", side_effect=RuntimeError("AI unavailable")):
            response = self.c.post("/submit", data={"q-0": "4"})
        result = self.payload(self.c.get(response.headers["Location"]))
        self.assertEqual(result["score"], 1)
        self.assertTrue(all(result["explanations"]))

    def test_results_are_private_to_the_learner(self):
        with patch("models.explanation_engine.explain_answers", return_value=["Explanation"] * 2):
            self.c.post("/submit", data={})
        other = self.client()
        self._seed_user("another_results_learner", "password123", "student")
        self._login(other, "another_results_learner", "password123")
        self.assertEqual(other.get(f"/results/{self.key}").status_code, 404)

    def test_old_saved_result_can_be_recovered(self):
        db = get_db()
        db.execute("INSERT INTO results(session_key,user_id,score,total,detail_json) "
                   "VALUES (?, (SELECT id FROM users WHERE username=?), 0, 1, ?)",
                   (self.key, "results_learner", json.dumps([{
                       "question": "Example", "selected": "", "correct": "Answer", "is_correct": False,
                   }])))
        db.commit()
        db.close()
        result = self.payload(self.c.get(f"/results/{self.key}"))
        self.assertEqual(result["score"], 0)
        self.assertIn("Not answered", result["explanations"][0])
