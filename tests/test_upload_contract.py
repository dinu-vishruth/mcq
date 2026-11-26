"""Keep already-open upload forms working across frontend releases."""
import re
from unittest.mock import patch

from core.repositories import session_repo
from tests.test_regression import FIXED_MCQS, Base

_GENERATION = "core.services.mcq_pipeline.generate_from_text"


class TestUploadContract(Base):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.multiple("config", GENERATION_COOLDOWN_SECONDS=0, GENERATION_MAX_PER_5MIN=0))
        self.enterContext(patch("core.services.generation_status.generation_available", return_value=True))
        self.c = self.client()
        self._seed_user("upload_contract", "Cedar river orbit 42!", "student")
        self._login(self.c, "upload_contract", "Cedar river orbit 42!")

    def test_generate_and_start_form_generates_instead_of_reopening_settings(self):
        with patch(_GENERATION, return_value=FIXED_MCQS) as generate:
            response = self.c.post("/upload", data={
                "extracted_text": "Photosynthesis converts light into chemical energy.",
                "num_questions": "2", "difficulty": "hard", "timer": "20",
            })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/mcq_test")
        generate.assert_called_once_with(
            "Photosynthesis converts light into chemical energy.", 2, "hard", topic=None,
        )
        with self.c.session_transaction() as session:
            self.assertEqual(session_repo.get(session["session_key"])["timer"], 1200)
        quiz = self.c.get("/mcq_test")
        self.assertEqual(quiz.status_code, 200)
        self.assertIn(b'data-page="quiz"', quiz.data)

    def test_upload_and_continue_still_opens_settings(self):
        with patch(_GENERATION) as generate:
            response = self.c.post("/upload", data={"extracted_text": "Study material.", "title": "New quiz source"})
        match = re.fullmatch(r"/knowledge\?upload=([a-f0-9]{32})", response.headers["Location"])
        self.assertIsNotNone(match)
        source = self.c.get(f"/api/quiz-uploads/{match.group(1)}")
        self.assertEqual(source.get_json()["title"], "New quiz source")
        items = self.c.get("/api/knowledge").get_json()["items"]
        self.assertFalse(any(item["title"] == "New quiz source" for item in items))
        generate.assert_not_called()

    def test_old_form_rejects_invalid_settings_without_generation(self):
        with patch(_GENERATION) as generate:
            response = self.c.post("/upload", data={
                "extracted_text": "Study material.", "num_questions": "100", "timer": "10",
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'between 1 and 30', response.data)
        generate.assert_not_called()

    def test_generation_failure_is_visible_instead_of_a_redirect(self):
        with patch(_GENERATION, side_effect=RuntimeError("Private provider error")):
            response = self.c.post("/upload", data={
                "extracted_text": "Study material.", "num_questions": "2", "timer": "10",
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Could not complete verified quiz generation', response.data)
        self.assertNotIn(b'Private provider error', response.data)

    def test_upload_page_uses_existing_versioned_assets(self):
        response = self.c.get("/upload")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        for extension in ("js", "css"):
            match = re.search(rb'/static/dist/app-[\w-]+\.' + extension.encode(), response.data)
            self.assertIsNotNone(match)
            asset = self.c.get(match.group().decode())
            self.assertEqual(asset.status_code, 200)
