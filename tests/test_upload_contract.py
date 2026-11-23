"""Keep already-open upload forms working across frontend releases."""
from unittest.mock import patch

from tests.test_regression import Base, FIXED_MCQS
import models.mcq_generator as mcq_module
import models.explanation_engine as explanation_module


class TestUploadContract(Base):
    @classmethod
    def setUpClass(cls):
        cls._original_generate = mcq_module.generate_mcqs
        cls._original_explain = explanation_module.explain_answers
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        mcq_module.generate_mcqs = cls._original_generate
        explanation_module.explain_answers = cls._original_explain

    def setUp(self):
        super().setUp()
        throttle = patch("config.GENERATION_COOLDOWN_SECONDS", 0)
        throttle.start()
        self.addCleanup(throttle.stop)
        self.c = self.client()
        self._seed_user("upload_contract", "password123", "student")
        self._login(self.c, "upload_contract", "password123")

    def test_generate_and_start_form_generates_instead_of_reopening_settings(self):
        with patch("core.services.ingestion_service.ingest_document", return_value={"document_id": 42}), patch(
            "models.mcq_generator.generate_mcqs", return_value=FIXED_MCQS
        ) as generate:
            response = self.c.post("/upload", data={
                "extracted_text": "Photosynthesis converts light into chemical energy.",
                "num_questions": "2", "difficulty": "hard", "timer": "20",
            })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/mcq_test")
        generate.assert_called_once_with(
            "Photosynthesis converts light into chemical energy.", num_questions=2, difficulty="hard",
        )
        quiz = self.c.get("/mcq_test")
        self.assertEqual(quiz.status_code, 200)
        self.assertIn(b'data-page="quiz"', quiz.data)
        self.assertIn(b'1200', quiz.data)

    def test_upload_and_continue_still_opens_settings(self):
        with patch("core.services.ingestion_service.ingest_document", return_value={"document_id": 42}), patch(
            "models.mcq_generator.generate_mcqs"
        ) as generate:
            response = self.c.post("/upload", data={"extracted_text": "Study material."})
        self.assertEqual(response.headers["Location"], "/knowledge?doc=42")
        generate.assert_not_called()

    def test_old_form_rejects_invalid_settings_without_generation(self):
        with patch("models.mcq_generator.generate_mcqs") as generate:
            response = self.c.post("/upload", data={
                "extracted_text": "Study material.", "num_questions": "100", "timer": "10",
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'between 1 and 30', response.data)
        generate.assert_not_called()

    def test_generation_failure_is_visible_instead_of_a_redirect(self):
        with patch("core.services.ingestion_service.ingest_document", return_value={"document_id": 42}), patch(
            "models.mcq_generator.generate_mcqs", side_effect=RuntimeError("AI service unavailable")
        ):
            response = self.c.post("/upload", data={
                "extracted_text": "Study material.", "num_questions": "2", "timer": "10",
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI service unavailable', response.data)

    def test_upload_page_uses_existing_versioned_assets(self):
        import re
        response = self.c.get("/upload")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        for extension in ("js", "css"):
            match = re.search(rb'/static/dist/app-[\w-]+\.' + extension.encode(), response.data)
            self.assertIsNotNone(match)
            asset = self.c.get(match.group().decode())
            self.assertEqual(asset.status_code, 200)
