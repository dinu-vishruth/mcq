"""Offline integration coverage for one-time onboarding and unsaved quiz inputs."""
import sqlite3
from unittest.mock import Mock, patch

import config
from core.llm import registry
from tests.test_features import _DB, FIXED_MCQS, Base


class TestLearningPreferences(Base):
    def test_completed_preferences_survive_logout_new_client_and_app_restart(self):
        client = self._client_factory()
        self._seed_and_login(client, "prefs_once")
        self.assertEqual(client.get("/dashboard").headers["Location"], "/onboarding")
        choices = {"goal": "interview", "style": "concepts", "daily_minutes": 60}
        self.assertEqual(client.post("/api/prefs", json=choices).status_code, 200)
        self.assertFalse(client.get("/api/prefs").json()["needs_onboarding"])
        self.assertEqual(client.get("/dashboard").status_code, 200)
        self.assertEqual(client.get("/onboarding").headers["Location"], "/dashboard")
        client.get("/logout")
        from fastapi_app.main import create_app
        from tests._client import FlaskLikeClient
        fresh = FlaskLikeClient(create_app(), config.SECRET_KEY)
        response = fresh.post("/login", data={"username": "prefs_once", "password": "password123"})
        self.assertEqual(response.headers["Location"], "/dashboard")
        stored = fresh.get("/api/prefs").json()
        self.assertFalse(stored["needs_onboarding"])
        for key, value in choices.items():
            self.assertEqual(stored["prefs"][key], value)
        for _ in range(3):
            self.assertEqual(fresh.get("/dashboard").status_code, 200)

    def test_preferences_reach_ai_and_do_not_leak_between_users(self):
        from core.services.personalization import learning_profile, system_prompt
        client = self._client_factory()
        self._seed_and_login(client, "personalized")
        client.post("/api/prefs", json={"goal": "interview", "style": "concepts", "daily_minutes": 15})
        with sqlite3.connect(_DB) as conn:
            user_id = conn.execute("SELECT id FROM users WHERE username='personalized'").fetchone()[0]
        fake = Mock()
        fake.complete_json.return_value = {"questions": FIXED_MCQS[:1]}
        from core.agents.question import QuestionAgent
        with learning_profile(user_id), patch("core.agents.question.get_llm", return_value=fake):
            QuestionAgent().run("WHERE filters rows.", 1, "easy")
        system = fake.complete_json.call_args.args[0][0]["content"]
        self.assertIn("job interviews", system)
        self.assertIn("underlying concepts", system)
        self.assertIn("15 minutes", system)
        self.assertEqual(system_prompt("base"), "base")
        with learning_profile(None):
            self.assertEqual(system_prompt("base"), "base")


class TestOneOffQuiz(Base):
    def _upload(self, client):
        response = client.post("/upload", data={
            "title": "Temporary Input", "extracted_text": "Transactions are atomic. " * 30,
        })
        self.assertEqual(response.status_code, 302)
        return response.headers["Location"].split("upload=")[1]

    def test_generation_uses_upload_without_library_or_original_file_storage(self):
        client = self._client_factory()
        self._seed_and_login(client, "temp_quiz")
        with patch("fastapi_app.routers.documents.persist_original_file") as store:
            upload_id = self._upload(client)
            store.assert_not_called()
        with patch("core.services.generation_status.generation_available", return_value=True), \
             patch("core.services.mcq_pipeline.generate_from_text", return_value=FIXED_MCQS) as generate:
            response = client.post("/api/practice/generate", json={"upload_id": upload_id})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["requested_count"], 5)
        self.assertTrue(response.json()["partial"])
        self.assertEqual(response.json()["quality_policy"], "verified_only")
        generate.assert_called_once()
        self.assertEqual(client.get("/api/knowledge").json()["items"], [])
        self.assertEqual(client.get(f"/api/quiz-uploads/{upload_id}").status_code, 404)
        key = response.json()["session_key"]
        self.assertEqual(client.post("/student_login", data={"session_key": key}).status_code, 302)
        self.assertEqual(client.get("/mcq_test").status_code, 200)

    def test_upload_is_private_and_expired_sources_are_rejected(self):
        owner = self._client_factory()
        self._seed_and_login(owner, "temp_owner")
        upload_id = self._upload(owner)
        other = self._client_factory()
        self._seed_and_login(other, "temp_other")
        self.assertEqual(other.get(f"/api/quiz-uploads/{upload_id}").status_code, 404)
        self.assertEqual(other.post("/api/practice/generate", json={"upload_id": upload_id}).status_code, 404)
        with sqlite3.connect(_DB) as conn:
            conn.execute("UPDATE quiz_uploads SET expires_at='2000-01-01' WHERE id=?", (upload_id,))
        self.assertEqual(owner.get(f"/api/quiz-uploads/{upload_id}").status_code, 404)
        self._upload(owner)
        with sqlite3.connect(_DB) as conn:
            self.assertIsNone(conn.execute("SELECT id FROM quiz_uploads WHERE id=?", (upload_id,)).fetchone())

    def test_missing_ai_key_fails_before_generation_and_keeps_input(self):
        client = self._client_factory()
        self._seed_and_login(client, "temp_missing_ai")
        upload_id = self._upload(client)
        with patch("config.llm_key_present", return_value=False), \
             patch("core.services.mcq_pipeline.generate_from_text") as generate:
            self.assertFalse(client.get("/api/generation/status").json()["available"])
            response = client.post("/api/practice/generate", json={"upload_id": upload_id})
        self.assertEqual(response.status_code, 503)
        self.assertIn("administrator", response.json()["error"])
        generate.assert_not_called()
        self.assertEqual(client.get(f"/api/quiz-uploads/{upload_id}").status_code, 200)

    def test_generation_failure_keeps_input_for_retry(self):
        client = self._client_factory()
        self._seed_and_login(client, "temp_retry")
        upload_id = self._upload(client)
        with patch("core.services.generation_status.generation_available", return_value=True), \
             patch("core.services.mcq_pipeline.generate_from_text", side_effect=ValueError("Provider unavailable")):
            response = client.post("/api/practice/generate", json={"upload_id": upload_id})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(client.get(f"/api/quiz-uploads/{upload_id}").status_code, 200)


class TestProviderCredentialSelection(Base):
    def test_openai_specific_key_is_passed_to_provider_auto_and_explicit(self):
        for selection in ("auto", "openai"):
            with self.subTest(selection=selection), patch.multiple(
                config, LLM_PROVIDER=selection, LLM_API_KEY="", OPENAI_API_KEY="sk-test-only",
                GEMINI_API_KEY="", ANTHROPIC_API_KEY="", LLM_MODEL="",
            ):
                self.assertEqual(registry.build_llm().api_key, "sk-test-only")

    def test_blank_selected_provider_key_is_not_reported_ready(self):
        from core.services.generation_status import generation_available
        with patch("config.llm_key_present", return_value=True), \
             patch("core.llm.get_llm", return_value=Mock(api_key="")):
            self.assertFalse(generation_available())
