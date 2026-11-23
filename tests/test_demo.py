from unittest.mock import patch

import config
from core.models.db import get_db
from core.services.demo_service import SAMPLE_TITLE
from tests._client import make_client_factory
from tests.test_regression import Base, FIXED_MCQS, app_module


class TestDemo(Base):
    @classmethod
    def setUpClass(cls):
        cls._client_factory = staticmethod(make_client_factory(app_module.app, config.SECRET_KEY))

    def test_demo_opens_without_credentials_and_generates_from_sample(self):
        client = self.client()
        response = client.post("/demo")
        self.assertEqual(response.headers["Location"], "/dashboard")
        dashboard = client.get("/dashboard")
        self.assertIn(b"Demo Learner", dashboard.data)
        self.assertIn(b'"needs_onboarding": false', dashboard.data)
        items = client.get("/api/knowledge").get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], SAMPLE_TITLE)
        self.assertTrue(items[0]["indexed"])
        with patch("core.services.mcq_pipeline.generate_from_document", return_value=FIXED_MCQS):
            generated = client.post("/api/practice/generate", json={
                "document_id": items[0]["id"], "num_questions": 2,
            })
        self.assertEqual(generated.status_code, 200)
        joined = client.post("/student_login", data={"session_key": generated.get_json()["session_key"]})
        self.assertEqual(joined.headers["Location"], "/mcq_test")
        self.assertIn(b'data-page="quiz"', client.get("/mcq_test").data)

    def test_visitors_have_separate_demo_resources(self):
        first, second = self.client(), self.client()
        first.post("/demo"); second.post("/demo")
        a = first.get("/api/knowledge").get_json()["items"]
        b = second.get("/api/knowledge").get_json()["items"]
        self.assertNotEqual(a[0]["id"], b[0]["id"])
        denied = second.post("/api/practice/generate", json={"document_id": a[0]["id"]})
        self.assertEqual(denied.status_code, 404)

    def test_repeat_demo_does_not_replace_an_existing_session(self):
        client = self.client()
        client.post("/demo")
        before = client.get("/api/knowledge").get_json()
        client.post("/demo")
        self.assertEqual(client.get("/api/knowledge").get_json(), before)

    def test_failed_preparation_does_not_leave_an_account(self):
        db = get_db()
        before = db.execute("SELECT COUNT(*) FROM users WHERE auth_provider='demo'").fetchone()[0]
        db.close()
        with patch("core.services.ingestion_service.ingest_document", side_effect=RuntimeError("unavailable")):
            response = self.client().post("/demo")
        self.assertIn(b"The demo could not start", response.data)
        db = get_db()
        self.assertEqual(db.execute("SELECT COUNT(*) FROM users WHERE auth_provider='demo'").fetchone()[0], before)
        db.close()

    def test_demo_keeps_csrf_protection(self):
        client = self.client()
        self.assertEqual(client._client.post("/demo").status_code, 400)

    def test_login_replaces_old_public_password_hint(self):
        response = self.client().get("/")
        self.assertIn(b'Try Demo', response.data)
        self.assertNotIn(b'student123', response.data)
