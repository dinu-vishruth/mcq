"""Run against an isolated schema in a real Postgres database.

Set MCQ_TEST_DATABASE_URL to enable. No existing tables or resources are used.
"""
import html
import json
import os
import re
import unittest
import uuid
from unittest.mock import patch

import config


@unittest.skipUnless(os.getenv("MCQ_TEST_DATABASE_URL"), "requires a test Postgres connection")
class TestSharedLibrary(unittest.TestCase):
    def test_saved_source_survives_a_new_application_and_generates_without_upload(self):
        import psycopg
        from fastapi.testclient import TestClient
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        connection_url = os.environ["MCQ_TEST_DATABASE_URL"]
        schema = "mcq_verify_" + uuid.uuid4().hex
        admin = psycopg.connect(connection_url, autocommit=True)
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            test_url = make_conninfo(connection_url, options=f"-c search_path={schema}")
            with patch.object(config, "DATABASE_URL", test_url):
                from core.repositories import document_repo, session_repo
                from fastapi_app import main

                client = TestClient(main.create_app(), follow_redirects=False)

                def csrf(c):
                    response = c.get("/upload" if c is not None and signed_in[0] else "/")
                    match = re.search(r'name="csrf-token" content="([^"]+)"', response.text) or re.search(
                        r'name="csrf_token" value="([^"]+)"', response.text)
                    return html.unescape(match.group(1))

                signed_in = [False]
                token = csrf(client)
                self.assertEqual(client.post("/signup", data={
                    "username": "library_verifier", "password": "TestPassword123", "csrf_token": token,
                }).status_code, 302)
                login = client.post("/login", data={
                    "username": "library_verifier", "password": "TestPassword123", "csrf_token": token,
                })
                self.assertEqual(login.status_code, 302)
                signed_in[0] = True
                token = csrf(client)
                material = ("Photosynthesis converts sunlight into chemical energy in chloroplasts. "
                            "Plants use carbon dioxide and water to produce glucose and oxygen. ") * 20
                saved = client.post("/ingest_resource", data={
                    "title": "Persistent biology", "extracted_text": material, "csrf_token": token,
                })
                self.assertEqual(saved.status_code, 302)
                resources = client.get("/api/knowledge").json()["items"]
                self.assertEqual(len(resources), 1)
                self.assertTrue(resources[0]["indexed"], "save must finish indexing before redirect")
                doc_id = resources[0]["id"]

                # Rebuild the app and all its connections. Only the shared DB and
                # signed browser cookie carry over, exactly as on a cold start.
                fresh = TestClient(main.create_app(), follow_redirects=False)
                fresh.cookies.update(client.cookies)
                setup = fresh.get(f"/practice?doc={doc_id}")
                bootstrap = json.loads(html.unescape(re.search(r"data-bootstrap='([^']+)'", setup.text).group(1)))
                self.assertEqual(bootstrap["document_id"], doc_id)
                self.assertEqual(bootstrap["selected_document"]["title"], "Persistent biology")
                self.assertTrue(document_repo.get_chunks(doc_id))

                questions = [{"question": "Where does photosynthesis occur?", "options": [
                    {"label": "A", "text": "Chloroplasts"}, {"label": "B", "text": "Roots"},
                    {"label": "C", "text": "Nuclei"}, {"label": "D", "text": "Ribosomes"},
                ], "answer_text": "Chloroplasts"}]
                with patch("core.services.mcq_pipeline.generate_from_document", return_value=questions) as generate:
                    result = fresh.post("/api/practice/generate", json={"document_id": doc_id, "num_questions": 1},
                                        headers={"X-CSRFToken": token})
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(generate.call_args.args[0], doc_id)
                key = result.json()["session_key"]
                joined = fresh.post("/student_login", data={"session_key": key, "csrf_token": token})
                self.assertEqual(joined.headers["location"], "/mcq_test")
                third = TestClient(main.create_app(), follow_redirects=False)
                third.cookies.update(fresh.cookies)
                self.assertIn('data-page="quiz"', third.get("/mcq_test").text)
                self.assertEqual(json.loads(session_repo.get(key)["mcqs_json"]), questions)
                # Ownership also survives a cold start; another user cannot load it.
                from core.models.db import get_db
                db = get_db()
                db.execute("UPDATE documents SET owner=? WHERE id=?", ("another_user", doc_id))
                db.commit()
                db.close()
                denied = third.get(f"/practice?doc={doc_id}")
                self.assertNotIn("Persistent biology", denied.text)
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            admin.close()
