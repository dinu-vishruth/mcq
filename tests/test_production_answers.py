"""The quiz page exposes choices, never an answer key or grading authority."""
import html
import json
import re

from core.models.db import get_db
from tests.production_support import QUESTION


def test_quiz_bootstrap_has_no_answer_key(production_app):
    client, _ = production_app.client()
    key = production_app.session()
    client.post("/student_login", data={"session_key": key})
    response = client.get("/mcq_test")
    assert response.status_code == 200
    assert "answer_text" not in response.text
    bootstrap = json.loads(html.unescape(re.search(r"data-bootstrap='(.*?)'", response.text, re.S)[1]))
    assert set(bootstrap["mcqs"][0]) == {"question", "options"}
    # The correct text must remain visible as one of the four unlabeled choices.
    assert {o["text"] for o in bootstrap["mcqs"][0]["options"]} == {o["text"] for o in QUESTION["options"]}


def test_grading_uses_database_not_posted_answer_key(production_app):
    client, user_id = production_app.client()
    key = production_app.session()
    client.post("/student_login", data={"session_key": key})
    response = client.post("/submit", data={"q-0": QUESTION["answer_text"], "answer_text": "forged"})
    assert response.status_code == 200
    conn = get_db()
    try:
        assert conn.execute("SELECT score FROM results WHERE user_id=?", (user_id,)).fetchone()[0] == 1
    finally:
        conn.close()


def test_posted_answer_key_cannot_make_wrong_choice_correct(production_app):
    client, user_id = production_app.client()
    client.post("/student_login", data={"session_key": production_app.session()})
    client.post("/submit", data={"q-0": "option-one", "answer_text": "option-one"})
    conn = get_db()
    try:
        assert conn.execute("SELECT score FROM results WHERE user_id=?", (user_id,)).fetchone()[0] == 0
    finally:
        conn.close()
