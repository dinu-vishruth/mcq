"""Offline regression coverage for quiz secrecy, deadlines and durable results."""
import html
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from core.repositories import session_repo
from tests.test_features import _DB, FIXED_MCQS, Base


def bootstrap(response):
    raw = re.search(r"data-bootstrap='([^']*)'", response.text).group(1)
    return json.loads(html.unescape(raw))


class TestQuizIntegrity(Base):
    def setUp(self):
        super().setUp()
        self.client = self._client_factory()
        self._seed_and_login(self.client, self._testMethodName)
        self.client.post('/api/prefs', json={
            'goal': 'exam', 'style': 'concepts', 'daily_minutes': 30,
        })
        self.key = session_repo.create_session(self._testMethodName, 'medium', 60, FIXED_MCQS)
        for target, value in (
            ('models.explanation_engine.explain_answers', ['Explanation', 'Explanation']),
            ('core.services.learning_service.analyse_submission', None),
        ):
            stub = patch(target, return_value=value)
            stub.start()
            self.addCleanup(stub.stop)
        with self.clock(1000):
            self.client.post('/student_login', data={'session_key': self.key})

    def clock(self, now):
        return patch('fastapi_app.routers.student.time', SimpleNamespace(time=lambda: now))

    def answers(self):
        return {f'q-{i}': q['answer_text'] for i, q in enumerate(FIXED_MCQS)}

    def result(self):
        with sqlite3.connect(_DB) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute('SELECT * FROM results WHERE session_key=?', (self.key,)).fetchone()

    def test_page_contains_no_answer_keys_and_reload_preserves_deadline(self):
        with self.clock(1010):
            first = self.client.get('/mcq_test')
        with self.clock(1030):
            second = self.client.get('/mcq_test')
        self.assertNotIn('answer_text', first.text)
        self.assertEqual(bootstrap(first)['deadline'], 1060)
        self.assertEqual(bootstrap(second)['deadline'], 1060)
        self.assertEqual(bootstrap(second)['timer'], 30)
        for question in bootstrap(first)['mcqs']:
            self.assertEqual(set(question), {'question', 'options'})

    def test_elapsed_time_ignores_forged_form_and_confidence_is_recorded(self):
        with self.clock(1042):
            response = self.client.post('/submit', data={
                **self.answers(), 'time_spent': '-999999', 'confidence-0': 'maybe',
                'confidence-1': 'invalid',
            })
        self.assertEqual(response.status_code, 200)
        row = self.result()
        self.assertEqual((row['score'], row['time_spent']), (2, 42))
        details = json.loads(row['detail_json'])
        self.assertEqual(details[0]['confidence'], 'maybe')
        self.assertIsNone(details[1]['confidence'])

    def test_draft_restores_answers_confidence_and_question(self):
        with self.clock(1020):
            saved = self.client.post('/api/quiz-attempt', json={
                'session_key': self.key, 'answers': {'0': FIXED_MCQS[0]['answer_text'], '1': 'fake'},
                'confidence': {'0': 'know', '1': 'fake'}, 'question_index': 1,
            })
            page = bootstrap(self.client.get('/mcq_test'))
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(page['answers'], {'0': FIXED_MCQS[0]['answer_text']})
        self.assertEqual(page['confidence'], {'0': 'know'})
        self.assertEqual(page['question_index'], 1)

    def test_drafts_are_private_and_require_a_started_attempt(self):
        other = self._client_factory()
        self._seed_and_login(other, 'quiz_integrity_other')
        with self.clock(1020):
            response = other.post('/api/quiz-attempt', json={
                'session_key': self.key, 'answers': {'0': FIXED_MCQS[0]['answer_text']},
            })
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self._client_factory().post('/api/quiz-attempt', json={
            'session_key': self.key,
        }).status_code, 401)

    def test_late_submission_grades_only_the_draft_saved_before_deadline(self):
        with self.clock(1050):
            self.client.post('/api/quiz-attempt', json={
                'session_key': self.key, 'answers': {'0': FIXED_MCQS[0]['answer_text']},
                'confidence': {'0': 'maybe'},
            })
        with self.clock(1070):
            rejected = self.client.post('/api/quiz-attempt', json={
                'session_key': self.key, 'answers': {'1': FIXED_MCQS[1]['answer_text']},
            })
            response = self.client.post('/submit', data=self.answers())
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(response.status_code, 200)
        row = self.result()
        self.assertEqual((row['score'], row['time_spent']), (1, 60))
        self.assertEqual(json.loads(row['detail_json'])[0]['confidence'], 'maybe')

    def test_auto_submit_accepts_bounded_network_grace(self):
        with self.clock(1061):
            self.client.post('/submit', data=self.answers())
        self.assertEqual((self.result()['score'], self.result()['time_spent']), (2, 60))

    def test_explanation_failure_still_saves_result_once(self):
        with self.clock(1040), patch('models.explanation_engine.explain_answers', side_effect=RuntimeError):
            response = self.client.post('/submit', data=self.answers())
            repeated = self.client.post('/submit', data=self.answers())
        self.assertEqual(response.status_code, 200)
        self.assertIn('temporarily unavailable', response.text)
        self.assertEqual(self.result()['score'], 2)
        self.assertIn('already taken', bootstrap(repeated)['error'])
        with sqlite3.connect(_DB) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM results WHERE session_key=?',
                                          (self.key,)).fetchone()[0], 1)
        with self.clock(1040):
            self.assertEqual(bootstrap(self.client.get('/dashboard'))['total_correct'], 2)

    def test_error_page_keeps_real_dashboard_statistics(self):
        with self.clock(1040):
            self.client.post('/submit', data=self.answers())
            response = self.client.post('/student_login', data={'session_key': 'invalid!'})
        data = bootstrap(response)
        self.assertEqual(data['total_quizzes'], 1)
        self.assertEqual(data['total_correct'], 2)
        self.assertEqual(data['error'], 'Invalid session key format')

    def test_concurrent_submissions_record_one_result(self):
        from fastapi_app.routers import student

        other = self._client_factory()
        self._seed_and_login(other, self._testMethodName)
        with self.clock(1010):
            other.post('/student_login', data={'session_key': self.key})
        barrier = Barrier(2)
        original = student._already_attempted

        def simultaneous_check(conn, key, user_id):
            attempted = original(conn, key, user_id)
            barrier.wait(timeout=10)
            return attempted

        with self.clock(1040), patch.object(student, '_already_attempted', simultaneous_check):
            with ThreadPoolExecutor(max_workers=2) as pool:
                submissions = [pool.submit(client.post, '/submit', data=self.answers())
                               for client in (self.client, other)]
                responses = [future.result(timeout=20) for future in submissions]
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(sum('data-page="result"' in response.text for response in responses), 1)
        with sqlite3.connect(_DB) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM results WHERE session_key=?',
                                          (self.key,)).fetchone()[0], 1)
        self.assertEqual(bootstrap(self.client.get('/dashboard'))['total_correct'], 2)
