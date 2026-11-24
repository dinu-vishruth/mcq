import json
import sqlite3

from core.repositories import session_repo
from tests.test_features import _DB, FIXED_MCQS, Base
from tests.test_quiz_integrity import bootstrap


class TestLearningEvidence(Base):
    def setUp(self):
        super().setUp()
        self.client = self._client_factory()
        self.name = self._testMethodName
        self._seed_and_login(self.client, self.name)
        self.client.post('/api/prefs', json={'goal': 'exam'})
        with sqlite3.connect(_DB) as conn:
            self.uid = conn.execute('SELECT id FROM users WHERE username=?', (self.name,)).fetchone()[0]

    def save_answer(self, *, correct=False, confidence=None, question=0, user_id=None, date='2026-01-01'):
        key = session_repo.create_session(self.name, 'medium', 120, FIXED_MCQS)
        q = FIXED_MCQS[question]
        detail = {'question': q['question'], 'correct': q['answer_text'],
                  'selected': q['answer_text'] if correct else 'columns',
                  'is_correct': correct, 'confidence': confidence}
        with sqlite3.connect(_DB) as conn:
            conn.execute('INSERT INTO results (session_key, user_id, score, total, submitted_at, detail_json) '
                         'VALUES (?, ?, ?, 1, ?, ?)',
                         (key, user_id or self.uid, int(correct), date, json.dumps([detail])))
        return key

    def test_confidence_categories_and_review_session_hide_answers(self):
        self.save_answer(confidence='know')
        self.save_answer(correct=True, confidence='maybe', question=1)
        data = self.client.get('/api/review').get_json()
        self.assertEqual(data['summary'], {'to_revisit': 1, 'uncertain': 1, 'corrected': 0,
                                          'misconceptions': 1, 'reviewable': 2})
        self.assertNotIn('questions', data)
        review = self.client.post('/api/review/start', json={}).get_json()
        self.client.post('/student_login', data={'session_key': review['session_key']})
        quiz = bootstrap(self.client.get('/mcq_test'))
        self.assertEqual(len(quiz['mcqs']), 2)
        self.assertTrue(all('answer_text' not in q for q in quiz['mcqs']))

    def test_correction_clears_queue_and_later_mistake_reopens_it(self):
        self.save_answer()
        self.save_answer(correct=True, date='2026-01-02')
        data = self.client.get('/api/review').get_json()
        self.assertEqual(data['summary']['corrected'], 1)
        self.assertEqual(data['items'], [])
        self.assertEqual(self.client.post('/api/review/start', json={}).status_code, 404)
        self.save_answer(date='2026-01-03')
        data = self.client.get('/api/review').get_json()
        self.assertEqual(data['summary']['corrected'], 0)
        self.assertEqual(data['summary']['to_revisit'], 1)
        self.assertEqual(data['items'][0]['attempts'], 3)

    def test_review_is_private_and_optional_confidence_stays_optional(self):
        self.save_answer(correct=True)
        other = self._client_factory()
        self._seed_and_login(other, self.name + '_other')
        with sqlite3.connect(_DB) as conn:
            uid = conn.execute('SELECT id FROM users WHERE username=?', (self.name + '_other',)).fetchone()[0]
        self.save_answer(user_id=uid)
        self.assertEqual(self.client.get('/api/review').get_json()['items'], [])
        self.assertEqual(other.get('/api/review').get_json()['summary']['to_revisit'], 1)
        anonymous = self._client_factory()
        self.assertEqual(anonymous.get('/api/review').status_code, 401)
        self.assertEqual(anonymous.post('/api/review/start', json={}).status_code, 401)

    def test_deleted_source_session_preserves_evidence_without_dead_review(self):
        key = self.save_answer()
        with sqlite3.connect(_DB) as conn:
            conn.execute('DELETE FROM sessions WHERE session_key=?', (key,))
        data = self.client.get('/api/review').get_json()
        self.assertEqual(data['summary']['to_revisit'], 1)
        self.assertEqual(data['summary']['reviewable'], 0)
        self.assertEqual(self.client.post('/api/review/start', json={}).status_code, 404)

    def test_submission_records_progress_without_awarding_rewards(self):
        from unittest.mock import patch
        key = session_repo.create_session(self.name, 'medium', 120, FIXED_MCQS)
        self.client.post('/student_login', data={'session_key': key})
        with patch('models.explanation_engine.explain_answers', return_value=['', '']), \
                patch('core.services.learning_service.analyse_submission'), \
                patch('core.repositories.prefs_repo.record_activity') as award:
            response = self.client.post('/submit', data={'q-0': FIXED_MCQS[0]['answer_text']})
        self.assertEqual(response.status_code, 303)
        award.assert_not_called()
        dashboard = self.client.get('/api/dashboard').get_json()
        self.assertNotIn('xp', dashboard)
        self.assertNotIn('streak', dashboard)
        self.assertNotIn('level', dashboard)
        self.assertEqual(dashboard['learning']['to_revisit'], 1)
