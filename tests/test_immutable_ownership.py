"""Renames and username reuse must never transfer private resources."""
import json
import sqlite3

from core.repositories import document_repo, job_repo, session_repo
from tests.test_features import _DB, FIXED_MCQS, Base


class TestImmutableOwnership(Base):
    def setUp(self):
        super().setUp()
        self.owner = self._client_factory()
        self.name = self._testMethodName
        self._seed_and_login(self.owner, self.name)
        with sqlite3.connect(_DB) as conn:
            self.uid = conn.execute("SELECT id FROM users WHERE username=?", (self.name,)).fetchone()[0]
        self.doc = document_repo.create(self.name, self.name, "Private notes", "paste", 20, owner_id=self.uid)
        self.key = session_repo.create_session(self.name, "medium", 300, FIXED_MCQS, owner_id=self.uid)

    def rename_and_reuse(self):
        with sqlite3.connect(_DB) as conn:
            conn.execute("UPDATE users SET username=? WHERE id=?", (self.name + "_renamed", self.uid))
        other = self._client_factory()
        self._seed_and_login(other, self.name)
        return other

    def test_rename_preserves_library_and_old_cookie_uses_current_identity(self):
        other = self.rename_and_reuse()
        own = self.owner.get('/api/knowledge').json()['items']
        self.assertIn(self.doc, [row['id'] for row in own])
        self.assertNotIn(self.doc, [row['id'] for row in other.get('/api/knowledge').json()['items']])
        self.assertEqual(other.delete(f'/api/knowledge/{self.doc}').status_code, 404)
        self.assertEqual(self.owner.get(f'/session_report/{self.key}').status_code, 200)
        self.assertEqual(other.get(f'/session_report/{self.key}').status_code, 403)
        clone = self.owner.post(f'/clone_session/{self.key}')
        self.assertEqual(clone.status_code, 302)
        with sqlite3.connect(_DB) as conn:
            owners = conn.execute('SELECT teacher_user_id FROM sessions WHERE teacher=?',
                                  (self.name + '_renamed',)).fetchall()
        self.assertEqual(owners, [(self.uid,)])

    def test_job_access_uses_id_and_never_returns_answers(self):
        job = job_repo.create(user_id=self.uid, username=self.name, document_id=self.doc, params={}, total=2)
        job_repo.set_done(job, {'session_key': self.key, 'count': 2, 'mcqs': FIXED_MCQS})
        other = self.rename_and_reuse()
        response = self.owner.get(f'/api/jobs/{job}')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('mcqs', response.json()['result'])
        self.assertNotIn('answer_text', response.text)
        self.assertEqual(other.get(f'/api/jobs/{job}').status_code, 404)

    def test_account_deletion_removes_owned_data_and_revokes_other_cookies(self):
        stale = self._client_factory()
        self._seed_and_login(stale, self.name)
        self.rename_and_reuse()
        with sqlite3.connect(_DB) as conn:
            other_id = conn.execute('SELECT id FROM users WHERE username=?', (self.name,)).fetchone()[0]
            # Another learner keeps their own completed result after its creator leaves.
            conn.execute('INSERT INTO results(session_key,user_id,score,total,detail_json) VALUES(?,?,0,1,?)',
                         (self.key, other_id, json.dumps([])))
            conn.execute('INSERT INTO quiz_attempts(session_key,user_id,started_at,deadline) VALUES(?,?,0,1)',
                         (self.key, self.uid))
        job_repo.create(user_id=self.uid, username=self.name, document_id=self.doc, params={}, total=2)
        self.assertEqual(self.owner.post('/delete_account').status_code, 302)
        self.assertEqual(stale.get('/api/knowledge').status_code, 401)
        self.assertIsNone(document_repo.get(self.doc))
        self.assertIsNone(session_repo.get(self.key))
        with sqlite3.connect(_DB) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM generation_jobs WHERE user_id=?',
                                          (self.uid,)).fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM quiz_attempts WHERE user_id=?',
                                          (self.uid,)).fetchone()[0], 0)
            self.assertIsNotNone(conn.execute('SELECT id FROM users WHERE id=?', (other_id,)).fetchone())
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM results WHERE user_id=? AND session_key=?',
                                          (other_id, self.key)).fetchone()[0], 1)

    def test_unmapped_legacy_document_is_not_assigned_by_matching_name(self):
        with sqlite3.connect(_DB) as conn:
            conn.execute('UPDATE documents SET owner_user_id=NULL WHERE id=?', (self.doc,))
        self.assertNotIn(self.doc, [row['id'] for row in self.owner.get('/api/knowledge').json()['items']])
        self.assertEqual(self.owner.delete(f'/api/knowledge/{self.doc}').status_code, 404)
