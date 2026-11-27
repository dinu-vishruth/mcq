"""Exercise delivery, token rollback, and real verification/reset routes."""
from unittest.mock import Mock

import config
from tests.test_admin_dashboard import PASSWORD, sign_in
from tests.test_admin_dashboard import workspace as workspace


def setup_mail(monkeypatch):
    monkeypatch.setattr(config, 'PUBLIC_BASE_URL', 'https://examly.example')
    monkeypatch.setattr(config, 'SMTP_HOST', 'smtp.example')
    monkeypatch.setattr(config, 'SMTP_FROM', 'sender@example.com')
    monkeypatch.setattr(config, 'SMTP_USERNAME', 'sender@example.com')
    monkeypatch.setattr(config, 'SMTP_PASSWORD', 'test-only')
    monkeypatch.setattr(config, 'SMTP_STARTTLS', True)


def test_signup_verification_then_recovery_and_token_replay(workspace, monkeypatch):
    _, conn, factory = workspace
    setup_mail(monkeypatch)
    sender = Mock()
    monkeypatch.setattr('core.services.account_recovery.send_token', sender)
    client = factory()
    assert client.post('/signup', data={'username': 'new-person', 'email': 'new@example.com', 'password': PASSWORD}).status_code == 302
    email, token, purpose = sender.call_args.args
    assert (email, purpose) == ('new@example.com', 'verify')
    assert conn.execute("SELECT email_verified FROM users WHERE username='new-person'").fetchone()[0] == 0
    assert client.get('/account/confirm?token=' + token).status_code == 200
    assert conn.execute("SELECT email_verified FROM users WHERE username='new-person'").fetchone()[0] == 0
    assert b'Account updated' in client.post('/account/confirm', data={'token': token}).data
    assert client.post('/account/recover', data={'email': email}).status_code == 200
    _, reset, purpose = sender.call_args.args
    assert purpose == 'reset'
    replacement = 'Another cedar mountain 83!'
    assert b'Account updated' in client.post('/account/confirm', data={'token': reset, 'password': replacement}).data
    assert b'invalid or expired' in client.post('/account/confirm', data={'token': reset, 'password': replacement}).data
    assert client.post('/login', data={'username': 'new-person', 'password': PASSWORD}).status_code == 200
    assert client.post('/login', data={'username': 'new-person', 'password': replacement}).status_code == 302


def test_failed_delivery_allows_retry_and_does_not_expose_token(workspace, monkeypatch, caplog):
    _, conn, factory = workspace
    setup_mail(monkeypatch)
    client = sign_in(factory, 'learner')
    sender = Mock(side_effect=RuntimeError('private smtp credential'))
    monkeypatch.setattr('core.services.account_recovery.send_token', sender)
    assert client.post('/account/verify').status_code == 503
    assert conn.execute('SELECT count(*) FROM account_tokens').fetchone()[0] == 0
    assert 'private smtp credential' not in caplog.text
    sender.side_effect = None
    assert client.post('/account/verify').status_code == 303
    assert sender.call_count == 2


def test_smtp_starts_tls_before_authentication(monkeypatch):
    from core.services.account_recovery import send_token
    setup_mail(monkeypatch)
    smtp = Mock()
    context = Mock()
    context.__enter__ = Mock(return_value=smtp)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr('core.services.account_recovery.smtplib.SMTP', Mock(return_value=context))
    send_token('recipient@example.com', 'private-token', 'verify')
    assert [call[0] for call in smtp.method_calls] == ['starttls', 'login', 'send_message']
    message = smtp.send_message.call_args.args[0]
    assert 'https://examly.example/account/confirm?token=private-token' in message.get_content()
