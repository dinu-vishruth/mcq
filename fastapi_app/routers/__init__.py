"""Routers, one module per former Flask blueprint.

    auth          /, /signup, /login, /logout, /profile, /delete_account
    google_auth   /auth/google, /auth/google/callback
    student       /dashboard, /progress, /journey, /mcq_test, /submit, ...
    documents     /upload, /ingest_resource
    teacher       /session_report, /download_report, /edit_session, ...
    api           /api/* JSON endpoints consumed by the React frontend
"""
