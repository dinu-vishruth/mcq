# MCQ Generator

## Run locally

Requires Python 3.10+ and an API key for a supported LLM provider.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set these values in `.env`:

```dotenv
SECRET_KEY=replace_with_a_long_random_secret
GROQ_API_KEY=replace_with_your_api_key
LLM_PROVIDER=auto
AI_PIPELINE=legacy
```

Start the server:

```powershell
python app.py
```

Open http://127.0.0.1:8000/.

## Owner dashboard and passwords

Set `ADMIN_OWNER_EMAIL` in `.env` locally and in your hosting environment for a
deployed app. Use the email address you own.
Sign in with Google using that address, or verify the contact email through
Account security, then open `/admin`. An unverified signup email never grants
owner access. No special admin password or public role selector is used.

The dashboard shows searchable accounts, recent authenticated activity and sign-ins,
and owner audit records. You can disable or re-enable accounts, revoke their
sessions, and pause signup, demos, or new quiz generation. Existing material is
preserved. Changes require a sign-in within the last 15 minutes. Activity shown as
"seen recently" is not a live connection count; peer IP addresses may be proxies.
Access events are retained for 30 days. The admin dashboard does not expose passwords,
provider tokens, or API keys.

New and reset passwords require 8–128 characters and must pass predictable-password
checks. Spaces, Unicode, and long passphrases are supported. The same server-side
policy applies to signup, Settings, and recovery. Existing weak passwords trigger
an upgrade prompt on the next password sign-in; Settings requires the current
password and signs out other sessions when the password changes. Google and GitHub
users manage their passwords with their provider.

## Build frontend changes

Node.js and npm are needed to rebuild the frontend. Built assets in `static/dist/` are committed and served by the backend.

```powershell
cd frontend
npm ci
npm run build
```

## Deployment

Vercel uses `vercel.json` and `app.py`; Render uses the `Procfile`.
Set environment variables in the hosting platform and redeploy after changes.

- Set a stable `SECRET_KEY` and a supported provider API key, such as `GROQ_API_KEY`.
- Set `DATABASE_URL` to persistent Postgres for production. Vercel's default `/tmp` SQLite storage is temporary.
- For durable file uploads, configure `UPLOAD_BACKEND=s3`, `S3_UPLOAD_BUCKET`, `S3_UPLOAD_REGION`, and any required credentials. Alternatively, use a persistent disk on a compatible host.
- Configure Google/GitHub sign-in using [OAuth deployment setup](docs/oauth-vercel.md).
- See `.env.example` for additional settings.

## Optional background worker

Install Redis and RQ support, set `REDIS_URL` and `GENERATION_BACKEND=rq`, then run:

```powershell
python worker.py
```

## Checks

```powershell
pip install pytest ruff
pytest -q
ruff check .
```

Live provider tests require credentials and can be run with `pytest -m live`.
