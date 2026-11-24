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
