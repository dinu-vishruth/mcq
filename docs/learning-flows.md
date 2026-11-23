# Learning flow fixes and deployment checks

## One-time onboarding

The server reads `user_prefs.onboarded` for the authenticated account. A new
account's first login opens `/onboarding`. Finish or Skip must successfully save
`/api/prefs` before leaving that page. Dashboard no longer mounts a questionnaire.
Already-completed accounts visiting `/onboarding` go straight to Dashboard.
Settings remains the voluntary way to edit preferences. Preferences are passed
to AI prompts in request-local context so concurrent learners cannot share them.

Account data must live in a persistent database. On Vercel, set `DATABASE_URL`
to your Postgres connection string. A SQLite file in `/tmp` disappears on cold
starts and cannot enforce one-time onboarding across instances. Do not rely on
browser storage or a session cookie as the durable completion record.

## Library versus Make Quiz

- `/ingest_resource` saves reusable resources and indexes their content.
- `/upload` saves only a temporary, user-owned quiz input in `quiz_uploads`.
  It does not create documents, embeddings, Library entries or original-file copies.
- Input is accessible for up to 24 hours, retained after generation errors,
  deleted after successful generation, and expired rows are purged on the next upload.
- Saved Library resources can still generate quizzes repeatedly without reuploading.
- Existing resources are preserved: old auto-saved uploads cannot safely be
  distinguished from resources users deliberately saved, so none are deleted.
- One-off uploads generate inline. Saved resources keep existing inline/RQ support;
  the quiz screen now waits for queued jobs before opening the quiz.

Both SQLite and Postgres receive migration 004 automatically on app startup.

## Why the AI configuration error happens

`models.mcq_generator.generate_mcqs` checks `config.llm_key_present()`. If false,
it raises the reported configuration error before calling any AI provider. A
callback URL or OAuth client secret does not configure the quiz AI service.
Local `.env` values do not transfer to Vercel (`.vercelignore` excludes them).

Separately, provider selection previously selected OpenAI from `OPENAI_API_KEY`
but passed an empty unified key to its adapter. The provider now receives the
OpenAI credential correctly. No credential values are logged or returned to UI.

In the Vercel project serving your production domain:

1. Set a real supported AI credential in Production, for example `GROQ_API_KEY`.
2. Use `LLM_PROVIDER=auto`, or select the provider matching your credential.
3. Set persistent `DATABASE_URL` and retain your stable `SECRET_KEY`.
4. Deploy the updated code and built `static/dist` assets to Production.
5. Check `/api/generation/status` while signed in. `available` must be true.
6. Complete onboarding, log out, log in, and revisit Dashboard: no questionnaire.
7. Upload through Make Quiz: Library stays unchanged; generate a playable quiz.
8. Add a resource through Library: reuse it for another quiz without reuploading.

The status endpoint validates local configuration, not provider connectivity,
key validity, billing or quota. Those are confirmed only by a successful real
generation.

## Private quizzes, AI time budgets, and evidence

Quizzes are private to their immutable owner ID, including join, draft, resume,
and submission. New session keys use a full UUID. Legacy quizzes with unassigned
ownership remain inaccessible until an operator applies a reviewed ownership map.

Migration 007 adds generation reservations for SQLite and Postgres. A transaction
admits at most one in-flight generation per user, even when cooldowns are disabled.
Failed attempts count toward the five-minute attempt limit. Reservations are
released on completion/failure and expire after a crash. Queued requests transfer
their reservation to the worker; duplicate deliveries cannot claim a job twice.
A queue outage returns a retryable failure instead of running a worker-sized job
inside the web request.

`PIPELINE_DEADLINE_SECONDS` supplies a request-local deadline shared by retrieval,
LLM requests, retries and fallback generation. Vercel inline requests are capped
at 50 seconds. Network timeouts shrink to the remaining budget; retries never
sleep beyond it or retry earlier than the provider's Retry-After. This is
cooperative deadline enforcement, not a mechanism for killing arbitrary Python
code or a database query. RQ's worker timeout supplies the separate job process
limit when queue mode is configured.

Submission commits the score and redirects without calling the AI. The visible
result page starts a separate CSRF-protected `POST /api/results/{key}/enrich`.
Its atomic claim prevents repeated provider calls/analysis across tabs, and
expired claims become a visible feedback failure without replaying partial
learning writes. Old results are marked done by the migration to avoid
reanalyzing historical attempts. Connections and blocking work run in the
thread pool, leaving the event loop available.
`RESULT_ENRICHMENT_DEADLINE_SECONDS` defaults to 12 seconds for optional AI feedback
and topic labeling. Existing grounded explanations are reused without another
explanation call; enrichment failure cannot discard the result.

Both generation modes, session persistence and edited quizzes use the shared MCQ
contract: four distinct text options and exactly one matching answer. RAG
verification accepts only typed boolean/confidence verdicts. Missing or malformed
verdicts remain explicitly unverified; explicit failures are rejected.

Source excerpts are retained only when they occur in the actual generation
context and source chunk. The application supplies document/chunk IDs rather
than trusting model-generated IDs. Supporting excerpts, verification status and
generated explanations are stored with scored results and shown only after
grading. A matching excerpt establishes provenance, not proof of factual truth.
