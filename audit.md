**Examly’s unfinished feeling comes mainly from gaps between its promises and its behavior:** incomplete source coverage, inconsistent question validation, misleading progress indicators, and learning analytics that track mistakes without reliably measuring recovery.

The core quiz workflow exists and has meaningful safeguards. However, I would **hold a public release until the authentication and generation-reliability issues below are fixed.**

**Audit scope and evidence**

I audited the committed development worktree at `c8b8f68`, branch `dev/examly-hardening`. This includes changes beyond the [public repository’s default branch](https://github.com/dinu-vishruth/mcq); the commit was not pushed in our preceding work.

The browser connection was unavailable, and the web fetch could not access the live application. Therefore:

- UI findings below come from source inspection, not rendered desktop/mobile observations.
- I cannot confirm which commit, database, AI provider, or environment settings production currently uses.
- Failure to fetch the URL is **not evidence that the application is down**.

Earlier in this conversation, the development code passed **263 backend tests, 11 frontend tests, two browser flows, lint, and the frontend build**. One PostgreSQL integration test was skipped and two live tests were deselected. Those checks establish useful regression coverage; they do not establish live question accuracy or production readiness.

For this audit, I additionally ran isolated, in-memory checks confirming three problems:

1. OAuth can link to an existing local account by its unverified email while retaining its password.
2. The legacy question validator accepts an answer absent from its options.
3. The fact verifier accepts the string `"false"` as a passing verdict.

**No project files were changed.**

---

**1. Architecture overview**

Examly is a **FastAPI application serving Jinja page shells containing React screens**. It is not currently a React single-page application, and Flask is no longer the active backend.

| Layer | Actual implementation | Evidence |
|---|---|---|
| Application entry | `app.py` exports the FastAPI application | [app.py:13](C:/Users/Dinu/Documents/mcq_generator/.development/app.py:13) |
| Backend | FastAPI routers, Pydantic request models, dependency-based authentication | [main.py:79](C:/Users/Dinu/Documents/mcq_generator/.development/fastapi_app/main.py:79), `fastapi_app/deps.py:57` |
| Frontend | React 18, TypeScript, Vite, Tailwind, Framer Motion, Lucide, Recharts | `frontend/package.json:14` |
| Page delivery | Jinja embeds initial JSON; React mounts the corresponding lazy-loaded screen | [main.tsx:9](C:/Users/Dinu/Documents/mcq_generator/.development/frontend/src/main.tsx:9), `frontend/src/bootstrap.ts:12` |
| Relational storage | SQLite by default; PostgreSQL when `DATABASE_URL` is configured | [db.py:46](C:/Users/Dinu/Documents/mcq_generator/.development/core/models/db.py:46) |
| Authentication | Password hashes, signed session cookies, CSRF tokens; Google and GitHub OAuth | `fastapi_app/routers/auth.py`, `google_auth.py`, `github_auth.py` |
| LLM integration | Shared interface with Groq/xAI/OpenAI-compatible, Gemini, and Anthropic adapters | `core/llm/registry.py:138`, `core/llm/base.py:33` |
| Embeddings | Automatic SentenceTransformer selection when available; otherwise feature hashing; optional remote embeddings | `core/embeddings/registry.py:28` |
| Retrieval storage | NumPy search over database vectors; optional Chroma, FAISS, pgvector implementations | `core/vectorstore/registry.py:19` |
| Generation orchestration | Synchronous inline generation by default; optional Redis/RQ worker path | `core/services/job_runner.py:25` |
| Deployment | Vercel Python entry point; committed frontend assets; alternative Uvicorn Procfile | `vercel.json:3`, `frontend/vite.config.ts:17`, `Procfile:1` |

Two details matter:

- The class named `SqliteNumpyStore` uses `get_db()`. With `DATABASE_URL`, its binary vectors reside in PostgreSQL and similarity calculations still happen in Python. Selecting this class does **not necessarily mean production vectors are stored in temporary SQLite**.
- The “agents” are explicit Python pipeline stages. They are not autonomous agents planning and operating tools. That is a reasonable design; describe it accurately in your portfolio.

**The actual document-to-result journey**

**Saved-resource path**

1. `/add_resource` opens `AddResource.tsx`.
2. PDF/DOCX/TXT extraction normally happens in the browser. PPTX and extraction fallbacks go to the server.
3. Browser-extracted text is sampled to approximately 150,000 characters.
4. `/ingest_resource` cleans, chunks, embeds, and stores the resource **before redirecting**.
5. `/journey` lists the saved resources.
6. `/practice?doc=ID` opens quiz settings with the selected resource.
7. `/api/practice/generate` checks ownership, availability, and generation throttling.
8. Retrieval selects document chunks and builds a context.
9. The configured generator creates questions and stores a quiz session.
10. A form POST to `/student_login` starts the attempt.
11. `/mcq_test` returns questions and shuffled options, keeping answer keys on the server.
12. `/submit` scores the attempt, saves the result, generates explanations, labels topics, and updates XP/streak.
13. `/results/{session_key}` loads the saved result.

**Temporary-upload path**

1. `/upload` extracts material.
2. The server stores temporary text in `quiz_uploads`, with a 24-hour expiry.
3. Quiz settings receive an `upload_id`.
4. `generate_from_text()` uses the **first `CONTEXT_MAX_CHARS` characters**, default 12,000.
5. This path does **not** ingest, embed, or retrieve the temporary document.
6. Successful generation deletes the temporary input.

Evidence: [documents.py:125](C:/Users/Dinu/Documents/mcq_generator/.development/fastapi_app/routers/documents.py:125), [api.py:290](C:/Users/Dinu/Documents/mcq_generator/.development/fastapi_app/routers/api.py:290), [mcq_pipeline.py:189](C:/Users/Dinu/Documents/mcq_generator/.development/core/services/mcq_pipeline.py:189).

This means **“RAG-based MCQ platform” describes the saved-resource workflow more accurately than every upload workflow.**

**Important endpoints**

| Endpoint | Purpose and frontend interaction |
|---|---|
| `POST /signup`, `/login`, `/demo` | Account creation, authentication, isolated demo workspace |
| `GET/POST /api/prefs` | Onboarding and learning preferences |
| `POST /ingest_resource` | Save and index a Library resource |
| `POST /upload` | Prepare temporary quiz material; also retains an older direct-generation form contract |
| `GET /api/knowledge` | Populate Journey and resource selection |
| `DELETE /api/knowledge/{id}` | Remove an owned resource |
| `GET /api/generation/status` | Check configured generation availability |
| `POST /api/practice/generate` | Generate a quiz; returns a session key or queued job ID |
| `GET /api/jobs/{id}` | Poll queued generation |
| `GET /api/jobs/{id}/events` | Implemented server-sent event stream; current MakeQuiz screen polls instead |
| `POST /student_login` | Join/start a quiz using its key |
| `POST /api/quiz-attempt` | Save answers, confidence, and current question |
| `POST /submit` | Authoritative scoring and post-submission processing |
| `GET /results/{key}` | Retrieve the authenticated user’s saved result |
| `GET /api/weak-topics` | Fetch accumulated topic miss rates |
| `POST /api/weak-topics/review` | Rebuild a quiz from actual missed questions |
| `GET /api/export` | Export a subset of account and learning data |

---

**2. Feature inventory**

“Implemented” below means wired into the current application, not verified against production.

| Feature | Status | Important qualification |
|---|---|---|
| Password registration/login | Implemented | No password-recovery route found |
| Google/GitHub login | Implemented, configuration-dependent | Unsafe email-based linking needs fixing |
| One-click demo | Implemented | Creates real database records and indexed sample material |
| Onboarding/preferences | Implemented | Preferences influence prompts; this is not automatic ability adaptation |
| PDF/DOCX/PPTX/TXT upload | Implemented | Browser/server extraction differ; no OCR |
| Saved Library | Implemented | Presented as “Learning Journey” |
| Temporary quiz uploads | Implemented | Separate from Library; not a retrieval workflow |
| Difficulty/count/timer/topic controls | Implemented | Difficulty is not independently validated in default mode |
| Retrieval and MMR | Implemented for saved resources | Advertised document-spread selection is incomplete |
| Structural question QA | Implemented in RAG mode | Default legacy path uses a weaker validator |
| Context/fact verification | Conditional | Disabled by pipeline selection in default legacy behavior; fail-open when used |
| Timed quizzes | Implemented | Server deadline, draft saving, shuffled options, skip, confidence |
| Durable results/idempotent submission | Implemented | Stronger than a basic generated-quiz demo |
| Explanations | Implemented, incomplete grounding | Normal submission does not pass the source document ID |
| Dashboard/progress | Implemented | Several metrics and claims need clearer semantics |
| Weak-topic review | Implemented | Replays historical mistakes; lacks recovery scheduling |
| XP/streak/badges | Implemented | Definitions differ between screens |
| JSON/PDF/CSV exports | Implemented | Account export is not “everything stored” |
| Session editing/archive/clone/report routes | Implemented backend routes | Legacy teacher-oriented features are poorly surfaced in the unified product |
| Durable background generation | Optional | Requires separately configured dependencies and worker |
| Original-file storage | Partial/optional | Normal browser extraction clears the file input, so original bytes are not generally retained |
| Flashcards/summaries/interview content | Component code only | `generate_content()` has tests but no active product route/UI |
| Free-text answer grading | Component code only | `AnswerEvaluationAgent` is not used by MCQ submission |
| AI revision guidance agent | Dormant component | Current Weak Topics screen rebuilds missed questions instead |
| Chat/planner interface | Not implemented as a user flow | Planner is conditional on `user_request`; current settings do not supply one |

Unused or misleading remnants include `core/routes/` Flask blueprints, stale Flask comments, the old teacher-oriented template, and an unused lexical difficulty classifier. The active `DifficultyAgent.run()` does not implement the lexical prefilter described in its docstring.

**Recommendation:** preserve useful compatibility wrappers until callers are verified, but remove dormant product claims from documentation now. Do not expose every unused component as a new feature.

---

**3. UI/UX audit — provisional score: 5.5/10**

The code already has reusable components, consistent primary colors, responsive breakpoints, fluid headings, and clear quiz option layouts. The weaker areas are trust, navigation, accessible interaction, and continuity after finishing a quiz.

This score assesses the implemented experience from code. **Rendered visual balance, font loading, mobile overflow, and touch usability remain unverified.**

| Priority | Problem and evidence | Why it affects users | Recommended improvement |
|---|---|---|---|
| **High** | Upload stages advance with `setTimeout`; temporary upload even shows “Creating Chunks” although that path does not chunk. `Upload.tsx:19,69`; `AddResource.tsx:68` | Progress appears precise while reporting unconfirmed work | Show actual local extraction progress, then an honest indeterminate server-processing state |
| **High** | API failures become empty/success-like screens. `WeakTopics.tsx:29`; `Dashboard.tsx:43` | A failed lookup can look like “no weak spots” or zero progress | Give loading, empty, failed, and unavailable states separate treatments with retry |
| **High** | Historical results cannot be opened from dashboard cards. `Dashboard.tsx:219` | Users cannot easily return to explanations, despite durable result routes existing | Link every recent session to its saved result |
| **High** | Results send “Generate New Quiz” to `/upload`. `Result.tsx:201` | Saved-resource users are pushed toward reuploading and lose study context | Offer “Practice this source again” and “Review missed questions” |
| **High** | Selected options/settings are ordinary buttons without radio/pressed semantics. `Quiz.tsx:202`; `MakeQuiz.tsx:231` | Screen-reader users lack a reliable selected-state announcement | Use native radios or equivalent accessible grouped controls |
| **High** | File input is `display:none`; its label is not keyboard-focusable. `Upload.tsx:121,134`; `AddResource.tsx:125,138` | Keyboard-only users lack a clear upload activation control | Use a visible button or visually hidden focusable input |
| **High** | Small text and primary-button color combinations have inadequate token contrast | Metadata and actions become difficult to read | Adjust text/accent tokens and verify all enabled states |
| **Medium** | “Learning Journey” means Library; `/knowledge` means quiz creation; `/practice` mounts the same screen. `AppShell.tsx:10`; `main.tsx:12` | Users must learn internal terminology to find ordinary actions | Use clear labels: Library, Create Quiz, Progress |
| **Medium** | Chunk count is called “topics.” `api.py:157`; `Journey.tsx:101` | An implementation detail is presented as educational understanding | Remove the count or label it honestly; introduce actual topic counts only when extracted |
| **Medium** | “AI Insight” is fixed score-threshold copy. `Result.tsx:149` | Generic encouragement looks like a personalized analysis | Rename it “Next step,” or derive a specific recommendation from saved evidence |
| **Medium** | “Adaptive practice session” links to generic configuration. `dashboard.py:53`; `Dashboard.tsx:27` | The interface promises adaptation that does not happen | Use “New practice session” until settings are automatically selected from performance |
| **Medium** | Sidebar XP/streak default to zero on screens that do not supply them. `AppShell.tsx:30`; `Journey.tsx:43`; `MakeQuiz.tsx:120` | Progress appears to change between pages | Supply one shared shell payload consistently |
| **Medium** | Weekday streak cells are colored using `i < streak`. `Dashboard.tsx:122` | The graphic does not represent actual activity dates | Build it from dated activity, or display only the streak number |
| **Medium** | Badge definitions disagree: Progress “Sharpshooter” uses overall 80% accuracy; achievements API uses a best score of 90%. `Progress.tsx:53`; `api.py:248` | Achievements seem inconsistent | Define badges once on the backend |
| **Medium** | Examly branding changes to “MCQ Generator” at login and results. `templates/login.html:7,36`; `Result.tsx:102` | The application feels assembled from separate products | Standardize brand, page titles, and terminology |
| **Medium** | Mobile drawer lacks focus management, Escape handling, and inactive-background treatment. `AppShell.tsx:30` | Keyboard focus can escape or reach translated off-screen navigation | Implement an accessible drawer and `aria-current` navigation |
| **Medium** | Three fixed result-stat columns and several horizontal action groups warrant narrow-screen testing. `Result.tsx:119,201` | Long labels and controls may become cramped | Verify at 320/375/390px; stack where needed |
| **Low** | Reduced-motion CSS exists, but Framer Motion and JavaScript score/confetti animation lack a corresponding policy. `index.css:37`; `Result.tsx:24,68` | Motion preference support is incomplete | Use Framer’s reduced-motion support and disable decorative animation appropriately |

**Measured contrast from source tokens**

Using the standard contrast calculation:

- `#71717a` on card background `#131316`: **3.84:1**
- White on primary blue `#3b82f6`: **3.68:1**

Both fall below the **4.5:1 normal-text threshold** for applicable text. These are token calculations, not a full rendered accessibility certification. Sources: `frontend/tailwind.config.js:17`, `Button.tsx:15`; [W3C contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).

**What currently creates the “student project” impression**

The strongest signals are inconsistent claims and disconnected screens: simulated processing, generic “AI” commentary, changing stats, duplicate terminology, and results that end with another upload. More gradients or animation would not resolve those problems.

---

**4. AI engineering audit — score: 5/10**

There is a useful separation between providers, retrieval, generation, validation, and persistence. However, the system currently lacks a consistent accepted-question contract, dependable source provenance, and convincing quality evaluation.

**Critical security finding: unsafe OAuth linking**

[Google linking at `google_auth.py:186`](C:/Users/Dinu/Documents/mcq_generator/.development/fastapi_app/routers/google_auth.py:186) and GitHub linking at `github_auth.py:174` search by provider ID, then by email.

However, `/profile` permits a local user to set an email without verification (`auth.py:148`). Linking retains the existing password.

This creates an account pre-hijacking path: someone can prepare a password account using another person’s email; that person’s subsequent OAuth login can attach their verified provider identity to the prepared account.

The isolated check confirmed email linking and password retention. **I did not test or exploit a live account.**

**Fix:** authenticate by provider subject ID. Linking an existing account should require authenticated, explicit proof of both identities. A provider-verified email alone does not verify the ownership of an existing local account’s email field.

**Additional access-boundary concern**

`student_login()` accepts any existing session key without checking session ownership (`student.py:258`). Keys contain only eight hexadecimal characters (`session_repo.py:17`), and the join route has no attempt throttling.

That is an implicit sharing mechanism. For a private study product, require ownership. If sharing is intentional, introduce an explicit sharing state and stronger access tokens.

**Pipeline findings**

| Area | Verified behavior / weakness | Recommended improvement |
|---|---|---|
| **Document extraction** | Server parsers handle PDF text/tables, DOCX paragraphs/tables, PPTX text/tables/notes/groups. No OCR. Browser PDF extraction simply concatenates text items. `pdf_processor.py:7`; `extract.ts:33` | Show extraction preview and warn about image-only or sparse text. Add OCR only if scanned documents prove important to users |
| **Extraction consistency** | Browser and server paths lose different structure; PDF table text may be duplicated by text plus table extraction | Build a small document-fixture set and compare extracted content before adding more parsing machinery |
| **Input limits** | Server files have size/magic/page checks. Client-extracted text bypasses those file checks and has no explicit extracted-character limit beyond request-body protection. `documents.py:76`; `middleware.py:79` | Bound text length and processing workload separately; bound archive expansion for Office files |
| **Chunking** | Defaults are 1,500 characters with 80 overlap. `clean_text()` first collapses all whitespace, removing paragraph boundaries. `chunker.py:69`; `text_cleaner.py:5` | Preserve meaningful paragraph/page boundaries before chunking; retain source metadata |
| **Embeddings** | Hosted dependency set excludes SentenceTransformers; auto mode therefore normally uses lexical feature hashing. `embeddings/registry.py:43`; `requirements.txt:25` | Describe hashing honestly. Compare it against the existing remote backend on retrieval fixtures before changing defaults |
| **Embedding correctness** | Cache lookup and reuse use only content hash. Model/dimension are stored but not used to isolate lookup or search. `embedding.py:73`; `sqlite_numpy.py:64,78` | Key vectors by content hash **and embedding identity**; require a controlled reindex when configuration changes |
| **Vector search** | NumPy loads all vectors for a document and ranks them in Python. This is a sensible small-scale baseline, without demonstrated large-scale performance | Keep it initially; measure document-size and concurrency limits before introducing another store |
| **MMR** | MMR is real, but candidate text is re-embedded individually during retrieval. `retriever.py:67` | Return/reuse stored candidate vectors; avoid extra remote calls |
| **Document spread** | `spread=True` only sorts selected hits by chunk index. It does not select across sections. `retriever.py:72` | Reserve selection slots across document sections, then rank within those sections |
| **Context coverage** | Context builder sorts by document position and stops at its character budget. Temporary uploads take a 12,000-character prefix. `context_builder.py:16`; `mcq_pipeline.py:189` | Allocate context across selected sections; make any partial-source use visible |
| **Prompt construction** | RAG prompts specify source-only answers, plausible distractors, explanations, and insufficiency. Legacy prompts are weaker. `prompts/mcq.py:26,68` | Use one grounding policy across workflows and keep uploaded text explicitly treated as untrusted data |
| **Structured output** | JSON mode guarantees neither field types nor MCQ semantics. Legacy validation checks presence and option count, not answer membership. `mcq_generator.py:25` | Validate every accepted question with one shared typed contract |
| **Fact verification** | Exceptions, missing verdicts, and malformed confidence values can result in acceptance. `bool("false")` also evaluates to true. `fact_verification.py:49,65` | Strictly validate verdicts. Record `verified`, `unverified`, or `rejected`; never silently turn a failed check into verified output |
| **Difficulty** | Default legacy mode relies on the prompt. RAG grading considers question stems, excluding options/context, and fails open. `difficulty.py:25` | Evaluate stem, options, answer, and reasoning demand together; use human-labeled examples |
| **Duplicates** | Deduplication normalizes case/outer whitespace only. Top-up prompts do not exclude previously accepted questions. `mcq_generator.py:114`; `mcq_pipeline.py:112` | Pass accepted stems into top-up exclusions; use inexpensive normalized similarity checks |
| **Explanation grounding** | Submission calls `explain_answers(details)` without `document_id`. Generated RAG explanations/source hints are not carried into result details. `student.py:359,424` | Reuse stored grounded explanations and provenance; retrieve source context when generating additional feedback |
| **Explanation usefulness** | Correct and unanswered questions generally receive answer restatements. `explanation.py:28,74` | Preserve a short “why” explanation for every question, with misconception feedback when appropriate |
| **Retry behavior** | HTTP 429 retries use exponential waits, but cap provider `Retry-After` at eight seconds. No overall remaining-time budget is propagated. `llm/base.py:66` | Respect long retry windows by returning a retryable status when the request budget cannot accommodate them |
| **Latency bounds** | Vercel config specifies 60 seconds; calls default to 45 seconds; a RAG iteration can make several sequential calls. Deadline checks occur between iterations. Legacy top-ups have no pipeline deadline. `vercel.json:7`; `mcq_pipeline.py:75`; `mcq_generator.py:91` | Enforce one end-to-end deadline across retrieval, generation, verification, retries, and fallback |
| **Concurrency** | `_legacy_fallback()` temporarily changes global `config.AI_PIPELINE`. `mcq_pipeline.py:276` | Call a dedicated legacy function directly; configuration must remain immutable during requests |
| **Submission availability** | Async `/submit` directly calls blocking LLM functions before returning the result redirect. `student.py:420` | Run blocking work safely outside the event loop; persist scores first and let optional enrichment complete independently |
| **Rate limiting** | Throttle counts successful quiz sessions, so simultaneous requests and repeated failed requests can bypass its intended cost limit. `rate_limits.py:32` | Atomically reserve generation attempts before spending provider calls; include in-flight work |
| **Background jobs** | Optional RQ support exists, but execution has no atomic claim preventing duplicate deliveries from generating multiple sessions. Queue failure can fall back inline with a large job budget. `job_runner.py:86` | Add a claimed state and idempotent result; never inherit worker-sized budgets inside a serverless request |
| **Observability** | Request IDs, stage durations, usage metadata, and structured logs exist. Some agents still print free-form messages and raw exception text. `observability.py:119`; `agents/base.py:8` | Standardize error codes, redact diagnostic content, and record chosen pipeline/fallback/verification status |
| **Privacy lifecycle** | Account/resource deletion does not call original-file deletion; optional pgvector text storage is not cleared by document deletion. Export omits documents and detailed answers. `auth.py:191`; `document_repo.py:116` | Define retention explicitly, track stored-object keys, and make deletion/export match their UI promises |

FastAPI automatically places normal `def` route handlers in its thread pool, but directly called blocking functions inside `async def` do not gain that protection. That supports the submission finding above. [Official FastAPI concurrency guidance](https://fastapi.tiangolo.com/async/).

**Evaluation is currently insufficient**

The committed evaluation report is explicitly **stub mode**, covering six cases. It is not a model-quality benchmark.

Additionally, `score_case()`:

- Treats answer-text occurrence in a document as “grounded.”
- Does not include that grounding metric in its pass condition.
- Does not establish distractor correctness, ambiguity, difficulty, or section coverage.

Evidence: [run_eval.py:108](C:/Users/Dinu/Documents/mcq_generator/.development/evals/run_eval.py:108), `evals/results/latest.md:4`.

A separate safety issue: the harness changes `DB_PATH` to a temporary location but does not clear/isolate `DATABASE_URL`. With PostgreSQL configured, its “throwaway database” assumption does not hold (`run_eval.py:293`; `db.py:23`).

**Do not run live evaluations until database isolation is explicit.**

---

**5. Product-value assessment**

**What problem does Examly solve?**

It reduces the effort of turning a student’s own materials into repeatable self-assessment, then preserving mistakes for later practice.

**Ideal initial user**

A college student preparing for a specific subject examination using lecture notes, textbook chapters, and slides.

The current goal selector also includes interviews, placements, certifications, competitive exams, and personal learning. The workflow is too generic to substantiate strong specialization across all those categories.

**What differentiates it from asking ChatGPT for questions?**

The credible advantages are:

- Reusable saved sources.
- Structured quiz configuration.
- Timed attempts with server-side scoring.
- Saved results and explanations.
- Retrieval over indexed material.
- Reattempting actual missed questions.

These advantages need reliable source coverage and continuity to become compelling.

**Most valuable existing feature**

The strongest product unit is **saved source → repeatable quiz → retained mistakes**. It gives users a reason to return without repeatedly uploading and prompting.

**Is weak-topic revision genuinely useful?**

Yes, as a **mistake-practice feature**. It reconstructs actual missed MCQs without another generation call.

It is not yet a dependable mastery system:

- Topic labels can change between attempts.
- Topics are not scoped to a subject/source.
- Cumulative miss rates react slowly to improvement.
- Historical mistakes remain eligible after successful reviews.
- Correct answers can reflect memorizing repeated options.
- Confidence is collected but does not drive revision priority.
- There are no due dates or delayed retention checks.

See [learning_repo.py:63](C:/Users/Dinu/Documents/mcq_generator/.development/core/repositories/learning_repo.py:63), `evaluation.py:65`, `learning_repo.py:120`.

**Does it improve knowledge over time?**

It supports practice and repetition. The current implementation does not provide evidence that its analytics measure knowledge improvement rather than repeated exposure.

A reasonable next model is simple: stable topic IDs, recent performance, successful-review state, and a next-review date. You do not need a sophisticated adaptive-learning algorithm.

**Positioning recommendation**

| Position | Assessment |
|---|---|
| **A. AI MCQ Generator** | Accurate but understates saved results and mistake review; weak differentiation |
| **B. AI-Powered Study Assistant** | **Strongest current positioning**, if narrowed to source-based quiz practice and revision |
| **C. Personalized Learning and Assessment Platform** | A future direction; presently implies stronger adaptation and validated mastery than implemented |

Suggested positioning:

> **Examly turns your study materials into practice quizzes and helps you revisit what you missed.**

**Remove or simplify:** duplicated terminology, unsupported “adaptive” claims, simulated technical progress, redundant achievement presentations, and unused feature claims.

**Improve:** source fidelity, grounded explanations, recent-result navigation, mistake recovery, error states, and accessibility.

**Add selectively:** source evidence, issue reporting for a question, and scheduled mistake review. Defer chat, additional content modes, more agents, and infrastructure expansion.

---

**6. The ten highest-impact improvements, ranked**

Effort estimates are engineering working days, including focused validation. They assume one developer familiar with the repository; they are not commitments.

| Rank / phase | Improvement and user benefit | Complexity / effort | Modules, dependencies, and risks | Verification |
|---|---|---|---|---|
| **1 — P0** | **Fix identity linking and quiz access.** Prevent account pre-hijacking and unintended access to private quizzes | Medium; **1–2 days** | OAuth routers, `auth.py`, `student.py`, `session_repo.py`. Existing linked accounts require careful review; preserve intentional sharing only if explicitly supported | Reproduce unverified-email linking locally and require rejection; test legitimate linking, owner access, stranger access, and legacy sessions |
| **2 — P0** | **Bound generation and submission work.** Prevent indefinite waiting and avoid one slow AI request blocking unrelated requests | High; **2–4 days** | `llm/base.py`, `mcq_generator.py`, `mcq_pipeline.py`, `student.py`, `rate_limits.py`, job runner, MakeQuiz. No new queue required for the first fix | Fake slow providers/429s; verify deadline termination, one in-flight reservation, responsive health checks, and durable results |
| **3 — P0/P1** | **Use one accepted-question contract and preserve source evidence.** Prevent malformed quizzes and unsupported explanations | Medium–high; **2–4 days** | QA agent, legacy generator, verifier, prompts, session/result payloads. Share existing validation; use installed Pydantic. Strict rejection may lower fill rates | Invalid answer/options/types rejected in every path; malformed verdicts remain unverified; results retain supporting source/explanation |
| **4 — P1** | **Fix whole-document coverage.** Make questions reflect the uploaded material beyond its opening section | Medium; **1–3 days** | `extract.ts`, chunker, retriever, context builder, temporary generation. Larger documents must remain within the same budget | Documents with known beginning/middle/end facts; check selected chunks and generated coverage under fixed budgets |
| **5 — P1** | **Version embedding storage and reuse vectors during MMR.** Make retrieval stable across deployments and avoid unnecessary embedding calls | Medium; **1–2 days** | Embedding agent, store queries, schema migrations, ingestion status. Existing vectors need controlled reindexing | Switch models/dimensions; ensure old vectors cannot silently mix; count calls during retrieval |
| **6 — P0 gate/P1** | **Establish a safe release/data lifecycle.** Preserve existing users and make deletion/export truthful | Medium–high; **2–4 days** | CI, migrations, ownership mapping script, DB configuration, export/delete routes, storage metadata, evaluation harness. Depends on production DB/storage access during implementation | Migration rehearsal on a copy; restart persistence test; complete delete/export check; isolated PostgreSQL CI and evaluation DB |
| **7 — P1** | **Make mistake review recognize recovery.** Show users what improved and what needs another attempt | Medium; **2–4 days** | Learning repository, evaluation agent, question/topic metadata, review endpoint, WeakTopics/Progress | Miss → review correctly → priority decreases; topic stays stable; later due review appears without erasing history |
| **8 — P1** | **Make processing and errors honest.** Distinguish actual progress, empty data, unavailable AI, and failure | Low–medium; **1–2 days** | Upload/AddResource, MakeQuiz, shared API helpers, Journey, WeakTopics. Avoid new status infrastructure unless needed | Exercise extraction failure, expired upload, 401/422/429/502, partial generation, missing job, and network loss |
| **9 — P1/P2** | **Connect the study journey and standardize terminology.** Make it easy to reopen results and continue the same source | Low–medium; **1–2 days** | AppShell, Dashboard, Result, templates, shared shell data. Keep existing route aliases | Follow source → quiz → result → missed review → same-source practice without reuploading; stats remain consistent |
| **10 — P1/P2** | **Complete accessibility and responsive QA.** Make core study tasks usable by keyboard, screen reader, and phone | Medium; **1–3 days** | Upload controls, Quiz, MakeQuiz segments, drawer, tokens, ProgressBar, Result motion/layout | Keyboard-only complete flow; selected-state announcements; contrast checks; reduced-motion behavior; 320–1440px visual checks |

For #3, implement the shared deterministic validator first. Source-evidence presentation can follow once provenance is stored correctly.

---

**7. Realistic implementation roadmap**

**P0 — Release blockers**

Complete before inviting public users:

- Fix unsafe OAuth linking.
- Decide and enforce private versus shared quiz access.
- Enforce shared MCQ validity across all generation paths.
- Remove request-time global pipeline mutation.
- Bound provider calls, retries, top-ups, and fallbacks.
- Move blocking enrichment out of the async submission execution path.
- Rehearse ownership migration before deploying `006` against existing data.
- Verify durable storage configuration; do not assume `/tmp` persistence.
- Isolate the evaluation harness from any production database.

The ownership migration intentionally leaves old records unassigned. That is safer than guessing ownership from usernames, but existing resources can disappear from users’ accessible Libraries until reviewed mappings are applied. Use the existing mapping script against a database copy first.

**P1 — Reliability and learning value**

After the release blockers:

- Correct section coverage and embedding versioning.
- Preserve grounded explanations.
- Introduce stable source/topic/question identities.
- Track successful reviews and review dates.
- Add truthful progress, actionable errors, and partial-generation notices.
- Link saved results and preserve selected-source context.
- Run PostgreSQL and frontend/browser checks in CI.
- Build a small human-reviewed question-quality benchmark.

**P2 — Polish**

Once the workflow is dependable:

- Consolidate branding, badges, and page terminology.
- Refine typography and information density after actual visual review.
- Simplify chart labels and replace overstated “mastery” language.
- Tune motion and small interaction feedback.
- Add optional theme support only if users need it.

**Planning range:** approximately **3–5 full-time engineering weeks** for the complete list, with overlap between related changes. Allow **5–8 calendar weeks** if you are learning while implementing part-time.

Do not make a visual redesign the first milestone. The first milestone should be: **a learner can trust their account, quiz, saved result, and source material.**

---

**8. Project-specific AI engineering learning roadmap**

Use the real application as your laboratory. Before asking an assistant to implement a change, write down the expected behavior and failure cases yourself.

**1. FastAPI dependencies and request lifecycle**

- **Meaning:** reusable request requirements, such as authenticated identity and database cleanup.
- **Where/functions:** `fastapi_app/deps.py:41`, `_session_user()`, `require_user()`, `require_api_user()`, `db()`.
- **Why Examly needs it:** consistent access checks and reliable connection closure.
- **Experiment:** trace an anonymous request to `/dashboard` and `/api/dashboard`; explain why their responses differ.
- **Interview question:** “What does a dependency that uses `yield` guarantee when a route raises an exception?”

**2. Blocking I/O, async execution, and time budgets**

- **Meaning:** an async function only lets other work proceed when its operations actually yield control.
- **Where/functions:** `practice_generate()` at `api.py:291`, `submit()` at `student.py:348`, `LLMProvider.complete()` at `llm/base.py:52`.
- **Why:** Examly waits on external providers, databases, and parsing.
- **Experiment:** substitute a slow fake provider, submit a quiz, and request `/health` concurrently. Compare before and after safely offloading the blocking work.
- **Interview question:** “Why can `requests.post()` inside an async route block the application?”

**3. Transactions, atomic claims, and idempotency**

- **Meaning:** a logical operation should commit completely, and repeating it should not duplicate its effects.
- **Where/functions:** submission’s conditional attempt claim at `student.py:401`; `run_generation_job()` at `job_runner.py:86`.
- **Why:** double clicks, multiple tabs, network retries, and duplicate job delivery are normal.
- **Experiment:** send simultaneous submissions for one attempt; inspect result count and XP changes.
- **Interview question:** “Why is checking whether a result exists before inserting it insufficient under concurrency?”

**4. Authentication versus authorization**

- **Meaning:** authentication identifies a person; authorization determines which resources they may access.
- **Where/functions:** `_session_user()`, `practice_generate()`, `student_login()`, `_upsert_google_user()`.
- **Why:** being logged in should not imply permission to read another user’s quiz.
- **Experiment:** build an access matrix for two users covering documents, jobs, sessions, and results.
- **Interview question:** “Why is a verified OAuth email not sufficient proof for linking an existing password account?”

**5. Extraction and chunking**

- **Meaning:** turning a document into searchable units while preserving enough structure to explain where facts came from.
- **Where/functions:** `extract_upload_text()` at `documents.py:61`, `clean_text()` at `text_cleaner.py:5`, `chunk_text()` at `chunker.py:69`.
- **Why:** retrieval quality cannot exceed the quality of extracted text.
- **Experiment:** inspect chunks from a text PDF, table-heavy PDF, slide deck, and scanned PDF. Compare sizes 500, 1,000, and 1,500.
- **Interview question:** “How does removing paragraph boundaries before chunking change retrieval quality?”

**6. Embeddings and similarity**

- **Meaning:** representing text numerically so related content can be ranked.
- **Where/functions:** `HashingEmbedder.embed_batch()`, `RemoteAPIEmbedder.embed_batch()`, `build_embedder()`.
- **Why:** Examly must find relevant passages without sending all material into every prompt.
- **Experiment:** compare queries using exact terms and synonyms against the same source using hashing and the existing remote backend.
- **Interview question:** “Why can feature hashing find word overlap yet miss a strong semantic match?”

**7. Vector retrieval and MMR**

- **Meaning:** similarity finds relevant passages; MMR balances relevance with diversity.
- **Where/functions:** `SqliteNumpyStore.query()` at `sqlite_numpy.py:75`, `mmr_select()` at `retriever.py:25`, `Retriever.retrieve()` at `:55`.
- **Why:** a quiz should not repeatedly test adjacent versions of one fact.
- **Experiment:** construct a document containing ten similar passages and several distinct sections; compare ordinary top-k, MMR, and explicit section allocation.
- **Interview question:** “Why does sorting selected chunks by position fail to guarantee document coverage?”

**8. Prompts, context budgets, and grounding**

- **Meaning:** deciding which evidence the model receives and what it may conclude from that evidence.
- **Where/functions:** `ContextBuilderAgent.run()` at `context_builder.py:16`, `rag_prompt()` at `prompts/mcq.py:68`, `generate_from_text()`.
- **Why:** missing evidence cannot be repaired merely by stronger instructions.
- **Experiment:** put a target fact near the end of a long document and trace whether each upload path places it in the final prompt.
- **Interview question:** “Can a perfectly written grounding prompt compensate for retrieving the wrong passages?”

**9. Structured outputs and validation**

- **Meaning:** parseable JSON is only the first level; field types and domain rules must also be correct.
- **Where/functions:** `complete_json()`, `_parse_json()`, `_structurally_valid()`, `QualityAssuranceAgent._validate_one()`, `FactVerificationAgent.run()`.
- **Why:** scoring relies on four distinct options and an answer matching exactly one option.
- **Experiment:** feed missing answers, duplicate options, numbers instead of strings, `"false"`, omitted verdicts, and truncated JSON into the validators.
- **Interview question:** “What is the difference between JSON validity, schema validity, and factual correctness?”

**10. Evaluation and experimentation**

- **Meaning:** measuring a change against a fixed set of representative cases.
- **Where/functions:** `score_case()` at `evals/run_eval.py:108`, `generate_live()`, `generate_stub()`.
- **Why:** a successful response or attractive demo is not evidence of good questions.
- **Experiment:** manually label 30–50 questions for support, answer correctness, distractor quality, ambiguity, difficulty, and coverage. Change one variable at a time.
- **Interview question:** “Why does finding an answer phrase in the source fail to prove a question is grounded?”

**11. Learning analytics**

- **Meaning:** translating attempts into evidence of understanding and deciding what to practice next.
- **Where/functions:** `EvaluationAgent.run()`, `_label_topics()`, `upsert_weak_topics()`, `missed_questions_for_topic()`, `top_weak_topics()`.
- **Why:** Examly’s return value depends on useful revision, not just generating more content.
- **Experiment:** simulate one initial mistake followed by several correct reviews. Inspect whether displayed weakness reflects recovery.
- **Interview question:** “How would you distinguish remembering an answer from learning the underlying concept?”

**12. Deployment, migrations, and observability**

- **Meaning:** running the same product safely across fresh processes, schema versions, and partial failures.
- **Where/functions:** `create_app()`, `init_db()`, `run_sql_migrations()`, `get_db()`, `log_event()`, `timed()`.
- **Why:** serverless filesystems, provider outages, and concurrent starts expose assumptions hidden during local development.
- **Experiment:** on an isolated database, restart the application, change embedding configuration, and rehearse a migration. Trace one request using its request ID.
- **Interview question:** “Which state belongs in a signed cookie, relational database, temporary filesystem, and object store—and why?”

A useful learning sequence is **request lifecycle → transactions/security → extraction/retrieval → validation/evaluation → learning analytics → deployment**. It follows the dependencies in the actual product.

---

**9. The three improvements to implement first**

1. **Close the authentication and quiz-access gaps.**
   Replace automatic email linking with explicit identity linking, define quiz privacy, and add focused access-control regressions.

2. **Make AI work bounded and results reliably available.**
   Introduce an end-to-end time budget, remove global configuration mutation, reserve in-flight generation, and prevent optional explanation work from blocking the result workflow.

3. **Accept only valid questions and preserve their evidence.**
   Share deterministic MCQ validation across both pipelines, strictly parse verification verdicts, and retain grounded explanations/source references for results.

These changes strengthen public-release readiness and give you substantial, defensible AI engineering work to explain in interviews.

**Audit complete. No code changed; implementation is awaiting your approval.**
