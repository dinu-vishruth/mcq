"""
Agentic AI + RAG package for the MCQ Generator (imported as `core` to avoid
colliding with the legacy Flask entry point in app.py).

Layout (built incrementally across migration phases):
    llm/          provider-agnostic LLM interface + adapters + registry
    prompts/      versioned prompt templates (never inline in business logic)
    embeddings/   Embedder interface + backends (ST local, remote API, hashing)
    vectorstore/  vector store interface + Chroma / SQLite-numpy / FAISS
    rag/          chunking, retrieval, context building
    agents/       independent single-responsibility agents
    services/     orchestration (ingestion, mcq pipeline, learning)
    repositories/ data access over the SQLite schema
    models/       db connection + additive migrations
    routes/       Flask blueprints (Phase 6)

Nothing here changes runtime behaviour until a feature flag in config is set.
"""
