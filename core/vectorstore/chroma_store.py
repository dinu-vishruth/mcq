"""
ChromaDB vector store. Default on capable hosts (chromadb is installed locally).

Persists to CHROMA_PATH. One Chroma collection per document keeps namespaces
isolated and deletes cheap. We pass precomputed embeddings (our Embedder owns
the model), so Chroma is used purely as an ANN index + metadata store.

On Vercel the read-only filesystem makes Chroma's persistence unreliable, so the
registry forces the SQLite store there regardless of VECTOR_STORE.
"""
from __future__ import annotations

import config
from core.vectorstore.base import VectorStore, SearchHit


class ChromaStore(VectorStore):
    def __init__(self, model_id: str, dim: int):
        import chromadb  # lazy import
        self.model_id = model_id
        self.dim = dim
        self._client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    def _collection(self, document_id: int):
        # cosine space; vectors are already normalised but this is explicit.
        return self._client.get_or_create_collection(
            name=f"doc_{document_id}", metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, document_id: int, items: list[dict]) -> None:
        if not items:
            return
        col = self._collection(document_id)
        col.upsert(
            ids=[f"{document_id}:{it['chunk_index']}" for it in items],
            embeddings=[it["vector"] for it in items],
            documents=[it["content"] for it in items],
            metadatas=[{"document_id": document_id, "chunk_index": it["chunk_index"]} for it in items],
        )

    def query(self, document_id: int, query_vector, top_k: int) -> list[SearchHit]:
        col = self._collection(document_id)
        n = col.count()
        if n == 0:
            return []
        res = col.query(query_embeddings=[query_vector], n_results=min(top_k, n))
        hits = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, _id in enumerate(ids):
            meta = metas[i] or {}
            # Chroma cosine distance = 1 - cosine_similarity.
            score = 1.0 - float(dists[i]) if i < len(dists) else 0.0
            hits.append(SearchHit(
                chunk_id=_id,
                document_id=int(meta.get("document_id", document_id)),
                chunk_index=int(meta.get("chunk_index", i)),
                content=docs[i] if i < len(docs) else "",
                score=score,
            ))
        return hits

    def has_document(self, document_id: int) -> bool:
        try:
            return self._collection(document_id).count() > 0
        except Exception:
            return False

    def delete_document(self, document_id: int) -> None:
        try:
            self._client.delete_collection(name=f"doc_{document_id}")
        except Exception:
            pass
