"""Evaluation owns temporary storage and never inherits the application's DB URL."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import config
from core.embeddings import registry as embeddings
from core.vectorstore import registry as vectors


@contextmanager
def isolated_evaluation():
    with TemporaryDirectory(prefix="examly_eval_") as directory:
        root = Path(directory)
        overrides = {
            "DATABASE_URL": "", "DB_PATH": str(root / "eval.db"), "DATA_DIR": directory,
            "UPLOAD_FOLDER": str(root / "uploads"), "REPORTS_DIR": str(root / "reports"),
            "CHROMA_PATH": str(root / "chroma"), "EMBEDDING_BACKEND": "hashing",
            "VECTOR_STORE": "sqlite", "SEED_DEMO_USERS": False,
        }
        original = {key: getattr(config, key) for key in overrides}
        cached = embeddings._cached, vectors._cached
        try:
            for key, value in overrides.items():
                setattr(config, key, value)
            embeddings._cached = vectors._cached = None
            if config.DATABASE_URL or not Path(config.DB_PATH).is_relative_to(root):
                raise RuntimeError("Evaluation storage isolation failed")
            yield
        finally:
            for key, value in original.items():
                setattr(config, key, value)
            embeddings._cached, vectors._cached = cached
