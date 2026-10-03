"""Qdrant client factory (one shared client per process).

Embedded/local mode holds a file lock on the DB folder, so creating several
clients (e.g. on every Streamlit rerun) fails; always go through get_client().
Set QDRANT_URL to use a Qdrant server instead.
"""
import atexit
from functools import lru_cache

from qdrant_client import QdrantClient

from lawrag.config import get_settings


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    client = QdrantClient(url=s.qdrant_url) if s.qdrant_url else QdrantClient(path=s.qdrant_db_path)
    atexit.register(client.close)  # release the local-mode file lock cleanly
    return client


def check_collection(client: QdrantClient, name: str):
    """Fail fast if the collection was built with a different embedding model."""
    s = get_settings()
    if not client.collection_exists(name):
        raise RuntimeError(f"Collection '{name}' not found. Run `python -m lawrag.ingest --rebuild` first.")
    meta = client.get_collection(name).config.metadata or {}
    if meta.get("dense_model") not in (None, s.dense_model):
        raise RuntimeError(f"Collection built with {meta['dense_model']} but DENSE_MODEL is {s.dense_model}; "
                           "re-run `python -m lawrag.ingest --rebuild`.")
