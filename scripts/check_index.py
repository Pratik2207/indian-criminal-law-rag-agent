"""Print collection stats: point count, chunks per Act, embedding model."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lawrag.config import get_settings  # noqa: E402
from lawrag.store import get_client  # noqa: E402

if __name__ == "__main__":
    s, client = get_settings(), get_client()
    name = s.qdrant_collection_name
    if not client.collection_exists(name):
        raise SystemExit(f"Collection '{name}' not found. Run `python -m lawrag.ingest --rebuild`.")
    info = client.get_collection(name)
    acts, offset = Counter(), None
    while True:
        pts, offset = client.scroll(name, limit=1000, offset=offset, with_payload=["act"], with_vectors=False)
        acts.update(p.payload["act"] for p in pts)
        if offset is None:
            break
    print(f"Collection '{name}': {info.points_count} points {dict(acts)}")
    print(f"Metadata: {info.config.metadata}")
