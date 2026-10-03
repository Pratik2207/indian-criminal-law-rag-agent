"""Ingest the statute PDFs into Qdrant.

Pipeline: PDF -> sections (lawrag.parser) -> section-aware chunks with a context
header -> dense (BGE) + sparse (BM25) vectors -> Qdrant, with deterministic
point IDs so re-running never duplicates data.

    python -m lawrag.ingest            # create/update the collection
    python -m lawrag.ingest --rebuild  # drop and rebuild from scratch
"""
import argparse
import logging
import re
import uuid
from collections import Counter

from qdrant_client import models

from lawrag.config import get_settings
from lawrag.embeddings import embed_documents, sparse_documents
from lawrag.parser import Section, parse_all
from lawrag.store import get_client

log = logging.getLogger(__name__)

MAX_CHARS = 1500  # ~350 tokens: well inside bge-small's 512-token window, header included
SUBSECTION_RE = re.compile(r"\n(?=\(\d+[A-Z]?\)\s)")


def section_header(s: Section) -> str:
    title = f": {s.title}" if s.title else ""
    chapter = f" ({s.chapter})" if s.chapter else ""
    return f"{s.act_name} — Section {s.section_no}{title}{chapter}"


def split_section(text: str) -> list[str]:
    """One chunk per section; long sections split at sub-section boundaries, then by lines."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts = SUBSECTION_RE.split(text)
    units = []
    for p in parts:
        if len(p) <= MAX_CHARS:
            units.append(p)
        else:  # schedules / very long sub-sections: window over lines
            buf = ""
            for line in p.split("\n"):
                if len(buf) + len(line) > MAX_CHARS and buf:
                    units.append(buf)
                    buf = ""
                buf += line + "\n"
            if buf.strip():
                units.append(buf)
    chunks, buf = [], ""
    for u in units:
        if len(buf) + len(u) > MAX_CHARS and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += u + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def build_chunks(sections: list[Section]) -> list[dict]:
    chunks, seen = [], set()
    for s in sections:
        pieces = split_section(s.text)
        for i, body in enumerate(pieces):
            key = re.sub(r"\W", "", body.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            header = section_header(s) + (f" [part {i + 1}/{len(pieces)}]" if len(pieces) > 1 else "")
            chunks.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"indian-law:{s.act}:{s.section_no}:{i}")),
                "embed_text": f"{header}\n{body}",
                "payload": {
                    "act": s.act, "act_name": s.act_name, "section_no": s.section_no, "title": s.title,
                    "chapter": s.chapter, "page": s.page, "source": s.source, "act_url": s.act_url,
                    "chunk_index": i, "n_chunks": len(pieces), "header": header, "text": body,
                },
            })
    return chunks


def sanity_check(sections: list[Section]):
    """Catch broken PDF extraction (e.g. text with all spaces dropped) before indexing."""
    for act in sorted({s.act for s in sections}):
        words = " ".join(s.text for s in sections if s.act == act).split()
        avg = sum(map(len, words)) / max(len(words), 1)
        n = sum(1 for s in sections if s.act == act)
        log.info("%s: %d sections, avg word length %.1f", act, n, avg)
        if avg > 12 or n < 50:
            raise RuntimeError(f"Extraction for {act} looks broken (avg word len {avg:.1f}, {n} sections)")


def ensure_collection(client, name: str, rebuild: bool):
    s = get_settings()
    if rebuild and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata={"dense_model": s.dense_model, "sparse_model": s.sparse_model, "schema": 2},
        )
        for field in ("act", "section_no"):
            client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def ingest(rebuild: bool = False, batch_size: int = 64):
    s = get_settings()
    sections = parse_all(s.knowledge_dir)
    sanity_check(sections)
    chunks = build_chunks(sections)
    log.info("%d chunks from %d sections (%s)", len(chunks), len(sections),
             dict(Counter(c["payload"]["act"] for c in chunks)))

    client = get_client()
    ensure_collection(client, s.qdrant_collection_name, rebuild)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["embed_text"] for c in batch]
        dense = embed_documents(texts)
        sparse = sparse_documents(texts)
        client.upsert(s.qdrant_collection_name, points=[
            models.PointStruct(
                id=c["id"],
                vector={"dense": d, "bm25": models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())},
                payload=c["payload"],
            )
            for c, d, sp in zip(batch, dense, sparse)
        ])
        log.info("upserted %d/%d", min(i + batch_size, len(chunks)), len(chunks))
    count = client.count(s.qdrant_collection_name).count
    log.info("Ingestion complete: %d points in '%s'", count, s.qdrant_collection_name)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="drop and recreate the collection")
    ingest(rebuild=ap.parse_args().rebuild)
