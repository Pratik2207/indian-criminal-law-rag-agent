"""Retrieval pipeline: query analysis -> exact lookup / hybrid search -> rerank.

Modes (for ablations in evals/retrieval_eval.py):
  dense          dense BGE vectors only
  hybrid         dense + BM25 sparse, fused with Reciprocal Rank Fusion
  hybrid_rerank  hybrid candidates re-scored by a cross-encoder
  full           hybrid_rerank + query analysis (explicit section references are
                 looked up exactly, IPC<->BNS / CrPC<->BNSS equivalents are added,
                 Act mentions restrict the search, non-English queries are translated)
"""
from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from qdrant_client import models

from lawrag.config import ROOT, get_settings
from lawrag.embeddings import embed_query, rerank, sparse_query
from lawrag.store import check_collection, get_client

log = logging.getLogger(__name__)
SECTION_MAP = ROOT / "data" / "section_map.csv"

ACT_ALIASES = [
    (r"bharatiya\s+nagarik\s+suraksha\s+sanhita|\bbnss\b", "BNSS"),
    (r"bharatiya\s+nyaya\s+sanhita|\bbns\b", "BNS"),
    (r"indian\s+penal\s+code|\bipc\b", "IPC"),
    (r"code\s+of\s+criminal\s+procedure|\bcr\.?\s?p\.?\s?c\.?(?![a-z])", "CrPC"),
]
# an "Illustration(s)." block runs until the next sub-section "(2)", Explanation or Exception
ILLUSTRATIONS_RE = re.compile(r"(?ms)^Illustrations?\.\s*$.*?(?=^\(\d+\)\s|^Explanation|^Exception|\Z)")
NUM_RE = re.compile(r"\b(?:section|sec\.?|s\.|u/s\.?)?\s*(\d{1,3}[A-Z]{0,2})\b", re.I)


@dataclass
class Hit:
    act: str
    section_no: str
    title: str
    chapter: str
    page: int
    source: str
    act_url: str
    header: str
    text: str
    score: float
    match: str = "semantic"  # semantic | exact | mapped

    @property
    def cite(self) -> str:
        return f"{self.act} s.{self.section_no}"


@dataclass
class QueryAnalysis:
    query: str  # possibly translated
    acts: list[str] = field(default_factory=list)  # Acts mentioned
    refs: list[tuple[str, str]] = field(default_factory=list)  # explicit (act, section) references
    mapped: list[tuple[str, str]] = field(default_factory=list)  # equivalents in the paired code
    semantic_query: str = ""  # query with section numbers removed (they only add lexical noise)

    @property
    def correspondences(self) -> list[str]:
        out = []
        for ref in self.refs:
            eqs = [e for e in equivalents(*ref)]
            if eqs:
                out.append(f"{ref[0]} s.{ref[1]} corresponds to " + ", ".join(f"{a} s.{n}" for a, n in eqs))
        return out


@lru_cache
def section_map() -> dict[tuple[str, str], list[tuple[str, str]]]:
    # Fail loudly: a missing map silently disables IPC<->BNS mapping, correspondence notes and checks.
    if not SECTION_MAP.exists():
        raise FileNotFoundError(f"{SECTION_MAP} not found; run `python scripts/build_section_map.py`.")
    m = defaultdict(list)
    with SECTION_MAP.open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            new, old = (r["new_act"], r["new_section"]), (r["old_act"], r["old_section"])
            m[new].append(old)
            m[old].append(new)
    return dict(m)


def equivalents(act: str, section_no: str) -> list[tuple[str, str]]:
    return section_map().get((act, section_no), [])


def _translate(query: str) -> str:
    """Translate non-English (e.g. Hindi) queries to English with the local LLM."""
    from lawrag.llm import complete

    out = complete(
        "Translate this Indian legal question into English. Output only the translation.\n\n" + query,
        max_tokens=100,
    )
    return out.strip() or query


def analyze(query: str, translate: bool = True) -> QueryAnalysis:
    q = query
    if translate and re.search(r"[ऀ-ॿ]", q):
        try:
            q = _translate(q)
        except Exception as e:  # LLM down: fall back to the raw query
            log.warning("translation failed: %s", e)
    mentions = []  # (pos, act)
    for pat, act in ACT_ALIASES:
        for m in re.finditer(pat, q, re.I):
            if not any(abs(m.start() - p) < 3 for p, _ in mentions):
                mentions.append((m.start(), act))
    qa = QueryAnalysis(query=q, acts=sorted({a for _, a in mentions}))
    if mentions:
        for m in NUM_RE.finditer(q):
            num = m.group(1).upper()
            if not num[0].isdigit() or re.match(r"\s*(years?|days?|months?|hours?|rupees|persons)\b",
                                                q[m.end():], re.I):
                continue
            pos = m.start(1)
            nearest = min(mentions, key=lambda pa: abs(pa[0] - pos))
            # "Section 103 of the BNS": keyword, may be further away; "IPC 302" / "302 IPC": must be adjacent
            has_kw = bool(re.match(r"(section|sec|s\.|u/s)", m.group(0).strip(), re.I))
            if abs(nearest[0] - pos) <= (40 if has_kw else 8):
                ref = (nearest[1], num)
                if ref not in qa.refs:
                    qa.refs.append(ref)
    for ref in qa.refs:
        for eq in equivalents(*ref):
            if eq not in qa.refs and eq not in qa.mapped:
                qa.mapped.append(eq)
    # "IPC 302" makes BM25 match unrelated sections numbered 302 in other Acts; drop the numbers
    sem = q
    if qa.refs:
        sem = re.sub(r"\b(?:section|sec\.?|s\.|u/s\.?)?\s*\d{1,3}[A-Z]{0,2}\b", " ", q, flags=re.I)
        sem = re.sub(r"\s+", " ", sem).strip()
    qa.semantic_query = sem if len(sem.split()) >= 2 else q
    return qa


class Retriever:
    def __init__(self, mode: str | None = None):
        s = get_settings()
        self.mode = mode or s.retrieval_mode
        self.collection = s.qdrant_collection_name
        self.client = get_client()
        check_collection(self.client, self.collection)
        self.candidate_k = s.candidate_k

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _to_hit(p, score: float, match: str = "semantic") -> Hit:
        pl = p.payload
        return Hit(pl["act"], pl["section_no"], pl.get("title", ""), pl.get("chapter", ""), pl.get("page", 0),
                   pl.get("source", ""), pl.get("act_url", ""), pl.get("header", ""), pl["text"], score, match)

    @staticmethod
    def _filter(acts: list[str]) -> models.Filter:
        # The Second Schedules are blank forms: lots of legal vocabulary, little substance, so they
        # crowd out real sections in semantic search. They stay reachable via exact lookup.
        must_not = [models.FieldCondition(key="section_no", match=models.MatchValue(value="Schedule II"))]
        must = [models.FieldCondition(key="act", match=models.MatchAny(any=acts))] if acts else None
        return models.Filter(must=must, must_not=must_not)

    def lookup(self, act: str, section_no: str, match: str = "exact") -> list[Hit]:
        pts, _ = self.client.scroll(
            self.collection, limit=50, with_payload=True,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="act", match=models.MatchValue(value=act)),
                models.FieldCondition(key="section_no", match=models.MatchValue(value=section_no)),
            ]),
        )
        pts.sort(key=lambda p: p.payload.get("chunk_index", 0))
        return [self._to_hit(p, 100.0, match) for p in pts]

    def expand(self, hits: list[Hit], max_chars: int = 4000) -> list[Hit]:
        """Small-to-big: chunks are matched, but the LLM gets the whole section (all parts, capped),
        so e.g. a punishment in sub-section (2) isn't lost when sub-section (1) matched."""
        for h in hits:
            if h.section_no.startswith("Schedule"):  # schedules are huge tables; keep the matched part
                continue
            parts = self.lookup(h.act, h.section_no)
            if len(parts) > 1:
                full = "\n".join(p.text for p in parts)
                if len(full) > max_chars:  # drop examples before cutting operative sub-sections
                    full = ILLUSTRATIONS_RE.sub("[illustrations omitted]\n", full)
                h.text = full if len(full) <= max_chars else full[:max_chars] + "\n[…section continues]"
        return hits

    def _candidates(self, query: str, limit: int, flt=None):
        dense = embed_query(query)
        if self.mode == "dense":
            return self.client.query_points(self.collection, query=dense, using="dense", limit=limit,
                                            query_filter=flt, with_payload=True).points
        sp = sparse_query(query)
        sparse = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
        return self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=limit, filter=flt),
                models.Prefetch(query=sparse, using="bm25", limit=limit, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True,
        ).points

    # ------------------------------------------------------------------- main
    def search(self, query: str, k: int | None = None, analysis: QueryAnalysis | None = None) -> list[Hit]:
        k = k or get_settings().top_k
        exact: list[Hit] = []
        flt = None
        if self.mode == "full":
            analysis = analysis or analyze(query)
            query = analysis.semantic_query or analysis.query
            for ref in analysis.refs:
                exact += self.lookup(*ref, match="exact")[:1]
            for ref in analysis.mapped:
                exact += self.lookup(*ref, match="mapped")[:1]
            flt = self._filter(analysis.acts if not analysis.refs else [])

        pts = self._candidates(query, self.candidate_k if "rerank" in self.mode or self.mode == "full" else k * 3, flt)
        hits = [self._to_hit(p, p.score) for p in pts]
        if self.mode in ("hybrid_rerank", "full") and hits:
            # header + first 1000 chars is enough to judge relevance and keeps the cross-encoder fast
            scores = rerank(query, [f"{h.header}\n{h.text[:1000]}" for h in hits])
            for h, sc in zip(hits, scores):
                h.score = float(sc)
            hits.sort(key=lambda h: h.score, reverse=True)

        # one entry per section, exact/mapped matches first
        out, seen = [], set()
        for h in exact + hits:
            key = (h.act, h.section_no)
            if key not in seen:
                seen.add(key)
                out.append(h)
        return out[:k]
