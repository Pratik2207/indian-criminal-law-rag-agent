"""Retrieval quality evaluation.

Runs one or more retriever configurations over the golden query sets and
reports Hit@k, Recall@5, MRR@10, nDCG@10, duplicate rate and latency, overall
and broken down by query type and by Act.

Relevance is judged at the level of (act, section). For the legacy index, whose
chunks carry no section metadata, each retrieved chunk is mapped to the section
whose text contains it (whitespace/punctuation-insensitive match against the
parsed statutes).

    python -m evals.retrieval_eval --configs baseline fix_embed
    python -m evals.retrieval_eval --configs dense hybrid hybrid_rerank
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lawrag.config import get_settings  # noqa: E402
from lawrag.parser import AMEND_MARK_RE, parse_all  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
PAIRED = {"IPC": "BNS", "BNS": "IPC", "CrPC": "BNSS", "BNSS": "CrPC"}
SOURCE_ACT = {"BNS_2023.pdf": "BNS", "BNSS_2023.pdf": "BNSS", "IPC_1860.pdf": "IPC", "CrPC_1973.pdf": "CrPC"}


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", AMEND_MARK_RE.sub("", t).lower())


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z]", "", t.lower())


class SectionIndex:
    """Parsed statutes, used for labelling legacy chunks and lenient matching."""

    def __init__(self):
        self.sections = parse_all(get_settings().knowledge_dir)
        self.by_act = defaultdict(list)
        self.title = {}
        for s in self.sections:
            self.by_act[s.act].append((s.section_no, norm(s.text)))
            self.title[(s.act, s.section_no)] = norm_title(s.title)

    def label(self, source: str, text: str):
        act = SOURCE_ACT.get(source)
        n = norm(text)
        if not act or len(n) < 12:
            return None
        probes = [n[len(n) // 2 - 20: len(n) // 2 + 20]] if len(n) >= 40 else [n]
        if len(n) >= 120:
            probes += [n[10:50], n[-50:-10]]
        for probe in probes:
            for sec, st in self.by_act[act]:
                if probe in st:
                    return (act, sec)
        return None

    def equivalent(self, a: tuple, b: tuple) -> bool:
        """Same-titled section in the paired code (IPC<->BNS, CrPC<->BNSS)."""
        if a == b:
            return True
        if PAIRED.get(a[0]) != b[0]:
            return False
        from lawrag.retrieval import equivalents

        if b in equivalents(*a):  # official correspondence table
            return True
        ta, tb = self.title.get(a), self.title.get(b)
        return bool(ta) and ta == tb


# ---------------------------------------------------------------- retrievers
def legacy_retriever(query_model: str, index: SectionIndex):
    """The original tools.py search on the original (backed-up) index."""
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(ROOT / "qdrant_db_baseline"))
    if query_model == "minilm":
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        enc = lambda q: m.encode(q).tolist()  # noqa: E731
    else:
        from lawrag.embeddings import embed_query as enc

    def run(query: str, k: int):
        pts = client.query_points("indian_law", query=enc(query), limit=k).points
        hits = []
        for p in pts:
            src, text = p.payload.get("source", ""), p.payload.get("text", "")
            hits.append({"label": index.label(src, text), "text": text, "score": p.score})
        return hits

    return run


def pipeline_retriever(mode: str):
    from lawrag.retrieval import Retriever

    r = Retriever(mode=mode)

    def run(query: str, k: int):
        return [{"label": (h.act, h.section_no), "text": h.text, "score": h.score}
                for h in r.search(query, k=k)]

    return run


CONFIGS = {
    "baseline": lambda idx: legacy_retriever("minilm", idx),  # as shipped: MiniLM query vs BGE index
    "fix_embed": lambda idx: legacy_retriever("bge", idx),  # same index, matching BGE query encoder
    "dense": lambda idx: pipeline_retriever("dense"),  # re-ingested, section-aware, dense only
    "hybrid": lambda idx: pipeline_retriever("hybrid"),  # + BM25 sparse, RRF fusion
    "hybrid_rerank": lambda idx: pipeline_retriever("hybrid_rerank"),  # + cross-encoder rerank
    "full": lambda idx: pipeline_retriever("full"),  # + query analysis (exact lookup, IPC<->BNS map)
}


# ------------------------------------------------------------------- metrics
def score_query(hits, gold, index: SectionIndex, lenient: bool):
    gold = [tuple(g) for g in gold]
    match = (lambda lab, g: lab is not None and index.equivalent(lab, g)) if lenient else \
            (lambda lab, g: lab == g)
    rel = [any(match(h["label"], g) for g in gold) for h in hits]
    first = next((i for i, r in enumerate(rel) if r), None)
    found5 = {g for g in gold for h in hits[:5] if match(h["label"], g)}
    # nDCG@10 with each gold section counted once
    seen, dcg = set(), 0.0
    for i, h in enumerate(hits[:10]):
        for g in gold:
            if g not in seen and match(h["label"], g):
                seen.add(g)
                dcg += 1 / math.log2(i + 2)
                break
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    texts5 = [norm(h["text"]) for h in hits[:5]]
    return {
        "hit1": float(first is not None and first < 1),
        "hit3": float(first is not None and first < 3),
        "hit5": float(first is not None and first < 5),
        "mrr10": 1 / (first + 1) if first is not None and first < 10 else 0.0,
        "recall5": len(found5) / len(gold),
        "ndcg10": dcg / idcg if idcg else 0.0,
        "dup5": 1 - len(set(texts5)) / max(len(texts5), 1),
    }


def load_golden(names):
    rows = []
    for n in names:
        p = EVAL_DIR / n
        if p.exists():
            rows += [json.loads(l) for l in p.open(encoding="utf8") if l.strip()]
    return rows


def evaluate(config: str, rows, index: SectionIndex, k: int = 10):
    run = CONFIGS[config](index)
    run("warm up", k)
    per_query, lat = [], []
    for r in rows:
        t0 = time.perf_counter()
        hits = run(r["query"], k)
        lat.append((time.perf_counter() - t0) * 1000)
        rec = {"id": r["id"], "query": r["query"], "type": r["type"], "lang": r.get("lang", "en"),
               "gold": r["gold"], "top5": [h["label"] for h in hits[:5]],
               "top_score": hits[0]["score"] if hits else None}
        if r["gold"]:
            rec["strict"] = score_query(hits, r["gold"], index, lenient=False)
            rec["lenient"] = score_query(hits, r["gold"], index, lenient=True)
        per_query.append(rec)
    return per_query, lat


def aggregate(per_query, key="strict", filt=lambda r: True):
    rs = [r for r in per_query if key in r and filt(r)]
    if not rs:
        return None
    return {m: statistics.mean(r[key][m] for r in rs) for m in rs[0][key]} | {"n": len(rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["baseline", "fix_embed"])
    ap.add_argument("--golden", nargs="+", default=["golden_handwritten.jsonl", "golden_synthetic.jsonl"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix for config names in reports, e.g. the reranker used")
    a = ap.parse_args()

    rows = load_golden(a.golden)
    index = SectionIndex()
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = {}
    for cfg in a.configs:
        print(f"\n=== {cfg} ({len(rows)} queries)", flush=True)
        per_query, lat = evaluate(cfg, rows, index, a.k)
        s = {
            "overall_strict": aggregate(per_query, "strict"),
            "overall_lenient": aggregate(per_query, "lenient"),
            "latency_ms_p50": statistics.median(lat),
            "latency_ms_p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
            "by_type": {t: aggregate(per_query, "strict", lambda r, t=t: r["type"] == t)
                        for t in sorted({r["type"] for r in per_query if r["gold"]})},
            "by_act": {act: aggregate(per_query, "strict", lambda r, act=act: any(g[0] == act for g in r["gold"]))
                       for act in ("BNS", "BNSS", "IPC", "CrPC")},
            "out_of_scope_top_scores": [r["top_score"] for r in per_query if r["type"] == "out_of_scope"],
            "in_scope_top_scores_median": statistics.median(
                [r["top_score"] for r in per_query if r["gold"] and r["top_score"] is not None]),
        }
        name = f"{cfg}[{a.tag}]" if a.tag else cfg
        summary[name] = s
        (out_dir / f"{stamp}_{cfg}{'_' + a.tag if a.tag else ''}_queries.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, default=list) for r in per_query), encoding="utf8")
        o = s["overall_strict"]
        print(f"strict : Hit@1 {o['hit1']:.3f} Hit@5 {o['hit5']:.3f} MRR@10 {o['mrr10']:.3f} "
              f"R@5 {o['recall5']:.3f} nDCG@10 {o['ndcg10']:.3f} dup@5 {o['dup5']:.3f}")
        o = s["overall_lenient"]
        print(f"lenient: Hit@1 {o['hit1']:.3f} Hit@5 {o['hit5']:.3f} MRR@10 {o['mrr10']:.3f} R@5 {o['recall5']:.3f}")
        print(f"latency p50 {s['latency_ms_p50']:.0f} ms, p95 {s['latency_ms_p95']:.0f} ms")
    (out_dir / f"{stamp}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    from evals.report import write_report

    print("\nreport:", write_report(summary, out_dir / f"{stamp}_report.md"))


if __name__ == "__main__":
    main()
