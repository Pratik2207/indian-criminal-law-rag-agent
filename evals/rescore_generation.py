"""Re-score saved generation-eval runs with the current citation verifier.

Retrieval is deterministic, so sources are recomputed (no LLM calls); the saved
answers and judge scores are reused. Use this after changing citations.py or the
metric definitions, so old and new runs are compared with the same rules.

    python -m evals.rescore_generation evals/results/*_generation_*.json
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(paths, rejudge: str | None = None):
    from lawrag.answer import Answerer, correspondence_notes
    from lawrag.citations import verify_and_link
    from evals.generation_eval import extract_tags, judge
    from evals.retrieval_eval import load_golden
    from lawrag.retrieval import equivalents

    gold = {r["id"]: r["gold"] for r in load_golden(["golden_handwritten.jsonl"])}
    ans = Answerer()
    cache = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf8"))
        rows = []
        for r in d["results"]:
            if r["type"] == "out_of_scope" or r.get("abstained") or not r.get("answer"):
                continue
            if r["query"] not in cache:
                prep = ans.prepare(r["query"])
                cache[r["query"]] = (prep.sources, correspondence_notes(prep.sources, prep.analysis.correspondences))
            sources, notes = cache[r["query"]]
            if rejudge:
                j = judge(r["query"], r["answer"], sources, rejudge, notes)
                r["faithfulness"], r["unsupported_claims"] = j.get("score"), j.get("unsupported_claims", [])
                print(f"  {r['id']} faithfulness={r['faithfulness']}", flush=True)
            v = verify_and_link(r["answer"], sources)
            by_id = {s.id: s for s in sources}
            tags = extract_tags(r["answer"])
            cited = {(by_id[t].act, by_id[t].section_no) for t in tags if t in by_id and by_id[t].kind == "statute"}
            cited |= set(v.mentions)
            retrieved = {(s.act, s.section_no) for s in sources if s.kind == "statute"}
            supported = retrieved | {e for sec in retrieved for e in equivalents(*sec)}
            g = {tuple(x) for x in gold[r["id"]]}
            rows.append({
                "answers_with_tags": float(bool(tags)),
                "tag_validity": (1 - len(v.invalid_tags) / len(tags)) if tags else 0.0,
                "citation_precision": len(cited & supported) / len(cited) if cited else 0.0,
                "gold_cited": float(bool(cited & g)),
                "hallucinated_section_rate": float(bool(v.nonexistent)),
                "claim_citation_mismatch_rate": float(bool(v.mismatched)),
                "wrong_correspondence_rate": float(bool(v.bad_mappings)),
                "faithfulness_1to5": r.get("faithfulness"),
            })
        def mean(k):
            vals = [x[k] for x in rows if x[k] is not None]
            return round(statistics.mean(vals), 3) if vals else None

        summary = {k: mean(k) for k in rows[0]}
        summary |= {"generator": d["summary"]["generator"], "n": len(rows),
                    "latency_s_p50": round(d["summary"]["latency_s_p50"], 1),
                    "oos_abstention_rate": d["summary"]["oos_abstention_rate"]}
        print(Path(p).name, json.dumps(summary))
        if rejudge:
            out = Path(p).with_name(Path(p).stem + "_rejudged.json")
            out.write_text(json.dumps({"summary": d["summary"] | summary, "results": d["results"]}, indent=2,
                                      ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rejudge", metavar="MODEL", help="re-run the LLM judge with full sources, e.g. qwen3:8b")
    a = ap.parse_args()
    main(a.paths, a.rejudge)
