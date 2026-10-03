"""End-to-end answer evaluation (retrieval + generation + citation verification).

Metrics per answer:
  tag_validity        share of [S#]/[W#] tags that point to a real source
  citation_precision  share of cited sections (tags + "Section N of X" mentions) that were retrieved
  gold_cited          a gold section (or its IPC<->BNS / CrPC<->BNSS equivalent) is cited
  hallucinated        answer mentions a section that does not exist in the indexed Acts
  abstained           system declined to answer (correct for out-of-scope queries)
  faithfulness        1-5 LLM-judge score: every claim supported by the sources?

    python -m evals.generation_eval --n 30
    python -m evals.generation_eval --n 30 --model llama3.2   # compare generators
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

JUDGE_PROMPT = """You are grading an AI legal assistant's answer for FAITHFULNESS to its sources.

Sources:
{sources}

Question: {question}

Answer:
{answer}

Score 1-5: 5 = every factual claim is supported by the sources; 3 = mostly supported, some
unsupported details; 1 = largely unsupported or contradicts the sources.
Reply with JSON only: {{"score": <1-5>, "unsupported_claims": ["..."]}}"""


def extract_tags(answer: str) -> list[str]:
    from lawrag.citations import TAG_RE, normalize_tags

    text = normalize_tags(answer)
    return [f"{k}{n}" for k, n in TAG_RE.findall(text)]


def judge(question, answer, sources, judge_model: str, notes: list[str] | None = None) -> dict:
    import requests

    from lawrag.config import get_settings

    # the judge must see exactly what the generator saw: full sources + correspondence notes
    src = "\n\n".join(f"[{s.id}] {s.label}\n{s.text}" for s in sources)
    if notes:
        src += "\n\nOfficial section correspondence (old code <-> new code): " + "; ".join(notes)
    r = requests.post(f"{get_settings().ollama_base_url}/api/chat", timeout=600, json={
        "model": judge_model, "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 16384},
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(sources=src, question=question, answer=answer)}],
    })
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"score": None, "unsupported_claims": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of in-scope handwritten queries")
    ap.add_argument("--types", nargs="*", help="only these query types, e.g. mapping lookup")
    ap.add_argument("--model", default=None, help="generator model (default: OLLAMA_MODEL)")
    ap.add_argument("--judge", default="none", help="LLM judge model for faithfulness, or \"none\" (default)")
    a = ap.parse_args()
    if a.model:
        os.environ["OLLAMA_MODEL"] = a.model
    from lawrag.config import get_settings

    get_settings.cache_clear()
    from lawrag.answer import Answerer, correspondence_notes
    from evals.retrieval_eval import load_golden
    from lawrag.retrieval import equivalents

    rows = load_golden(["golden_handwritten.jsonl"])
    in_scope = [r for r in rows if r["gold"] and r["lang"] == "en"][: a.n]
    if a.types:  # e.g. --types mapping lookup (these sit beyond the first 30 queries)
        in_scope = [r for r in rows if r["gold"] and r["lang"] == "en" and r["type"] in a.types]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    ans = Answerer()
    model = get_settings().ollama_model
    results = []
    for r in in_scope + oos:
        t0 = time.perf_counter()
        try:
            res = ans.answer(r["query"])
        except Exception as e:  # e.g. LLM timeout: record and keep going
            print(f"{r['id']} ERROR {type(e).__name__}: {e}", flush=True)
            results.append({"id": r["id"], "query": r["query"], "type": r["type"], "error": str(e)})
            continue
        dt = time.perf_counter() - t0
        rec = {"id": r["id"], "query": r["query"], "type": r["type"], "abstained": res.abstained,
               "latency_s": dt, "answer": res.raw_answer}
        if not res.abstained:
            v = res.verification
            tags = extract_tags(res.raw_answer)
            by_id = {s.id: s for s in res.sources}
            cited = {(by_id[t].act, by_id[t].section_no) for t in tags if t in by_id and by_id[t].kind == "statute"}
            cited |= set(v.mentions)
            retrieved = {(s.act, s.section_no) for s in res.sources if s.kind == "statute"}
            # sections given to the model via the official correspondence notes are supported too
            retrieved |= {e for sec in list(retrieved) for e in equivalents(*sec)}
            gold = {tuple(g) for g in r["gold"]}
            gold_eq = gold | {e for g in gold for e in equivalents(*g)}
            rec |= {
                "tag_validity": (1 - len(v.invalid_tags) / len(tags)) if tags else 0.0,
                "n_tags": len(tags),
                "citation_precision": len(cited & retrieved) / len(cited) if cited else 0.0,
                "gold_cited": float(bool(cited & gold)),
                "gold_cited_lenient": float(bool(cited & gold_eq)),
                "hallucinated": float(bool(v.nonexistent)),
                "mismatched": float(bool(v.mismatched)),
                "bad_mapping": float(bool(v.bad_mappings)),
                "bad_mappings": v.bad_mappings,
                "nonexistent": v.nonexistent,
                # mapping questions: the answer text must name the correct new-code section (a mention of
                # the old section alone, which is also gold, does not count)
                **({"mapping_correct": float(any(tuple(g) in set(v.mentions) for g in r["gold"] if g[0] in ("BNS", "BNSS"))
                                             and not v.bad_mappings)} if r["type"] == "mapping" else {}),
            }
            if r["gold"] and a.judge != "none":
                j = judge(r["query"], res.raw_answer, res.sources, a.judge,
                          correspondence_notes(res.sources, res.analysis.correspondences))
                rec["faithfulness"] = j.get("score")
                rec["unsupported_claims"] = j.get("unsupported_claims", [])
        results.append(rec)
        print(f"{r['id']} abst={res.abstained} {dt:.1f}s "
              f"gold={rec.get('gold_cited')} prec={rec.get('citation_precision', 0):.2f} "
              f"faith={rec.get('faithfulness')}", flush=True)

    errors = [x for x in results if "error" in x]
    ok = [x for x in results if "error" not in x]
    ins = [x for x in ok if x["type"] != "out_of_scope"]
    answered = [x for x in ins if not x["abstained"]]
    def mean(k, xs=answered):
        vals = [x[k] for x in xs if x.get(k) is not None]
        return statistics.mean(vals) if vals else None  # e.g. faithfulness when --judge none
    summary = {
        "generator": model, "judge": a.judge, "n_in_scope": len(ins), "n_out_of_scope": len(oos),
        "false_abstention_rate": sum(x["abstained"] for x in ins) / len(ins),
        "oos_abstention_rate": sum(x["abstained"] for x in ok if x["type"] == "out_of_scope") / max(len(oos), 1),
        "n_errors": len(errors),  # e.g. LLM timeouts; excluded from the rates below
        "tag_validity": mean("tag_validity"), "answers_with_tags": sum(x["n_tags"] > 0 for x in answered) / max(len(answered), 1),
        "citation_precision": mean("citation_precision"), "gold_cited": mean("gold_cited"),
        "gold_cited_lenient": mean("gold_cited_lenient"), "hallucinated_section_rate": mean("hallucinated"),
        "claim_citation_mismatch_rate": mean("mismatched"),
        "wrong_correspondence_rate": mean("bad_mapping"),
        "mapping_correct": mean("mapping_correct"),
        "faithfulness_1to5": mean("faithfulness"),
        "latency_s_p50": statistics.median(x["latency_s"] for x in answered) if answered else None,
    }
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", model)
    (out / f"{stamp}_generation_{safe}.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
