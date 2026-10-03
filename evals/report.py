"""Render a retrieval-eval summary as a Markdown comparison report."""
from pathlib import Path

METRICS = [("hit1", "Hit@1"), ("hit3", "Hit@3"), ("hit5", "Hit@5"), ("mrr10", "MRR@10"),
           ("recall5", "Recall@5"), ("ndcg10", "nDCG@10"), ("dup5", "Dup@5 ↓")]


def _table(rows: dict, title: str) -> str:
    out = [f"### {title}", "", "| Config | n | " + " | ".join(m[1] for m in METRICS) + " |",
           "|---|---|" + "---|" * len(METRICS)]
    for name, agg in rows.items():
        if agg:
            out.append(f"| {name} | {agg['n']} | " + " | ".join(f"{agg[m]:.3f}" for m, _ in METRICS) + " |")
    return "\n".join(out) + "\n"


def write_report(summary: dict, path: Path) -> Path:
    parts = ["# Retrieval evaluation report", ""]
    parts.append(_table({c: s["overall_strict"] for c, s in summary.items()}, "Overall (strict: exact act + section)"))
    parts.append(_table({c: s["overall_lenient"] for c, s in summary.items()},
                        "Overall (lenient: equivalent IPC↔BNS / CrPC↔BNSS section also counts)"))
    parts.append("### Latency (retrieval only)\n\n| Config | p50 ms | p95 ms |\n|---|---|---|")
    parts += [f"| {c} | {s['latency_ms_p50']:.0f} | {s['latency_ms_p95']:.0f} |" for c, s in summary.items()]
    parts.append("")
    types = sorted({t for s in summary.values() for t in s["by_type"]})
    for t in types:
        parts.append(_table({c: s["by_type"].get(t) for c, s in summary.items()}, f"Query type: {t} (strict)"))
    for act in ("BNS", "BNSS", "IPC", "CrPC"):
        parts.append(_table({c: s["by_act"].get(act) for c, s in summary.items()}, f"Gold Act: {act} (strict)"))
    path.write_text("\n".join(parts), encoding="utf8")
    return path
