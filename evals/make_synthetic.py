"""Generate synthetic eval queries from statute sections with a local LLM.

Each query is written from one section's text (without naming the section or
Act), so that section is its gold answer. Review the output by hand before use.

    python -m evals.make_synthetic --per-act 13 --out evals/golden_synthetic.jsonl
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lawrag.config import get_settings  # noqa: E402
from lawrag.parser import parse_all  # noqa: E402

PROMPT = """You write evaluation questions for an Indian criminal law search engine.

Below is one section of the {act_name}. Write ONE natural question (max 25 words) that a
law student or citizen might ask, which this section answers.
Rules: do NOT mention the section number, do NOT name the Act, use plain words
(paraphrase, don't copy phrases). Output only the question.

Section title: {title}
Section text:
{text}
"""


def generate(prompt: str) -> str:
    s = get_settings()
    r = requests.post(
        f"{s.ollama_base_url}/api/generate",
        json={"model": s.ollama_model, "prompt": prompt, "stream": False, "think": False,
              "options": {"temperature": 0.7}},
        timeout=300,
    )
    r.raise_for_status()
    out = r.json()["response"]
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"')
    return out.splitlines()[0].strip() if out else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-act", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="evals/golden_synthetic.jsonl")
    a = ap.parse_args()

    random.seed(a.seed)
    secs = [s for s in parse_all(get_settings().knowledge_dir)
            if s.section_no[0].isdigit() and s.title and 300 < len(s.text) < 4000
            and "definition" not in s.title.lower() and "omitted" not in s.text[:80].lower()]
    rows = []
    for act in ("BNS", "BNSS", "IPC", "CrPC"):
        pool = [s for s in secs if s.act == act]
        for i, s in enumerate(random.sample(pool, a.per_act)):
            q = generate(PROMPT.format(act_name=s.act_name, title=s.title, text=s.text[:2500]))
            rows.append({"id": f"s_{act}_{i:02d}", "query": q, "gold": [[s.act, s.section_no]],
                         "type": "synthetic", "lang": "en", "gold_title": s.title})
            print(rows[-1]["id"], s.section_no, s.title[:40], "->", q, flush=True)
    with open(a.out, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
