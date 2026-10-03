# Improvement Report: Indian Criminal Law RAG Agent

_Audit, fixes and measured results, October 2026._

## TL;DR

- The original system's retrieval was mostly broken, and no measurement existed to show it. On a new 107-query golden set, the shipped pipeline found a correct section in the top 5 for **18.7%** of queries. The README claimed "85–90% retrieval accuracy".
- Three root causes:
  1. Queries and documents were embedded with **different models**.
  2. The **BNS PDF was extracted with every space removed**, so the main current law was essentially unsearchable.
  3. Chunking ignored legal structure, and the index held **2,416 duplicate chunks**.
- After the fixes: **Hit@5 = 88.8% strict / 92.5% lenient, MRR@10 = 0.726** (from 0.168), at ~1.4 s retrieval latency on a laptop CPU.
- Default model is now **qwen2.5:3b**: it fits fully on a 4 GB GPU and answers in ~14 s (was ~96 s), citing the gold section in 90% of answers with zero citation errors flagged (§3.4).
- Generation (qwen3:8b, 30 queries): a gold section was cited in **97%** of answers, citation precision **0.975**, **no non-existent sections** and **no wrong old↔new correspondences**. The remaining errors (wrong source tag on 2 of 30 answers) are flagged automatically in the UI. The original llama3.2 cited the gold section in only 77%.
- New features: **linked citations** (exact PDF page, India Code, Indian Kanoon), a **citation verifier** (flags invalid tags, unretrieved sections, non-existent sections, claim/citation mismatches and wrong old↔new correspondences), **IPC↔BNS / CrPC↔BNSS mapping** from the official correspondence tables, **exact section lookup**, **abstention**, **web and case-law search that keeps URLs**, a streaming chat UI with a real source panel, feedback logging, and an **eval harness**.

---

## 1. Findings in the original code

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| A1 | Critical | Query/document embedding mismatch | `ingestion.py` used `BAAI/bge-small-en-v1.5`, `tools.py` used `all-MiniLM-L6-v2`. Both are 384-d, so nothing errored, but cosine scores were close to noise. BGE's query instruction was also missing. | `embeddings.py` is shared by ingestion and query, and uses `query_embed()`. The model name is stored in the collection metadata and checked at startup (`store.check_collection`). |
| A2 | Critical | BNS 2023 effectively missing | Only **82 of 7,442** chunks were from BNS. MarkItDown output was `"Whoever,beinglegallyboundto…"`, and chunks reached **45,647 chars**, far beyond the 512-token embedding window. | `lawrag/parser.py` (PyMuPDF, layout-aware), with an ingestion sanity check that fails if the average word length is above 12. BNS now has all **358/358** sections. |
| A3 | High | Non-idempotent ingestion | `uuid4()` IDs: every re-run duplicated the index. 2,224 texts were stored ×2, and boilerplate up to ×58. | Deterministic `uuid5(act:section:chunk)` IDs, text dedup, `--rebuild`. A re-run keeps the same **2,604** points. |
| A4 | High | Structure-blind chunks with no metadata | 1,991 chunks under 100 chars, chunks straddling sections, payload `{source, text}` only. | One chunk per section (long sections split at sub-sections), a context header (`Act — Section N: Title (Chapter)`), and a payload with act, section_no, title, chapter, page and URL. |
| A5 | High | Citations lost through 3 paraphrasing agents | The retrieval agent paraphrased the tool output, research paraphrased that, reasoning paraphrased again. The "Retrieved Sources" panel was a stub. | Deterministic retrieve → single grounded generation call with tagged sources ([S1], [W1]) → verifier. The source panel shows every chunk with score, page and links. |
| A6 | High | Web links dropped | Serper results' `link` field never reached the answer. The wrapper tried 6 call signatures and swallowed every error. | `web_search.py`: typed results with URLs, restricted to trusted legal domains, plus the Indian Kanoon API. Cited as `[W#](url)`. |
| A7 | Medium | Duplicate code and dead config | `app.py` redefined the agents and ignored `crew.py`/YAML. `.env` model/path were ignored (hardcoded). Deprecated `crewai_tools.tool`. Unpinned deps. Qdrant client opened at import (lock errors on rerun). | `config.py` (pydantic-settings) is the single source of config. Shared cached Qdrant client. `crew.py` is now the optional agentic mode, driven by the YAML. Pinned requirements. |
| A8 | Medium | Missing legal features | No IPC↔BNS mapping, no abstention, no disclaimer, no conversation memory. | Section map + query analysis, calibrated abstention, disclaimer, chat history. |
| A9 | Medium | Unsupported metrics in README/report | "85–90% accuracy" and "3–6 s" with no eval code or data. | `evals/` harness. The README now reports measured numbers. |
| A10 | Low | Hygiene | No tests, CI, logging. 22 MB recording in the repo root. | `tests/`, GitHub Actions, logging, `.gitignore` updates. |

## 2. Architecture

**Before:** Streamlit → CrewAI (Retrieval agent → Research agent → Reasoning agent, each a llama3.2 3B call) → answer. Retrieval quality depended on an LLM rewriting the query, and citations depended on three rounds of paraphrase.

**After:** a deterministic core, with an agent only where it helps.

```
query ─► analyze (refs, IPC↔BNS map, act filter, translate) ─► exact lookup + hybrid (BGE dense + BM25, RRF)
      ─► cross-encoder rerank ─► abstain? ─► expand to full sections ─► one grounded LLM call citing [S#]/[W#]
      ─► verify & link ─► UI
```

The CrewAI path is kept as an optional **agentic research mode**. Its tools register every source, so the same verifier and links apply.

| Module | Role |
|---|---|
| `lawrag/parser.py` | PDF → sections. Gazette layout (BNS/BNSS: titles in margin notes) and India Code layout (IPC/CrPC: `N. Title.—`). Drops footnotes and amendment markers. |
| `lawrag/ingest.py` | Section-aware chunks, dense + sparse vectors, deterministic IDs, sanity checks. |
| `lawrag/retrieval.py` | Query analysis, exact lookup, hybrid search, rerank, one result per section. |
| `lawrag/answer.py` | Abstention, small-to-big section expansion, official-correspondence notes, prompt, generation (blocking or streaming). |
| `lawrag/citations.py` | Tag → link, invalid tags, unsupported / non-existent sections, claim–citation mismatch, wrong old↔new correspondence. |
| `lawrag/web_search.py` | Indian Kanoon + Serper with URLs. |
| `data/section_map.csv` | 1,074 official correspondences (549 BNS↔IPC, 525 BNSS↔CrPC), built by `scripts/build_section_map.py` from the UP Police tables. |

## 3. Evaluation

### 3.1 Golden set

- **60 hand-written queries** (`evals/golden_handwritten.jsonl`): concept (31), procedure (17), IPC↔BNS mapping (3), section lookup (4), Hindi (3), out-of-scope (5). Every gold `(act, section)` was **checked against the parsed section titles**.
- **52 synthetic queries** (`evals/golden_synthetic.jsonl`): generated by qwen3:8b from 13 random sections per Act, without naming the section (`evals/make_synthetic.py`). Gold is that single section.
- **Strict** = exact act + section. **Lenient** = the equivalent section in the paired code also counts (official map or identical title).
- For the legacy index, which has no metadata, retrieved chunks are mapped to sections by normalized text containment. Coverage was 65–100% per Act; the unmapped chunks are headers, form schedules and margin notes, which can't be gold anyway.

### 3.2 Retrieval results (107 in-scope queries, k=10)

| Config | What changed | Hit@1 | Hit@5 | MRR@10 | Recall@5 | nDCG@10 | Hit@5 (lenient) | Dup@5 | p50 latency |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | As shipped | 0.150 | 0.187 | 0.168 | 0.118 | 0.134 | 0.262 | 0.237 | 29 ms |
| `fix_embed` | Same index, BGE query encoder (A1) | 0.224 | 0.327 | 0.271 | 0.235 | 0.238 | 0.374 | 0.275 | 25 ms |
| `dense` | Re-ingested: clean extraction, section chunks (A2–A4) | 0.533 | 0.701 | 0.611 | 0.674 | 0.639 | 0.757 | 0.000 | 13 ms |
| `hybrid` | + BM25 sparse, RRF fusion | 0.533 | 0.776 | 0.632 | 0.751 | 0.664 | 0.832 | 0.000 | 81 ms |
| `hybrid_rerank` | + cross-encoder (MiniLM-L-6) | 0.561 | 0.860 | 0.684 | 0.813 | 0.721 | 0.907 | 0.000 | 1.3 s |
| **`full`** | + query analysis (exact lookup, IPC↔BNS map, act filter, translation) | **0.617** | **0.888** | **0.726** | **0.855** | **0.765** | **0.925** | 0.000 | 1.4 s |

By query type (`full` vs `baseline`, Hit@5): concept **1.000** (0.323) · procedure **0.941** (0.353) · lookup **1.000** (0.000) · mapping **1.000** (0.000) · synthetic **0.788** (0.077). By gold Act: BNS **0.915** (0.213) · BNSS **0.969** (0.219) · IPC **0.929** (0.262) · CrPC **0.833** (0.233).

Full tables: `evals/results/latest_retrieval_report.md`.

**Reranker choice.** `BAAI/bge-reranker-base` reached Hit@5 0.832 / MRR 0.678 in `hybrid_rerank`, but took ~20 s per query on CPU. `Xenova/ms-marco-MiniLM-L-6-v2` reached **0.860 / 0.684** at ~1.3 s, so it is the default (`RERANK_MODEL`).

**Abstention threshold** (MiniLM reranker scores). Out-of-scope top scores were −11.1, −2.4, −11.2, −4.5, −10.4. The in-scope 5th percentile was −1.6, and the minimum −5.7. At **−4.0**, 4 of 5 out-of-scope queries abstain, with 2% false abstention. The threshold is reranker-specific: re-calibrate if `RERANK_MODEL` changes.

**Remaining retrieval errors** are mostly synthetic paraphrases of procedural CrPC sections (e.g. "commission to examine witnesses"), where the top hits were neighbouring sections. Also: one Hindi query (anticipatory bail) where the translation lost the term.

### 3.3 Generation results

30 in-scope hand-written English queries + 5 out-of-scope, `full` retrieval, top-6 sources (`evals/generation_eval.py`). All runs re-scored with the final verifier rules (`evals/rescore_generation.py`). Citation precision counts sections from the official correspondence table given to the model as supported.

| Metric | llama3.2 (original model) | qwen3:8b, v1 prompt | qwen3:8b, + correspondence notes | **qwen3:8b, + full-section context (final)** |
|---|---|---|---|---|
| Answers carrying citation tags | 0.700 | 0.967 | 0.933 | **0.967** |
| Citation precision | 0.867 | 1.000 | 0.987 | **0.975** |
| Gold section cited | 0.767 | 1.000 | 0.967 | **0.967** |
| Non-existent section cited | 0.000 | 0.000 | 0.000 | **0.000** |
| Wrong old↔new correspondence claimed | 0.000 | 0.033 | 0.000 | **0.000** |
| Claim–citation mismatch (flagged in UI) | 0.000 | 0.000 | 0.033 | **0.067** |
| Out-of-scope abstention | 0.8 | 0.8 | 0.8 | **0.8** (+1 declined by the model) |
| False abstention (in-scope) | 0.0 | 0.0 | 0.0 | **0.0** |
| Latency p50 (GTX 1650, 4 GB) | 22 s | 74 s | 74 s | **96 s** |

**How to read this**
- **qwen3:8b clearly beats llama3.2.** With llama3.2, 30% of answers carried no citation tags and the gold section was cited in only 77%. It is 3–4× faster, though.
- **Each prompt/context fix was driven by a concrete failure in these runs:**
  - The v1 answer to "defamation" said "IPC 499 → BNS 353" (correct: 356), so the official correspondences are now passed to the model.
  - The agentic smoke test said BNS 303 "has no punishment" because sub-section (2) was in an unretrieved chunk. Matched chunks are now expanded to the whole section (illustrations dropped first if over 4,000 chars).
  - The differences between the last three columns are 1–2 queries out of 30, which is within run-to-run noise. Bigger samples are needed to rank them.
- **The two mismatches in the final run are real errors that the UI flags.** One answer states the correct mapping (IPC 499 → BNS 356) but tags the wrong source (BNS 353). Another cites IPC 390 for a claim about IPC 383.
- **Faithfulness (LLM judge) is not reported as a headline number.**
  - The first judge truncated each source to 1,200 characters. After full-section expansion it marked correct claims as "unsupported", and the score fell from 4.67 to 4.13, an eval bug, not a model regression.
  - With the judge fixed (full sources + correspondence notes, 16k context), the same model gave **5/5 to all 30 answers**. A self-judging 8B model isn't discriminative enough to be trusted.
  - Use a stronger, different judge model (`--judge`), or human review, before relying on this metric.
- **Latency is dominated by the GPU.** qwen3:8b needs 6.6 GB, but the GTX 1650 has 4 GB, so ~64% of the model runs on the CPU. Answers stream token by token in the UI.

### 3.4 Model selection for a 4 GB GPU

qwen3:8b answers took 74–96 s on the development laptop (GTX 1650, 4 GB). It needs 6.6 GB, so Ollama ran ~64% of it on the CPU. Ollama gets ~2.3 GB of usable VRAM there.

**Speed and GPU fit.** Measured prompts peak at ~3.9k tokens, so the context window dropped from 8k to **6k** (`OLLAMA_NUM_CTX`); a smaller KV cache leaves more room for the model. At 6k:

| Model | On GPU | Generation speed |
|---|---|---|
| qwen2.5:3b | **100%** | **56 tok/s** |
| llama3.2 3B | 75% | 41 tok/s |
| qwen3:4b | 60% | 18 tok/s |

**Quality.** The first qwen2.5:3b run cited the gold section in only 70% of answers. Inspection showed it *was* citing, but in loose formats: `(S1)`, `[S2, Section 307]`, `[S1, S2: Section 420]`. Two fixes followed:
- The verifier now normalizes any bracket or parenthesis group containing source ids into proper tags. This helps every model.
- The prompt spells out the exact tag format, with an example.

**Robustness.** One run hung on a single question for 10 minutes; small models can loop. Answers are now capped at 1,024 tokens with `repeat_penalty` 1.1, the HTTP timeout is 300 s, and the eval records errors instead of crashing.

| Model | Gold section cited | Answers with tags | Citation precision | Non-existent sections | Median answer time |
|---|---|---|---|---|---|
| **qwen2.5:3b, final prompt (default)** | **90.0%** | 93.3% | 0.989 | 0 | **14 s** |
| qwen3:8b | 96.7% | 96.7% | 0.950 | 0 | 96 s |
| llama3.2 | 83.3% | 76.7% | 0.933 | 0 | 22 s |

**Mapping questions ("What is IPC 302 called in BNS?") are resolved deterministically.**
- Even with the official correspondence in its prompt, the 3B model answered these wrongly ("IPC 302 is called Section 302 in the BNS").
- Now the app detects them, gives the model only the exact and mapped sections, and states the official correspondence as a fact. The model just explains the new section.
- A 📌 banner built from the table (not the LLM) shows the mapping, with links.
- Result: **3/3 mapping answers correct** (`mapping_correct`, a stricter metric that requires the answer *text* to name the right new section). The generic "gold cited" metric had scored the wrong answers as correct, because they mentioned the old section.

While restructuring the repository, the section-map path briefly pointed to a non-existent file, and the loader silently returned an empty map. It now fails loudly, and a unit test guards it. All qwen2.5:3b numbers above were measured after that fix.

With qwen2.5:3b also translating the 3 Hindi queries, `full` retrieval measures **Hit@5 0.879 / MRR 0.718** (vs 0.888 / 0.726 with qwen3:8b translating), a one-query difference.

qwen2.5:3b is the default: about 7× faster for ~7 points less gold-citation recall, with no flagged citation errors (mismatch and wrong-correspondence rates both 0) in its final run. A first version of the format example used a real section number ("Section 103 of the BNS"), which the 3B model copied into an unrelated answer; the example now uses placeholders only. On a machine with 8 GB+ VRAM, `OLLAMA_MODEL=qwen3:8b` is the better choice.

## 4. Feature summary (what you get in the UI)

1. **Linked citations.** Each `[S1]` becomes `[[BNS s.103]](…/BNS_2023.pdf#page=34)`, opening the exact page of the indexed PDF (Streamlit static serving). The source panel adds the official **India Code** Act page and an **Indian Kanoon** search link for the section and its case law.
2. **Web/case-law links.** With `INDIAN_KANOON_API_TOKEN` and/or `SERPER_API_KEY`, judgments and articles are cited as `[W1](https://indiankanoon.org/doc/…)`. Only trusted domains are allowed.
3. **Citation checks** under the answer: ⚠ for non-existent sections and claim/citation mismatches, ℹ for sections mentioned but not retrieved.
4. **Old↔new code awareness.** Asking about "IPC 302" retrieves BNS 103 too, and the prompt states the official correspondence.
5. **Abstention + disclaimer** for out-of-scope questions.
6. **Streaming chat** with history, a sources expander, and 👍/👎 feedback logged to `logs/feedback.jsonl`. Bad answers become new golden-set rows.
7. **Agentic mode** toggle (CrewAI researcher + writer) for open-ended research.

## 5. Recommended next steps (not yet implemented)

| Priority | Recommendation | Why |
|---|---|---|
| High | **Independent, stronger judge** (a different model family, or human review of a sample) for faithfulness | The self-judge scored every answer 5/5 once given full context, so it can't separate good answers from bad. |
| Medium | **Auto-repair of flagged citations**: when the verifier finds a claim–citation mismatch, re-tag the sentence with the source that actually contains the named section, or ask the model to fix it | The remaining errors (6.7%) are wrong tags on otherwise correct statements, and they are detected deterministically. |
| High | **Grow the golden set** from `logs/feedback.jsonl` and have a lawyer review gold labels; add multi-hop questions | 107 queries and 5 out-of-scope cases are enough to rank configs, but not to give tight confidence intervals. |
| Medium | **Native multilingual retrieval** (`BAAI/bge-m3`, dense + sparse in one model) | Removes the LLM translation hop for Hindi/Marathi. |
| Medium | **Add the Bharatiya Sakshya Adhiniyam / Evidence Act** | Many criminal-law questions are evidentiary. |
| Medium | **Offence-date routing**: ask for, or detect, the offence date | IPC/CrPC apply to offences before 1 July 2024. |
| Medium | **Parse the First Schedule table** (cognizable/bailable/triable-by) into structured rows | "Is X bailable?" is a top user question. The table is currently chunked as text. |
| Low | **Qdrant server (Docker)** instead of embedded mode | Lets evals and the app run concurrently. The local mode holds a file lock. |
| Low | **Section-level India Code deep links** | Needs India Code's per-section IDs, which can't be derived from the section number. Current links go to the exact PDF page plus the Act page. |

## 6. Reproduce

```bash
pip install -r requirements-dev.txt
python -m lawrag.ingest --rebuild
python -m evals.retrieval_eval --configs baseline fix_embed dense hybrid hybrid_rerank full
python -m evals.generation_eval --n 30 --model qwen3:8b
pytest -q
```

`baseline`/`fix_embed` need a copy of the original index at `qdrant_db_baseline/` (collection `indian_law`, built by the old `ingestion.py`).
