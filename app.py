import json
import logging
import os
import shutil
import time
from pathlib import Path

os.environ["CREWAI_TELEMETRY"] = "false"

import streamlit as st  # noqa: E402

from lawrag.config import ROOT, get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO)
STATIC = ROOT / "static"
FEEDBACK_LOG = ROOT / "logs" / "feedback.jsonl"

st.set_page_config(page_title="Indian Criminal Law RAG Agent", page_icon=":scales:", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def load_answerer():
    from lawrag.answer import Answerer

    # Serve the statute PDFs so citations can link to the exact page (needs enableStaticServing).
    STATIC.mkdir(exist_ok=True)
    for pdf in Path(get_settings().knowledge_dir).glob("*.pdf"):
        if not (STATIC / pdf.name).exists():
            shutil.copy(pdf, STATIC / pdf.name)
    return Answerer()


def log_feedback(entry: dict):
    FEEDBACK_LOG.parent.mkdir(exist_ok=True)
    with FEEDBACK_LOG.open("a", encoding="utf8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def render_sources(res):
    statutes = [s for s in res.sources if s.kind == "statute"]
    web = [s for s in res.sources if s.kind != "statute"]
    cited = set(res.verification.used_ids) if res.verification else set()
    with st.expander(f"Sources ({len(statutes)} statute sections, {len(web)} web/case law)", expanded=False):
        if res.analysis and (res.analysis.refs or res.analysis.mapped):
            refs = ", ".join(f"{a} s.{n}" for a, n in res.analysis.refs)
            mapped = ", ".join(f"{a} s.{n}" for a, n in res.analysis.mapped)
            st.caption(f"Detected references: {refs or '—'} · Old↔new equivalents: {mapped or '—'}")
        for s in statutes:
            mark = "✅ cited" if s.id in cited else "not cited"
            links = " · ".join(f"[{k}]({v})" for k, v in s.extra_urls.items())
            st.markdown(f"**[{s.id}] [{s.label}]({s.url})** · {mark}  \n{links}")
            st.text(s.text[:1500])
        for s in web:
            st.markdown(f"**[{s.id}] [{s.label}]({s.url})** ({s.kind})  \n{s.text}")


def answer_turn(query: str, history: list[dict]):
    from lawrag.answer import ABSTAIN_MSG, DISCLAIMER

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        with st.status("Retrieving statute sections…") as status:
            res = answerer.prepare(query, use_web=use_web)
            status.update(label=f"Retrieved {len(res.sources)} sources in {res.timings['retrieval_s']:.1f}s",
                          state="complete")
        placeholder = st.empty()
        if res.abstained:
            res.answer_markdown = ABSTAIN_MSG + "\n\n" + DISCLAIMER
        else:
            buf = ""
            try:
                for tok in answerer.stream(res, history):
                    buf += tok
                    placeholder.markdown(buf + "▌")
            except Exception as e:  # Ollama not running, model missing, ...
                st.error(f"LLM error: {e}. Is Ollama running with model `{get_settings().ollama_model}`?")
                st.stop()
            answerer.finish(res, buf)  # verify citations and turn [S1] tags into links
        placeholder.markdown(res.answer_markdown)
        res.timings["total_s"] = time.perf_counter() - t0
        if show_debug:
            render_sources(res)
        st.session_state.messages.append({"role": "assistant", "content": res.answer_markdown,
                                          "raw": res.raw_answer, "result": res})


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    s = get_settings()
    st.caption(f"LLM: `{s.ollama_model}` via Ollama · Index: `{s.qdrant_collection_name}`")
    use_web = st.toggle("Search case law and the web", value=False,
                        help="Uses Indian Kanoon (INDIAN_KANOON_API_TOKEN) and/or Serper (SERPER_API_KEY). "
                             "Results are cited with links.")
    if use_web and not (s.serper_api_key or s.indian_kanoon_api_token):
        st.warning("Set SERPER_API_KEY and/or INDIAN_KANOON_API_TOKEN in .env to enable web search.")
    agentic = st.toggle("Agentic research mode (CrewAI, slower)", value=False,
                        help="An agent decides which searches/lookups to run. Default mode is a single fast "
                             "retrieve-then-answer pass.")
    show_debug = st.toggle("Show retrieval details", value=True)
    if st.button("New conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("⚖️ Indian Criminal Law RAG Agent")
st.caption("Grounded answers from the BNS, BNSS, IPC and CrPC, with linked, verified citations. Runs locally.")

answerer = load_answerer()
st.session_state.setdefault("messages", [])

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("result") and show_debug:
            render_sources(m["result"])

query = st.chat_input("Ask a legal question, e.g. What is the punishment for murder under BNS?")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    history = [{"role": m["role"], "content": m.get("raw", m["content"])} for m in st.session_state.messages[:-1]]
    if agentic:
        with st.chat_message("assistant"):
            from lawrag.answer import DISCLAIMER, AnswerResult
            from lawrag.crew import IndianLawRagCrew

            t0 = time.perf_counter()
            with st.status("Agents researching…"):
                try:
                    raw, verification, sources = IndianLawRagCrew(answerer.retriever).run(query)
                except Exception as e:
                    st.error(f"Agentic mode failed: {e}")
                    st.stop()
            res = AnswerResult(query, None, sources, raw_answer=raw, verification=verification)
            res.answer_markdown = verification.markdown + "\n\n" + DISCLAIMER
            res.timings["total_s"] = time.perf_counter() - t0
            st.markdown(res.answer_markdown)
            if show_debug:
                render_sources(res)
            st.session_state.messages.append({"role": "assistant", "content": res.answer_markdown,
                                              "raw": raw, "result": res})
    else:
        answer_turn(query, history)

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last = st.session_state.messages[-1]
    fb = st.feedback("thumbs", key=f"fb_{len(st.session_state.messages)}")
    if fb is not None:
        r = last["result"]
        log_feedback({"ts": time.time(), "query": r.query, "thumbs_up": bool(fb),
                      "sources": [s.label for s in r.sources], "answer": r.raw_answer,
                      "abstained": r.abstained, "timings": r.timings})
        st.toast("Thanks, feedback saved to logs/feedback.jsonl")
