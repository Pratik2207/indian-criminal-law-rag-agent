"""Web UI server: a small chat page (web/) plus a JSON/streaming API over lawrag.

    python server.py            # open http://127.0.0.1:8000

API keys stay in .env on the server. The browser only learns whether web search is
available, never the keys themselves.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("CREWAI_TELEMETRY", "false")
# Show CrewAI's live output (crew/task/agent/tool panels) in any terminal: UTF-8 for the emoji titles,
# line buffering so text appears as it happens even when stdout is a pipe (Git Bash, IDE consoles).
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from markdown_it import MarkdownIt  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from lawrag.config import ROOT, get_settings  # noqa: E402

log = logging.getLogger("lawrag.server")
trace = logging.getLogger("lawrag.trace")  # per-question reasoning trace printed in the server terminal
if not trace.handlers:  # set here, not in __main__, so it also prints under `uvicorn server:app`
    _h = logging.StreamHandler(sys.stdout)  # same stream as CrewAI's panels, so lines stay in order
    _h.setFormatter(logging.Formatter("  %(asctime)s  %(message)s", "%H:%M:%S"))
    trace.addHandler(_h)
    trace.setLevel(logging.INFO)
    trace.propagate = False
WEB = ROOT / "web"
STATIC = ROOT / "static"
FEEDBACK_LOG = ROOT / "logs" / "feedback.jsonl"
MD = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")

_answerer = None
_ready = threading.Event()
_load_error: str | None = None
# Embedded Qdrant and the 4 GB GPU are both single-user; answer one question at a time.
_busy = threading.Lock()


def _load():
    global _answerer, _load_error
    try:
        from lawrag.answer import Answerer

        # Citations link to app/static/<pdf>#page=N, so serve the statute PDFs there.
        STATIC.mkdir(exist_ok=True)
        for pdf in Path(get_settings().knowledge_dir).glob("*.pdf"):
            if not (STATIC / pdf.name).exists():
                shutil.copy(pdf, STATIC / pdf.name)
        _answerer = Answerer()
        log.info("models and index loaded")
    except Exception as e:  # shown in the UI instead of a stack trace
        log.exception("startup failed")
        _load_error = str(e)
    finally:
        _ready.set()


def _web_available() -> bool:
    s = get_settings()
    return bool(s.serper_api_key or s.indian_kanoon_api_token)


def _event(kind: str, **data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _result_payload(res) -> dict:
    cited = set(res.verification.used_ids) if res.verification else set()
    sources = [{
        "id": s.id, "label": s.label, "url": s.url, "kind": s.kind, "cited": s.id in cited,
        "text": s.text[:1200], "links": s.extra_urls,
    } for s in res.sources]
    return {"html": MD.render(res.answer_markdown), "raw": res.raw_answer, "sources": sources,
            "abstained": res.abstained, "timings": {k: round(v, 1) for k, v in res.timings.items()}}


def _trace_pipeline(res):
    """Print the default pipeline's decisions to the terminal (the browser shows only the answer)."""
    a = res.analysis
    if a.query != res.query:
        trace.info("Translate %r -> %r", res.query, a.query)
    trace.info("Analyze  acts %s | refs %s | mapped %s | search text %r", a.acts or "-",
               [f"{x} {n}" for x, n in a.refs] or "-", [f"{x} {n}" for x, n in a.mapped] or "-", a.semantic_query)
    for s in res.sources:
        trace.info("Source   [%s] %s", s.id, s.label)
    if res.llm_query:
        trace.info("Decide   mapping question: answered from the official table, model explains only")
    trace.info("Decide   %s (retrieval %.1fs)", "ABSTAIN, no model call" if res.abstained else "answer with the model",
               res.timings.get("retrieval_s", 0))


def _describe_step(step) -> dict | None:
    from crewai.agents.parser import AgentAction

    if not isinstance(step, AgentAction):
        return None
    try:
        args = json.loads(step.tool_input) if isinstance(step.tool_input, str) else step.tool_input
    except ValueError:
        args = step.tool_input
    if step.tool == "lookup_section" and isinstance(args, dict):
        text = f"Looked up {args.get('act', '')} section {args.get('section_no', '')}"
    elif isinstance(args, dict) and "query" in args:
        verb = "Searched case law and the web" if "web" in step.tool else "Searched the statutes"
        text = f"{verb} for “{args['query']}”"
    else:
        text = f"Used {step.tool}"
    thought = (step.thought or "").removeprefix("Thought:").strip()
    return {"text": text, "thought": thought[:300]}


def _answer_stream(query: str, history: list[dict], use_web: bool):
    from lawrag.answer import ABSTAIN_MSG, DISCLAIMER

    t0 = time.perf_counter()
    yield _event("status", text="Searching the Acts…")
    res = _answerer.prepare(query, use_web=use_web)
    _trace_pipeline(res)
    n = len(res.sources)
    yield _event("status", text=f"Found {n} source{'s' if n != 1 else ''}. Writing the answer…")
    if res.abstained:
        res.answer_markdown = ABSTAIN_MSG + "\n\n" + DISCLAIMER
    else:
        buf = ""
        for tok in _answerer.stream(res, history):
            buf += tok
            yield _event("token", text=tok)
        _answerer.finish(res, buf)
        v = res.verification
        trace.info("Verify   cited %s | invalid tags %s | nonexistent %s | mismatched %s | bad mappings %s",
                   v.used_ids or "-", v.invalid_tags or "-", v.nonexistent or "-", v.mismatched or "-",
                   v.bad_mappings or "-")
    res.timings["total_s"] = time.perf_counter() - t0
    trace.info("Done     %s", ", ".join(f"{k} {v:.1f}s" for k, v in res.timings.items()))
    yield _event("done", **_result_payload(res))


def _agent_stream(query: str):
    from lawrag.answer import DISCLAIMER, AnswerResult
    from lawrag.crew import IndianLawRagCrew

    t0 = time.perf_counter()
    q: queue.Queue = queue.Queue()
    out: dict = {}

    def on_step(step):
        info = _describe_step(step)
        if info:
            trace.info("Thought  %s", info["thought"] or "-")
            trace.info("Action   %s %s", step.tool, step.tool_input)
            trace.info("Observe  %s", " | ".join(ln for ln in str(step.result).splitlines() if ln.startswith("["))
                       or str(step.result)[:200])
            q.put(("step", info))
        elif type(step).__name__ == "AgentFinish":
            trace.info("Finish   %s", str(step.output)[:200].replace("\n", " "))

    def work():
        try:
            out["result"] = IndianLawRagCrew(_answerer.retriever).run(query, step_callback=on_step)
        except Exception as e:
            out["error"] = e
        finally:
            q.put(("end", None))

    yield _event("status", text="Researching…")
    threading.Thread(target=work, daemon=True).start()
    while True:
        kind, info = q.get()
        if kind == "end":
            break
        yield _event("step", **info)
    if "error" in out:
        raise out["error"]
    raw, verification, sources = out["result"]
    res = AnswerResult(query, None, sources, raw_answer=raw, verification=verification)
    res.answer_markdown = verification.markdown + "\n\n" + DISCLAIMER
    res.timings["total_s"] = time.perf_counter() - t0
    yield _event("done", **_result_payload(res))


def _guarded(gen):
    """Serialize requests and turn failures into an 'error' event the page can show."""
    _ready.wait()
    if _load_error:
        yield _event("error", text=f"The server could not start: {_load_error}")
        return
    if not _busy.acquire(blocking=False):
        yield _event("status", text="Waiting for the previous question to finish…")
        _busy.acquire()
    try:
        yield from gen()
    except Exception as e:
        log.exception("answer failed")
        msg = str(e)
        if "11434" in msg or "Connection" in type(e).__name__ or "connect" in msg.lower():
            msg = f"Couldn't reach the language model. Check that Ollama is running with {get_settings().ollama_model}."
        yield _event("error", text=msg)
    finally:
        _busy.release()


# ---------------------------------------------------------------- routes
async def index(request: Request):
    return FileResponse(WEB / "index.html")


async def status(request: Request):
    return JSONResponse({"ready": _ready.is_set() and not _load_error, "error": _load_error,
                         "model": get_settings().ollama_model, "web_available": _web_available()})


async def chat(request: Request):
    body = await request.json()
    query = str(body.get("query", "")).strip()[:2000]
    if not query:
        return JSONResponse({"error": "Empty question."}, status_code=400)
    history = [{"role": m["role"], "content": str(m["content"])} for m in body.get("history", [])[-8:]
               if m.get("role") in ("user", "assistant")]
    use_web = bool(body.get("web")) and _web_available()
    trace.info("─" * 60)
    trace.info("Question %r  [mode: %s]", query, "agent" if body.get("agent") else "web" if use_web else "default")
    if body.get("agent"):
        gen = lambda: _agent_stream(query)  # noqa: E731
    else:
        gen = lambda: _answer_stream(query, history, use_web)  # noqa: E731
    return StreamingResponse(_guarded(gen), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def feedback(request: Request):
    body = await request.json()
    entry = {"ts": time.time(), "query": str(body.get("query", ""))[:2000], "thumbs_up": bool(body.get("up")),
             "mode": str(body.get("mode", "")), "sources": body.get("sources", [])[:20],
             "answer": str(body.get("answer", ""))[:20000]}
    FEEDBACK_LOG.parent.mkdir(exist_ok=True)
    with FEEDBACK_LOG.open("a", encoding="utf8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return JSONResponse({"ok": True})


@contextlib.asynccontextmanager
async def lifespan(app):
    # Load in the background so the page opens at once and shows "Loading…" until ready.
    threading.Thread(target=_load, daemon=True).start()
    yield


STATIC.mkdir(exist_ok=True)
app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/status", status),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/feedback", feedback, methods=["POST"]),
        Mount("/assets", StaticFiles(directory=WEB), name="assets"),
        Mount("/app/static", StaticFiles(directory=STATIC), name="statutes"),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    # 127.0.0.1: reachable from this computer only
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
