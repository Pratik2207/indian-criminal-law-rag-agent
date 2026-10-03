"""End-to-end answering: retrieve -> (abstain?) -> grounded generation -> verify + link citations."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from lawrag.citations import TAG_RE, Source, Verification, kanoon_link, statute_link, verify_and_link
from lawrag.config import get_settings
from lawrag.llm import chat, chat_stream
from lawrag.retrieval import Hit, QueryAnalysis, Retriever, analyze, equivalents

log = logging.getLogger(__name__)

DISCLAIMER = ("_This is legal information generated from the statute texts, not legal advice. "
              "Verify against the official text and consult a lawyer for your situation._")
ABSTAIN_MSG = ("I couldn't find provisions in the indexed Acts (BNS, BNSS, IPC, CrPC) that answer this. "
               "The question may fall outside Indian criminal law, or may need case law or another statute.")

SYSTEM = """You are a careful Indian criminal-law research assistant.
Answer ONLY from the numbered sources provided. Rules:
1. End every factual sentence with the tag of the source that supports it, written exactly like [S1] (square brackets, one tag per bracket, e.g. [S1][S3]). Use only tags that appear in the sources.
   Format example (placeholders, not real law): "Under Section <number> of the <Act>, <what that source says> [S2]."
2. Always name the Act and section in words too (e.g. "Section 103 of the BNS").
3. Never mention a section number that is not in the sources. If the sources don't answer the question, say so plainly.
4. The BNS and BNSS replaced the IPC and CrPC for offences/proceedings from 1 July 2024. When both old and new provisions are in the sources, give the new one first and mention the old equivalent. Only state that an old section corresponds to a new one if that pair is listed under "Official section correspondence"; never guess a correspondence.
5. Be concise: a direct answer first, then key details (ingredients, punishment, bailable/cognizable if stated). Use short paragraphs or bullets.
6. Do not give personal legal advice."""


@dataclass
class AnswerResult:
    query: str
    analysis: QueryAnalysis | None
    sources: list[Source]
    answer_markdown: str = ""
    raw_answer: str = ""
    verification: Verification | None = None
    abstained: bool = False
    timings: dict = field(default_factory=dict)
    llm_query: str = ""  # task sent to the LLM when it differs from the user's question

    @property
    def task(self) -> str:
        return self.llm_query or self.analysis.query


OLD_TO_NEW = {"IPC": "BNS", "CrPC": "BNSS"}
NEW_TO_OLD = {v: k for k, v in OLD_TO_NEW.items()}


def is_mapping_question(analysis: QueryAnalysis | None) -> bool:
    """A named section plus the other code of its pair, e.g. "What is IPC 302 called in BNS?"."""
    if not analysis or not analysis.refs or not analysis.mapped:
        return False
    pairs = {**OLD_TO_NEW, **NEW_TO_OLD}
    return any(pairs.get(act) in analysis.acts for act, _ in analysis.refs)


def mapping_task(analysis: QueryAnalysis, hits: list[Hit]) -> str:
    titles = {(h.act, h.section_no): h.title for h in hits}
    name = lambda a, n: f"Section {n} of the {a}" + (f" ({titles[(a, n)]})" if titles.get((a, n)) else "")  # noqa: E731
    facts = [f"{name(*ref)} corresponds to " + " and ".join(name(*e) for e in equivalents(*ref))
             for ref in analysis.refs if equivalents(*ref)]
    return ("Fact from the official correspondence table: " + "; ".join(facts) + ". "
            "State this correspondence in one sentence, then briefly explain what the new section provides, "
            "including the punishment if the source states it. "
            f"(The user asked: {analysis.query})")


def hits_to_sources(hits: list[Hit]) -> list[Source]:
    out = []
    for i, h in enumerate(hits, 1):
        title = f" — {h.title}" if h.title else ""
        out.append(Source(
            id=f"S{i}", label=f"{h.cite}{title}", url=statute_link(h.source, h.page), text=h.text, kind="statute",
            act=h.act, section_no=h.section_no, header=h.header,
            extra_urls={"India Code (official Act page)": h.act_url,
                        "Indian Kanoon (section & case law)": kanoon_link(h.header.split(" — ")[0], h.section_no)},
        ))
    return out


def web_to_sources(results, start: int = 1) -> list[Source]:
    return [Source(id=f"W{i}", label=r.title, url=r.url, text=f"{r.snippet} {r.date}".strip(), kind=r.kind)
            for i, r in enumerate(results, start)]


def correspondence_notes(sources: list[Source], extra: list[str] | None = None) -> list[str]:
    """Official old<->new equivalents of every retrieved section, so the model never has to guess them."""
    notes = list(extra or [])
    for s in sources:
        if s.kind != "statute":
            continue
        eqs = equivalents(s.act, s.section_no)
        if eqs:
            note = f"{s.act} s.{s.section_no} corresponds to " + ", ".join(f"{a} s.{n}" for a, n in eqs)
            if note not in notes:
                notes.append(note)
    return notes


def build_messages(query: str, sources: list[Source], history: list[dict] | None = None,
                   notes: list[str] | None = None) -> list[dict]:
    blocks = []
    for s in sources:
        head = (s.header or s.label) if s.kind == "statute" else f"{s.label} ({s.url})"
        blocks.append(f"[{s.id}] {head}\n{s.text}")
    msgs = [{"role": "system", "content": SYSTEM}]
    # Earlier turns' [S#] tags refer to earlier sources; strip them so they can't be re-used for these sources.
    msgs += [{"role": m["role"], "content": TAG_RE.sub("", m["content"])} for m in (history or [])[-4:]]
    head = ""
    if notes:  # placed first: small models attend most to the start of the prompt
        head = "Official section correspondence (old code <-> new code): " + "; ".join(notes) + ".\n\n"
    msgs.append({"role": "user", "content": head + "Sources:\n\n" + "\n\n---\n\n".join(blocks)
                 + f"\n\nQuestion: {query}"})
    return msgs


def correspondence_banner(res: "AnswerResult") -> str:
    """Deterministic old<->new mapping for sections named in the question, taken from the official
    table rather than generated, because small models get these lookups wrong."""
    if not res.analysis or not res.analysis.refs:
        return ""
    by_key = {(s.act, s.section_no): s for s in res.sources if s.kind == "statute"}

    def fmt(act, sec):
        s = by_key.get((act, sec))
        title = f" ({s.label.split(' — ', 1)[1]})" if s and " — " in s.label else ""
        return f"[{act} s.{sec}]({s.url}){title}" if s else f"{act} s.{sec}"

    rows = [f"{fmt(*ref)} → " + ", ".join(fmt(*e) for e in equivalents(*ref))
            for ref in res.analysis.refs if equivalents(*ref)]
    if not rows:
        return ""
    return "> 📌 **Official correspondence** (from the government correspondence table): " + "; ".join(rows) + "\n\n"


class Answerer:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever(mode="full")
        self.settings = get_settings()

    def prepare(self, query: str, use_web: bool = False) -> AnswerResult:
        t0 = time.perf_counter()
        analysis = analyze(query)
        hits = self.retriever.expand(self.retriever.search(query, k=self.settings.top_k, analysis=analysis))
        llm_query = ""
        if is_mapping_question(analysis):
            # The lookup is answered from the official table; the LLM only explains the sections involved,
            # which small models do reliably (they get "IPC 302 is called ...?" wrong on their own).
            hits = [h for h in hits if h.match != "semantic"]
            llm_query = mapping_task(analysis, hits)
        res = AnswerResult(query, analysis, hits_to_sources(hits), llm_query=llm_query)
        res.timings["retrieval_s"] = time.perf_counter() - t0
        exact = any(h.match != "semantic" for h in hits)
        top = max((h.score for h in hits if h.match == "semantic"), default=float("-inf"))
        if not exact and top < self.settings.abstain_threshold:
            res.abstained = True
        if use_web:
            import lawrag.web_search as web_search

            t1 = time.perf_counter()
            res.sources += web_to_sources(web_search.search(analysis.query))
            res.timings["web_s"] = time.perf_counter() - t1
            if any(s.kind != "statute" for s in res.sources):
                res.abstained = False
        return res

    def finish(self, res: AnswerResult, raw: str) -> AnswerResult:
        res.raw_answer = raw
        res.verification = verify_and_link(raw, res.sources)
        md = res.verification.markdown
        if res.verification.nonexistent:
            md += ("\n\n> ⚠ **Citation check:** the answer mentions "
                   + ", ".join(f"{a} s.{n}" for a, n in res.verification.nonexistent)
                   + ", which does not exist in the indexed Acts. Treat it as an error.")
        if res.verification.mismatched:
            md += ("\n\n> ⚠ **Citation check:** " + "; ".join(
                f"the text names {a} s.{n} but cites {c}" for (a, n), c in res.verification.mismatched)
                + ". Check these against the sources.")
        if res.verification.bad_mappings:
            md += ("\n\n> ⚠ **Correspondence check:** " + "; ".join(
                f"{o[0]} s.{o[1]} does not correspond to {n[0]} s.{n[1]}"
                + (f" (official: {', '.join(f'{a} s.{s}' for a, s in equivalents(*o))})" if equivalents(*o) else "")
                for o, n in res.verification.bad_mappings) + ".")
        if res.verification.unsupported:
            md += ("\n\n> ℹ Mentioned but not among the retrieved sources (unverified here): "
                   + ", ".join(f"{a} s.{n}" for a, n in res.verification.unsupported))
        res.answer_markdown = correspondence_banner(res) + md + "\n\n" + DISCLAIMER
        return res

    def answer(self, query: str, use_web: bool = False, history: list[dict] | None = None) -> AnswerResult:
        res = self.prepare(query, use_web)
        if res.abstained:
            res.answer_markdown = ABSTAIN_MSG + "\n\n" + DISCLAIMER
            return res
        t0 = time.perf_counter()
        raw = chat(build_messages(res.task, res.sources, history,
                                  correspondence_notes(res.sources, res.analysis.correspondences)))
        res.timings["generation_s"] = time.perf_counter() - t0
        return self.finish(res, raw)

    def stream(self, res: AnswerResult, history: list[dict] | None = None) -> Iterator[str]:
        """Yield raw tokens; call finish() with the joined text afterwards."""
        yield from chat_stream(build_messages(res.task, res.sources, history,
                                              correspondence_notes(res.sources, res.analysis.correspondences)))
