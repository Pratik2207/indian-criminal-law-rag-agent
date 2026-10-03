"""Optional agentic mode (CrewAI).

The default app path (answer.py) is a deterministic retrieve -> generate pipeline.
This crew is for open-ended research questions where an agent deciding *what* to
search (several statute searches, section lookups, case law) helps.

Every tool call registers its results as numbered sources ([S#] statutes,
[W#] web/case law) in a SourceRegistry, so the final answer can be verified and
linked with the same citations.verify_and_link() as the pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CREWAI_TELEMETRY", "false")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.project import CrewBase, agent, crew, task  # noqa: E402
from crewai.tools import tool  # noqa: E402

from lawrag.answer import hits_to_sources, web_to_sources  # noqa: E402
from lawrag.citations import Source, verify_and_link  # noqa: E402
from lawrag.config import get_settings  # noqa: E402
from lawrag.retrieval import Retriever  # noqa: E402


class SourceRegistry:
    def __init__(self):
        self.sources: list[Source] = []

    def add_statutes(self, hits) -> list[Source]:
        known = {(s.act, s.section_no): s for s in self.sources if s.kind == "statute"}
        new = []
        for s in hits_to_sources(hits):
            if (s.act, s.section_no) in known:
                new.append(known[(s.act, s.section_no)])
                continue
            s.id = f"S{sum(x.kind == 'statute' for x in self.sources) + 1}"
            self.sources.append(s)
            new.append(s)
        return new

    def add_web(self, results) -> list[Source]:
        start = sum(x.kind != "statute" for x in self.sources) + 1
        new = web_to_sources(results, start)
        self.sources += new
        return new


def _fmt(sources: list[Source]) -> str:
    if not sources:
        return "No results."
    return "\n\n---\n\n".join(f"[{s.id}] {s.label}\nURL: {s.url}\n{s.text[:4000]}" for s in sources)


def build_tools(registry: SourceRegistry, retriever: Retriever):
    @tool("search_statutes")
    def search_statutes(query: str) -> str:
        """Search the BNS, BNSS, IPC and CrPC for sections relevant to a plain-English query.
        Returns numbered sources like [S1] with the section text and a link."""
        return _fmt(registry.add_statutes(retriever.expand(retriever.search(query, k=5))))

    @tool("lookup_section")
    def lookup_section(act: str, section_no: str) -> str:
        """Fetch the exact text of one section. act is one of BNS, BNSS, IPC, CrPC; section_no like '103' or '304A'.
        Also returns the equivalent section in the old/new code when one exists."""
        from lawrag.retrieval import equivalents

        hits = retriever.lookup(act.strip(), section_no.strip().upper())[:1]
        for eq in equivalents(act.strip(), section_no.strip().upper()):
            hits += retriever.lookup(*eq, match="mapped")[:1]
        return _fmt(registry.add_statutes(hits))

    @tool("search_case_law_and_web")
    def search_case_law_and_web(query: str) -> str:
        """Search Indian Kanoon judgments and trusted legal websites. Returns numbered sources like [W1] with URLs."""
        import lawrag.web_search as web_search

        return _fmt(registry.add_web(web_search.search(query)))

    return [search_statutes, lookup_section, search_case_law_and_web]


@CrewBase
class IndianLawRagCrew:
    """Researcher (tool use) -> legal writer (grounded, cited answer)."""

    agents_config = str(Path(__file__).parent / "prompts" / "agents.yaml")
    tasks_config = str(Path(__file__).parent / "prompts" / "tasks.yaml")

    def __init__(self, retriever: Retriever | None = None):
        s = get_settings()
        self.llm = LLM(model=f"ollama/{s.ollama_model}", base_url=s.ollama_base_url, temperature=0.1)
        self.registry = SourceRegistry()
        self.tools = build_tools(self.registry, retriever or Retriever(mode="full"))
        if not (s.serper_api_key or s.indian_kanoon_api_token):
            self.tools = self.tools[:2]

    @agent
    def research_agent(self) -> Agent:
        return Agent(config=self.agents_config["research_agent"], tools=self.tools, llm=self.llm,
                     allow_delegation=False, max_iter=6, verbose=True)

    @agent
    def legal_reasoning_agent(self) -> Agent:
        return Agent(config=self.agents_config["legal_reasoning_agent"], llm=self.llm,
                     allow_delegation=False, max_iter=3, verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def legal_reasoning_task(self) -> Task:
        return Task(config=self.tasks_config["legal_reasoning_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=True, tracing=False)

    def run(self, query: str):
        result = self.crew().kickoff(inputs={"query": query})
        raw = getattr(result, "raw", str(result))
        return raw, verify_and_link(raw, self.registry.sources), self.registry.sources


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the punishment for murder under BNS and what was it under IPC?"
    raw, verification, sources = IndianLawRagCrew().run(q)
    print(verification.markdown)
    print("\nnonexistent sections:", verification.nonexistent, "| invalid tags:", verification.invalid_tags)
