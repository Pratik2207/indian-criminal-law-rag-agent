"""Citation linking and verification.

The generator cites sources with inline tags ([S1] for statute chunks, [W1] for
web/case-law results). This module:
  * turns tags into clickable markdown links,
  * flags tags that point to no source,
  * flags section numbers mentioned in the answer that were not retrieved, or
    that do not exist in the indexed Acts at all (hallucinated sections).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import quote_plus

from lawrag.config import get_settings
from lawrag.store import get_client

# "[S1]", also with a sub-section suffix the model sometimes adds: "[S3(2)]"
TAG_RE = re.compile(r"\[([SW])(\d+)(?:\([^)\]]{1,6}\))*\]")
TAG_LIST_RE = re.compile(r"\[((?:[SW]\d+(?:\(\w+\))*\s*[,;]\s*)+[SW]\d+(?:\(\w+\))*)\]")
# Small models cite loosely: "(S1)", "[S2, Section 307]", "[S1, S2: Section 420]". Any bracket/paren group
# that contains source ids is rewritten to canonical tags "[S1][S2]".
LOOSE_GROUP_RE = re.compile(r"[\[(]([^\[\]()\n]{0,80}?\b[SW]\d+\b[^\[\]()\n]{0,80}?)[\])]")
SOURCE_ID_RE = re.compile(r"\b([SW]\d+)\b")


def normalize_tags(text: str) -> str:
    text = TAG_LIST_RE.sub(lambda m: "".join(f"[{t.strip()}]" for t in re.split(r"[,;]", m.group(1))), text)

    def fix(m):
        if TAG_RE.fullmatch(m.group(0)):  # already canonical, e.g. "[S1]" or "[S3(2)]"
            return m.group(0)
        return "".join(f"[{i}]" for i in dict.fromkeys(SOURCE_ID_RE.findall(m.group(1))))

    return LOOSE_GROUP_RE.sub(fix, text)
ACT_WORDS = (r"BNSS|BNS|IPC|CrPC|Cr\.P\.C\.?|Bharatiya Nagarik Suraksha Sanhita|Bharatiya Nyaya Sanhita|"
             r"Indian Penal Code|Code of Criminal Procedure")
ACT_NORMAL = {"bnss": "BNSS", "bns": "BNS", "ipc": "IPC", "crpc": "CrPC", "bharatiyanagariksurakshasanhita": "BNSS",
              "bharatiyanyayasanhita": "BNS", "indianpenalcode": "IPC", "codeofcriminalprocedure": "CrPC"}
# "Section 103 of BNS", "s. 103(1) BNS", "Sections 378 and 379 IPC" (first number only), ...
MENTION_A = re.compile(rf"\b(?:sections?|sec\.?|s\.)\s*(\d{{1,3}}[A-Z]{{0,2}})(?:\(\w+\))*\s*(?:(?:of|in|under)\s+)?(?:the\s+)?({ACT_WORDS})", re.I)
# "BNS 103", "IPC Section 302", "BNS s.103"
MENTION_B = re.compile(rf"\b({ACT_WORDS})\s*(?:sections?|sec\.?|s\.)?\s*(\d{{1,3}}[A-Z]{{0,2}})\b(?!\s*(?:years|days|months|rupees|of\s+20))", re.I)


def norm_act(a: str) -> str:
    return ACT_NORMAL.get(re.sub(r"[^a-z]", "", a.lower()), a)


def statute_link(source: str, page: int) -> str:
    """Link to the exact page of the indexed PDF (served by Streamlit static serving)."""
    return f"app/static/{source}#page={page}"


def kanoon_link(act_name: str, section_no: str) -> str:
    return "https://indiankanoon.org/search/?formInput=" + quote_plus(f'"section {section_no}" "{act_name}"')


@dataclass
class Source:
    id: str  # "S1" / "W1"
    label: str  # "BNS s.103 — Punishment for murder" / page title
    url: str
    text: str
    kind: str  # statute | web | caselaw
    act: str = ""
    section_no: str = ""
    extra_urls: dict = field(default_factory=dict)
    header: str = ""  # full label shown to the LLM, e.g. "Bharatiya Nyaya Sanhita, 2023 — Section 103: ..."


@dataclass
class Verification:
    markdown: str
    used_ids: list[str]
    invalid_tags: list[str]
    mentions: list[tuple[str, str]]
    unsupported: list[tuple[str, str]]  # mentioned, exists in corpus, but not among the sources
    nonexistent: list[tuple[str, str]]  # mentioned, but no such section in the indexed Acts
    # sentence names a section of an Act but cites a different section of that same Act
    mismatched: list[tuple[tuple[str, str], str]] = field(default_factory=list)
    # claimed old<->new correspondence that contradicts the official table: ((old), (new))
    bad_mappings: list[tuple[tuple[str, str], tuple[str, str]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.invalid_tags or self.nonexistent or self.mismatched or self.bad_mappings)


@lru_cache
def known_sections() -> frozenset[tuple[str, str]]:
    client, s = get_client(), get_settings()
    out, offset = set(), None
    while True:
        pts, offset = client.scroll(s.qdrant_collection_name, limit=1000, offset=offset,
                                    with_payload=["act", "section_no"], with_vectors=False)
        out |= {(p.payload["act"], p.payload["section_no"]) for p in pts}
        if offset is None:
            return frozenset(out)


def section_mentions(text: str) -> list[tuple[str, str]]:
    found = []
    for m in MENTION_A.finditer(text):
        found.append((norm_act(m.group(2)), m.group(1).upper()))
    for m in MENTION_B.finditer(text):
        found.append((norm_act(m.group(1)), m.group(2).upper()))
    return list(dict.fromkeys(found))


def verify_and_link(answer: str, sources: list[Source]) -> Verification:
    by_id = {s.id: s for s in sources}
    text = normalize_tags(answer)
    used, invalid = [], []

    def link(m):
        sid = f"{m.group(1)}{m.group(2)}"
        src = by_id.get(sid)
        if not src:
            invalid.append(sid)
            return f"[{sid} ⚠]"
        if sid not in used:
            used.append(sid)
        short = src.label.split(" — ")[0]
        return f"[[{short}]]({src.url})"

    linked = TAG_RE.sub(link, text)
    mentions = section_mentions(text)
    retrieved = {(s.act, s.section_no) for s in sources if s.kind == "statute"}
    known = known_sections()
    unsupported = [m for m in mentions if m not in retrieved and m in known]
    nonexistent = [m for m in mentions if m not in known]
    return Verification(linked, used, invalid, mentions, unsupported, nonexistent, claim_mismatches(text, by_id),
                        bad_correspondences(text))


MAPPING_WORDS = re.compile(r"replac|correspond|equivalent|formerly|previously|earlier|renumber|now\b|old\b|new\b|"
                           r"called|known as|same as|maps? to|counterpart|substitut", re.I)
OLD_NEW = {"IPC": "BNS", "CrPC": "BNSS"}


def bad_correspondences(text: str):
    """E.g. "Section 499 IPC (now Section 353 of the BNS)" when the official table says IPC 499 -> BNS 356."""
    from lawrag.retrieval import equivalents

    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not MAPPING_WORDS.search(sent):
            continue
        ms = section_mentions(sent)
        for old in (m for m in ms if m[0] in OLD_NEW):
            official = [e for e in equivalents(*old) if e[0] == OLD_NEW[old[0]]]
            claimed = [m for m in ms if m[0] == OLD_NEW[old[0]]]
            if official and claimed and not any(c in official for c in claimed):
                for c in claimed:
                    if (old, c) not in out:
                        out.append((old, c))
    return out


def claim_mismatches(text: str, by_id: dict[str, Source]) -> list[tuple[tuple[str, str], str]]:
    """E.g. "Section 302 of the BNS ... [S1]" where S1 is BNS s.103."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        cited = [by_id[f"{k}{n}"] for k, n in TAG_RE.findall(sent)
                 if f"{k}{n}" in by_id and by_id[f"{k}{n}"].kind == "statute"]
        if not cited:
            continue
        cited_keys = {(s.act, s.section_no) for s in cited}
        for act, sec in section_mentions(sent):
            same_act = [s for s in cited if s.act == act]
            if same_act and (act, sec) not in cited_keys:
                item = ((act, sec), ", ".join(s.label.split(" — ")[0] for s in same_act))
                if item not in out:
                    out.append(item)
    return out
