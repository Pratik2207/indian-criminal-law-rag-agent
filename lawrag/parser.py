"""Structure-aware parser for the statute PDFs in data/statutes/.

Splits each Act into sections with metadata (act, section number, title,
chapter, page), so chunks never straddle two sections and every chunk can be
cited and linked precisely.

Two PDF layouts are handled:
  * "gazette"   (BNS, BNSS): body in the centre column, section titles printed
                 as margin notes beside the section number.
  * "indiacode" (IPC, CrPC): "302. Punishment for murder.—Whoever ..." with an
                 "Arrangement of Sections" table of contents up front and
                 amendment footnotes at the bottom of pages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

ACTS = {
    "BNS_2023.pdf": {
        "act": "BNS",
        "name": "Bharatiya Nyaya Sanhita, 2023",
        "layout": "gazette",
        "act_url": "https://www.indiacode.nic.in/indiacode/handle/123456789/20062",
    },
    "BNSS_2023.pdf": {
        "act": "BNSS",
        "name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "layout": "gazette",
        "act_url": "https://www.indiacode.nic.in/handle/123456789/21419",
    },
    "IPC_1860.pdf": {
        "act": "IPC",
        "name": "Indian Penal Code, 1860",
        "layout": "indiacode",
        "act_url": "https://www.indiacode.nic.in/handle/123456789/2263?sam_handle=123456789/1362",
    },
    "CrPC_1973.pdf": {
        "act": "CrPC",
        "name": "Code of Criminal Procedure, 1973",
        "layout": "indiacode",
        "act_url": "https://indiacode.nic.in/handle/123456789/1611?sam_handle=123456789%2F1362",
    },
}

SECTION_RE = re.compile(r"^\s*(?:\d+\*?\[)?(\d{1,3})([A-Z]{0,3})\.(?:\s|(?=[A-Z(]))")
CHAPTER_RE = re.compile(r"^\s*CHAPTER\s*([IVXLC]+[A-Z]?)\b")
SCHEDULE_RE = re.compile(r"^\s*THE\s+(FIRST|SECOND)\s+SCHEDULE", re.I)
FOOTNOTE_RE = re.compile(
    r"^\s*\d+\.\s+(Subs\.|Ins\.|Omitted|Rep\.|The words|Added|Certain words|"
    r"Sub-section|Cl\.|Original|Now see|See |w\.e\.f|Renumbered|Section|Clause|"
    r"Proviso|Explanation|Illustration|Inserted|Substituted|The brackets)"
)
AMEND_MARK_RE = re.compile(r"\d+\*?\[")


@dataclass
class Section:
    act: str
    act_name: str
    section_no: str  # e.g. "103", "304A", "Schedule I"
    title: str
    chapter: str
    page: int  # 1-based PDF page where the section starts
    source: str  # PDF filename
    act_url: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        t = "\n".join(self.lines)
        t = AMEND_MARK_RE.sub("", t)
        t = re.sub(r"[ \t]+", " ", t)
        return re.sub(r"\n{3,}", "\n\n", t).strip()


def _accept(num: int, prev: int) -> bool:
    """Section numbers increase monotonically; small gaps allowed for repealed ones."""
    return prev <= num <= prev + 6


def _clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(".—– ").strip()


# ---------------------------------------------------------------- gazette
def _is_noise_block(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
    return ascii_ratio < 0.9 or "GAZETTE OF INDIA" in text or "xxxGID" in text


def _parse_gazette(doc, meta: dict, source: str) -> list[Section]:
    sections: list[Section] = []
    cur: Section | None = None
    prev = 0
    chapter = ""
    pending_chapter = False
    in_schedule = False
    for pno, page in enumerate(doc, start=1):
        w = page.rect.width
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        margin = [b for b in blocks if b[0] < 0.13 * w or b[0] > 0.80 * w]
        body = sorted(
            (b for b in blocks if b not in margin and not _is_noise_block(b[4])),
            key=lambda b: b[1],
        )
        # A block can contain the end of one section and the start of the next:
        # split it at section starts, estimating each piece's y from its line offset.
        pieces = []
        for b in body:
            lines = b[4].strip("\n").split("\n")
            lh = (b[3] - b[1]) / max(len(lines), 1)
            start = 0
            for i in range(1, len(lines)):
                if SECTION_RE.match(lines[i]):
                    pieces.append((b[1] + start * lh, "\n".join(lines[start:i])))
                    start = i
            pieces.append((b[1] + start * lh, "\n".join(lines[start:])))
        for y, text in pieces:
            text = text.strip()
            if not text or set(text) <= set("_- \n"):
                continue
            if SCHEDULE_RE.match(text) and prev > 1:
                in_schedule = True
                cur = Section(meta["act"], meta["name"], "Schedule " + ("I" if "FIRST" in text.upper() else "II"),
                              "First Schedule — Classification of offences" if "FIRST" in text.upper() else "Second Schedule — Forms",
                              "", pno, source, meta["act_url"])
                sections.append(cur)
                continue
            if in_schedule:
                cur.lines.append(text)
                continue
            m_ch = CHAPTER_RE.match(text)
            if m_ch:
                chapter = "Chapter " + m_ch.group(1)
                pending_chapter = True
                continue
            if pending_chapter and text.isupper():
                chapter = f"{chapter} — {_clean_title(text).title()}"
                pending_chapter = False
                continue
            pending_chapter = False
            m = SECTION_RE.match(text)
            if m and _accept(int(m.group(1)), prev):
                num = int(m.group(1))
                prev = num
                title = ""
                near = sorted(margin, key=lambda mb: abs(mb[1] - y))
                if near and abs(near[0][1] - y) < 16:
                    title = _clean_title(near[0][4])
                cur = Section(meta["act"], meta["name"], f"{num}{m.group(2)}", title, chapter,
                              pno, source, meta["act_url"], [text])
                sections.append(cur)
            elif cur is not None:
                cur.lines.append(text)
    return sections


# --------------------------------------------------------------- indiacode
def _parse_indiacode(doc, meta: dict, source: str) -> list[Section]:
    sections: list[Section] = []
    cur: Section | None = None
    prev = 0
    chapter = ""
    started = False
    for pno, page in enumerate(doc, start=1):
        lines = page.get_text().splitlines()
        # drop footnotes (bottom of page) and bare page numbers
        for i, ln in enumerate(lines):
            if FOOTNOTE_RE.match(ln) and started:
                lines = lines[:i]
                break
        for idx, ln in enumerate(lines):
            s = ln.strip()
            if not s or s.isdigit():
                continue
            if started and prev > 1 and SCHEDULE_RE.match(s):
                first = "FIRST" in s.upper()
                cur = Section(meta["act"], meta["name"], "Schedule " + ("I" if first else "II"),
                              "First Schedule — Classification of offences" if first else "Second Schedule — Forms",
                              "", pno, source, meta["act_url"])
                sections.append(cur)
                prev = 10**6  # no more numbered sections after the schedules start
                continue
            m_ch = CHAPTER_RE.match(s)
            if m_ch and started:
                nxt = next((l.strip() for l in lines[idx + 1: idx + 3] if l.strip()), "")
                chapter = f"Chapter {m_ch.group(1)}" + (f" — {_clean_title(nxt).title()}" if nxt.isupper() else "")
                continue
            m = SECTION_RE.match(s)
            if m:
                lookahead = " ".join(l.strip() for l in lines[idx: idx + 4])
                has_dash = re.search(r"\.?\s*[—–-]{1,2}", lookahead[: 320]) and "—" in lookahead[:320]
                num = int(m.group(1))
                if not started and num == 1 and has_dash:
                    started = True
                if started and has_dash and _accept(num, prev):
                    prev = num
                    head = lookahead.split("—", 1)[0]
                    title = SECTION_RE.sub("", AMEND_MARK_RE.sub("", head), count=1)
                    cur = Section(meta["act"], meta["name"], f"{num}{m.group(2)}", _clean_title(title),
                                  chapter, pno, source, meta["act_url"], [s])
                    sections.append(cur)
                    continue
            if started and cur is not None:
                cur.lines.append(s)
    return sections


def parse_pdf(path: str | Path) -> list[Section]:
    path = Path(path)
    meta = ACTS[path.name]
    doc = pymupdf.open(path)
    if meta["layout"] == "gazette":
        return _parse_gazette(doc, meta, path.name)
    return _parse_indiacode(doc, meta, path.name)


def parse_all(knowledge_dir: str | Path) -> list[Section]:
    out: list[Section] = []
    for pdf in sorted(Path(knowledge_dir).glob("*.pdf")):
        if pdf.name in ACTS:
            out.extend(parse_pdf(pdf))
    return out


if __name__ == "__main__":
    import collections
    import sys

    secs = parse_all(sys.argv[1] if len(sys.argv) > 1 else "data/statutes")
    print(collections.Counter(s.act for s in secs))
    for s in secs[:3]:
        print(s.act, s.section_no, s.title, s.chapter, s.page, s.text[:120].replace("\n", " "))
