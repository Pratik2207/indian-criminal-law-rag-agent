"""Build data/section_map.csv (BNS<->IPC, BNSS<->CrPC) from the official
correspondence tables published by UP Police.

    python scripts/build_section_map.py
"""
import csv
import io
import re
from pathlib import Path

import pymupdf
import requests

BASE = "https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/"
TABLES = [
    ("BNS", "IPC", BASE + "202406281710564823BNS_IPC_Comparative.pdf"),
    ("BNSS", "CrPC", BASE + "202407031502194192BNSS2023-20-45.pdf"),
]
LEFT_RE = re.compile(r"\s*(\d{1,3}[A-Z]{0,3})\s*[\.(\s]")
RIGHT_RE = re.compile(r"(?:^|\n|to\s)\s*(\d{1,3}[A-Z]{0,3})\s*\.")
OUT = Path(__file__).resolve().parent.parent / "data" / "section_map.csv"


def parse_table(pdf_bytes: bytes) -> set[tuple[str, str]]:
    pairs, cur = set(), None
    for page in pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf"):
        for t in page.find_tables().tables:
            for row in t.extract():
                if len(row) < 2:
                    continue
                left, right = row[0] or "", row[1] or ""
                m = LEFT_RE.match(left)
                if m:
                    cur = m.group(1)
                if cur:
                    for mm in RIGHT_RE.finditer(right):
                        pairs.add((cur, mm.group(1)))
    return pairs


def main():
    rows = []
    for new_act, old_act, url in TABLES:
        pdf = requests.get(url, timeout=120).content
        pairs = parse_table(pdf)
        rows += [(new_act, n, old_act, o, url) for n, o in pairs]
        print(f"{new_act}<->{old_act}: {len(pairs)} pairs")
    key = lambda r: (r[0], int(re.match(r"\d+", r[1]).group()), r[1], r[2], r[3])  # noqa: E731
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["new_act", "new_section", "old_act", "old_section", "source"])
        w.writerows(sorted(set(rows), key=key))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
