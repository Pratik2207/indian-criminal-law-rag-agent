"""Optional web and case-law search that keeps the source URL of every result.

* Serper (Google results), filtered to trusted Indian legal domains.
* Indian Kanoon API (judgments), https://api.indiankanoon.org/documentation/
Both return typed results so the answer can cite them as [W1](url).
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from lawrag.config import get_settings

log = logging.getLogger(__name__)

TRUSTED_DOMAINS = ("indiankanoon.org", "sci.gov.in", "main.sci.gov.in", "indiacode.nic.in", "egazette.gov.in",
                   "livelaw.in", "barandbench.com", "scobserver.in", "prsindia.org", "mha.gov.in", "pib.gov.in",
                   "lawcommissionofindia.nic.in", "ecourts.gov.in")


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    date: str = ""
    kind: str = "web"  # web | caselaw


def _trusted(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def serper_search(query: str, n: int = 5) -> list[WebResult]:
    key = get_settings().serper_api_key
    if not key:
        return []
    try:
        r = requests.post("https://google.serper.dev/search", timeout=20,
                          headers={"X-API-KEY": key, "Content-Type": "application/json"},
                          json={"q": f"{query} India criminal law", "gl": "in", "num": 20})
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("Serper search failed: %s", e)
        return []
    out = []
    for item in r.json().get("organic", []):
        url = item.get("link", "")
        if url and _trusted(url):
            out.append(WebResult(item.get("title", url), url, item.get("snippet", ""), item.get("date", "")))
    return out[:n]


def kanoon_search(query: str, n: int = 5) -> list[WebResult]:
    token = get_settings().indian_kanoon_api_token
    if not token:
        return []
    try:
        r = requests.post("https://api.indiankanoon.org/search/", timeout=30,
                          headers={"Authorization": f"Token {token}", "Accept": "application/json"},
                          params={"formInput": query, "pagenum": 0})
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("Indian Kanoon search failed: %s", e)
        return []
    out = []
    for d in r.json().get("docs", [])[:n]:
        strip = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or ""))  # noqa: E731
        out.append(WebResult(strip(d.get("title")), f"https://indiankanoon.org/doc/{d['tid']}/",
                             strip(d.get("headline")), d.get("publishdate") or d.get("docsource", ""),
                             kind="caselaw"))
    return out


def search(query: str, n: int = 5) -> list[WebResult]:
    results = kanoon_search(query, n=3) + serper_search(query, n=n)
    seen, out = set(), []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out[:n]
