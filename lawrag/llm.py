"""Thin client for the local Ollama server."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator

import requests

from lawrag.config import get_settings

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _payload(messages, stream: bool, fmt=None, temperature: float = 0.1, max_tokens: int | None = None):
    s = get_settings()
    body = {"model": s.ollama_model, "messages": messages, "stream": stream, "think": False,
            # num_predict caps runaway generations (small models occasionally loop); repeat_penalty discourages it
            "options": {"temperature": temperature, "num_ctx": s.ollama_num_ctx,
                        "num_predict": max_tokens or s.ollama_max_tokens, "repeat_penalty": 1.1}}
    if fmt is not None:
        body["format"] = fmt
    return body


def chat(messages: list[dict], fmt=None, temperature: float = 0.1, max_tokens: int | None = None) -> str:
    s = get_settings()
    r = requests.post(f"{s.ollama_base_url}/api/chat", json=_payload(messages, False, fmt, temperature, max_tokens),
                      timeout=300)
    r.raise_for_status()
    return THINK_RE.sub("", r.json()["message"]["content"]).strip()


def chat_stream(messages: list[dict], temperature: float = 0.1) -> Iterator[str]:
    s = get_settings()
    with requests.post(f"{s.ollama_base_url}/api/chat", json=_payload(messages, True, None, temperature),
                       stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
                if chunk.get("done"):
                    break


def complete(prompt: str, **kw) -> str:
    return chat([{"role": "user", "content": prompt}], **kw)
