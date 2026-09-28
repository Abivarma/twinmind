"""LLM providers: Claude API (default) or a fully local Ollama model."""
from __future__ import annotations

import json
import logging
import re
from typing import Iterator

log = logging.getLogger("twin.llm")

_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLM:
    name = "base"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 4000,
                 live: bool = False) -> str:
        raise NotImplementedError

    def stream(self, system: str, messages: list[dict], max_tokens: int = 8000) -> Iterator[str]:
        yield self.complete(system, messages, max_tokens)


class ClaudeLLM(LLM):
    name = "anthropic"
    _FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5-1")

    def __init__(self, cfg: dict):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = cfg["model"]
        self.live_model = cfg.get("live_model") or cfg["model"]
        self.live_effort = cfg.get("live_effort", "low")
        self.deep_effort = cfg.get("deep_effort", "medium")
        self.fallbacks = cfg.get("fallbacks", True)

    def _kwargs(self, model: str, effort: str) -> dict:
        kw: dict = {"model": model, "output_config": {"effort": effort}}
        if self.fallbacks and model in self._FALLBACK_MODELS:
            # Server-side refusal fallbacks: re-runs a declined request on a fallback model.
            kw["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
            kw["extra_body"] = {"fallbacks": "default"}
        return kw

    def complete(self, system, messages, max_tokens=4000, live=False):
        model = self.live_model if live else self.model
        effort = self.live_effort if live else self.deep_effort
        try:
            resp = self.client.messages.create(
                max_tokens=max_tokens, system=system, messages=messages,
                **self._kwargs(model, effort))
        except self.anthropic.RateLimitError:
            log.warning("Claude rate limited")
            return ""
        except self.anthropic.APIStatusError as e:
            log.error("Claude API error %s: %s", e.status_code, e.message)
            return ""
        except self.anthropic.APIConnectionError as e:
            log.error("Claude connection error: %s", e)
            return ""
        except TypeError as e:  # raised by the SDK when no credentials are configured
            log.error("Claude not configured (set ANTHROPIC_API_KEY or run `ant auth login`): %s", e)
            return ""
        if resp.stop_reason == "refusal":
            log.warning("Claude declined the request")
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")

    def stream(self, system, messages, max_tokens=16000):
        try:
            with self.client.messages.stream(
                max_tokens=max_tokens, system=system, messages=messages,
                **self._kwargs(self.model, self.deep_effort)) as s:
                yield from s.text_stream
        except self.anthropic.APIError as e:
            log.error("Claude stream error: %s", e)
            yield f"\n\n[error talking to Claude: {e}]"
        except TypeError:
            yield "\n\n[Claude is not configured: set ANTHROPIC_API_KEY or run `ant auth login`, or switch to Ollama in config.toml]"


class OllamaLLM(LLM):
    """Local model via Ollama's native /api/chat. Private mode: nothing leaves the Mac."""

    name = "ollama"

    def __init__(self, cfg: dict):
        import httpx

        self.httpx = httpx
        self.url = cfg["ollama_url"].rstrip("/") + "/api/chat"
        self.model = cfg["ollama_model"]

    def _body(self, system, messages, max_tokens, stream):
        return {"model": self.model, "stream": stream,
                "messages": [{"role": "system", "content": system}, *messages],
                "options": {"num_predict": max_tokens, "num_ctx": 32768}}

    def complete(self, system, messages, max_tokens=4000, live=False):
        try:
            r = self.httpx.post(self.url, json=self._body(system, messages, max_tokens, False),
                                timeout=300)
            r.raise_for_status()
            return _THINK.sub("", r.json()["message"]["content"]).strip()
        except self.httpx.HTTPError as e:
            log.error("Ollama error: %s", e)
            return ""

    def stream(self, system, messages, max_tokens=8000):
        in_think = False
        try:
            with self.httpx.stream("POST", self.url, timeout=300,
                                   json=self._body(system, messages, max_tokens, True)) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    piece = json.loads(line).get("message", {}).get("content", "")
                    if "<think>" in piece:
                        in_think = True
                    if not in_think:
                        yield piece
                    if "</think>" in piece:
                        in_think = False
        except self.httpx.HTTPError as e:
            yield f"\n\n[error talking to Ollama: {e}]"


def make_llm(cfg: dict) -> LLM:
    provider = cfg["llm"]["provider"]
    if provider == "ollama":
        return OllamaLLM(cfg["llm"])
    return ClaudeLLM(cfg["llm"])


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction (models sometimes wrap JSON in prose or fences)."""
    text = _THINK.sub("", text or "")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
