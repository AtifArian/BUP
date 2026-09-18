"""Offline tests for the multi-provider model chain in app/llm.py (no network)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402


class FakeResponse:
    def __init__(self, status, content=None, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    monkeypatch.setenv("GROQ_MODEL", "g1")
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "g2")
    monkeypatch.setenv("GEMINI_MODEL", "gem1")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("LLM_MODEL_CHAIN", raising=False)
    monkeypatch.setattr(llm, "_RATE_LIMITED", {})
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_chain_order_and_provider_skipping(monkeypatch):
    assert [m.id for m in llm.chain()] == ["groq:g1", "groq:g2", "gemini:gem1"]
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,groq")
    assert [m.id for m in llm.chain()][0] == "gemini:gem1"
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert [m.id for m in llm.chain()] == ["gemini:gem1"]
    assert llm.chain()[0].url.endswith("/v1beta/openai/chat/completions")


def test_explicit_chain_interleaves_providers(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_CHAIN",
                       "gemini:gemini-3.5-flash-lite, groq:openai/gpt-oss-120b,gemini:gemini-3.1-flash-lite")
    assert [m.id for m in llm.chain()] == [
        "gemini:gemini-3.5-flash-lite", "groq:openai/gpt-oss-120b", "gemini:gemini-3.1-flash-lite"]
    monkeypatch.setenv("GROQ_API_KEY", "")   # entries for a provider without a key are dropped
    assert [m.id for m in llm.chain()] == ["gemini:gemini-3.5-flash-lite", "gemini:gemini-3.1-flash-lite"]
    monkeypatch.setenv("LLM_MODEL_CHAIN", "nonsense")  # malformed entry -> ignored, no crash
    assert llm.chain() == []


def test_falls_through_groq_429_to_gemini(monkeypatch):
    calls = []

    def fake_post(model, body, timeout):
        calls.append(model.id)
        if model.provider == "groq":
            return FakeResponse(429, headers={"retry-after": "600"})
        return FakeResponse(200, json.dumps({"directives": []}))

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm.chat_json("s", "u") == {"directives": []}
    assert calls == ["groq:g1", "groq:g2", "gemini:gem1"]
    # Second request skips the rate-limited Groq models entirely.
    calls.clear()
    llm.chat_json("s", "u")
    assert calls == ["gemini:gem1"]


def test_400_drops_optional_params_then_succeeds(monkeypatch):
    bodies = []

    def fake_post(model, body, timeout):
        bodies.append(dict(body))
        if "reasoning_effort" in body or "response_format" in body:
            return FakeResponse(400)
        return FakeResponse(200, "```json\n{\"directives\": [1]}\n```")  # fenced output tolerated

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm.chat_json("s", "u") == {"directives": [1]}
    assert "reasoning_effort" in bodies[0] and "reasoning_effort" not in bodies[1]
    assert "response_format" in bodies[1] and "response_format" not in bodies[2]


def test_all_models_down_raises_llmerror(monkeypatch):
    def fake_post(model, body, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm, "_post", fake_post)
    with pytest.raises(llm.LLMError):
        llm.chat_json("s", "u", deadline=time.monotonic() + 5)


def test_no_provider_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(llm.LLMError, match="no LLM provider"):
        llm.chat_json("s", "u")
