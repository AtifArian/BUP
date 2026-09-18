"""Groq chat client (OpenAI-compatible REST API) with model chain and rate-limit handling.

Each Groq model has its own tokens-per-minute budget, so on HTTP 429 we move on
to the next model in the chain. If every model is rate limited, we wait for the
shortest retry-after as long as the overall deadline allows.
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx

log = logging.getLogger("gridwise.llm")

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _url() -> str:
    base = os.getenv("GROQ_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/chat/completions"


class LLMError(Exception):
    """The model could not be reached or returned unusable output."""


def _models() -> list[str]:
    raw = [
        os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        *os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(","),
    ]
    return [m.strip() for m in dict.fromkeys(raw) if m.strip()]


def _retry_after(r: httpx.Response) -> float:
    try:
        return float(r.headers.get("retry-after", "2"))
    except ValueError:
        return 2.0


def chat_json(system: str, user: str, deadline: float | None = None) -> dict:
    """Send one JSON-mode chat request and return the parsed object.

    `deadline` is a time.monotonic() value; no request is started or waited
    for beyond it. Never logs the API key.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not configured")
    per_call = float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
    if deadline is None:
        deadline = time.monotonic() + float(os.getenv("LLM_BUDGET_SECONDS", "20"))
    effort = os.getenv("GROQ_REASONING_EFFORT", "low")  # gpt-oss: low|medium|high

    models = _models()
    cooldown: dict[str, float] = {}  # model -> monotonic time it may be retried
    last = "no model attempted"

    while True:
        for model in models:
            now = time.monotonic()
            left = deadline - now
            if left < 1:
                raise LLMError(f"deadline reached ({last})")
            if cooldown.get(model, 0) > now:
                continue
            body = {
                "model": model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if effort:
                body["reasoning_effort"] = effort
            try:
                r = httpx.post(_url(), headers={"Authorization": f"Bearer {key}"},
                               json=body, timeout=min(per_call, left))
                
                # If a fallback model doesn't support reasoning_effort, Groq returns HTTP 400
                if r.status_code == 400 and "reasoning_effort" in body:
                    log.info("LLM %s rejected reasoning_effort; retrying without it", model)
                    del body["reasoning_effort"]
                    left = deadline - time.monotonic()
                    if left > 1:
                        r = httpx.post(_url(), headers={"Authorization": f"Bearer {key}"},
                                       json=body, timeout=min(per_call, left))
            except httpx.HTTPError as e:
                last = f"{model}: {type(e).__name__}"
                log.warning("LLM call failed: %s", last)
                cooldown[model] = deadline  # don't retry a model that timed out
                continue

            if r.status_code == 429:
                cooldown[model] = time.monotonic() + _retry_after(r)
                last = f"{model}: rate limited"
                log.warning("LLM rate limited: %s (retry after %.1fs)", model, _retry_after(r))
                continue
            if r.status_code != 200:
                last = f"{model}: HTTP {r.status_code}"
                log.warning("LLM call failed: %s", last)
                cooldown[model] = deadline if r.status_code < 500 else time.monotonic() + 1
                continue
            try:
                return json.loads(r.json()["choices"][0]["message"]["content"])
            except (KeyError, IndexError, ValueError, TypeError) as e:
                last = f"{model}: unparseable output ({type(e).__name__})"
                log.warning("LLM call failed: %s", last)
                continue  # next model may do better

        # Every model is cooling down: wait for the soonest one if time allows.
        soonest = min(cooldown.get(m, 0) for m in models)
        wait = soonest - time.monotonic()
        if soonest >= deadline or time.monotonic() + max(wait, 0) > deadline - 1:
            raise LLMError(f"all models unavailable ({last})")
        time.sleep(max(wait, 0.05))
