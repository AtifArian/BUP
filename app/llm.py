"""Chat client for OpenAI-compatible providers (Groq, Gemini) with a model chain.

Every configured model is tried in order. LLM_MODEL_CHAIN gives the exact
order as provider:model entries (e.g. "gemini:gemini-3.5-flash-lite,
groq:openai/gpt-oss-120b"); otherwise the chain is built per provider from
LLM_PROVIDER_ORDER and each provider's MODEL/FALLBACK_MODEL. A model is
skipped for the rest of the
request when it times out, errors or is rate limited, and 429 cooldowns are
remembered across requests so a quota-exhausted model does not cost every
request a wasted round trip. If every model is cooling down, we wait for the
soonest one as long as the overall deadline allows.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger("gridwise.llm")

PROVIDERS = {
    # provider: (base-url env, default base url, key env,
    #            model env, default model, fallback env, default fallbacks,
    #            reasoning-effort env, default effort)
    "groq": ("GROQ_BASE_URL", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
             "GROQ_MODEL", "openai/gpt-oss-120b",
             "GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b,qwen/qwen3.8-27b",
             "GROQ_REASONING_EFFORT", "low"),
    "gemini": ("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai",
               "GEMINI_API_KEY",
               "GEMINI_MODEL", "gemini-3.5-flash-lite",
               "GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite",
               "GEMINI_REASONING_EFFORT", "low"),
}

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMError(Exception):
    """The model could not be reached or returned unusable output."""


@dataclass(frozen=True)
class Model:
    provider: str
    name: str
    url: str
    key: str
    effort: str  # "" = do not send reasoning_effort

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.name}"


_RATE_LIMITED: dict[str, float] = {}  # model id -> monotonic time its 429 cooldown ends


def _split(csv: str) -> list[str]:
    return [m.strip() for m in csv.split(",") if m.strip()]


def _provider(provider: str) -> tuple[str, str, str] | None:
    """(chat url, key, reasoning effort) for a configured provider, or None if it has no key."""
    spec = PROVIDERS.get(provider)
    if spec is None:
        log.warning("unknown LLM provider %r", provider)
        return None
    url_env, url_default, key_env, _, _, _, _, effort_env, effort_default = spec
    key = os.getenv(key_env, "").strip()
    if not key:
        return None
    url = os.getenv(url_env, url_default).rstrip("/") + "/chat/completions"
    return url, key, os.getenv(effort_env, effort_default).strip()


def chain() -> list[Model]:
    """Configured models in the order they are tried. Providers without a key are skipped."""
    models: list[Model] = []
    explicit = os.getenv("LLM_MODEL_CHAIN", "").strip()
    if explicit:
        for item in _split(explicit):
            provider, sep, name = item.partition(":")
            if not sep or not name:
                log.warning("LLM_MODEL_CHAIN entry %r must be provider:model", item)
                continue
            cfg = _provider(provider.lower())
            if cfg is None:
                log.warning("LLM_MODEL_CHAIN entry %r skipped: provider has no API key", item)
                continue
            url, key, effort = cfg
            models.append(Model(provider.lower(), name, url, key, effort))
        return list(dict.fromkeys(models))
    for provider in _split(os.getenv("LLM_PROVIDER_ORDER", "groq,gemini")):
        cfg = _provider(provider.lower())
        if cfg is None:
            continue
        url, key, effort = cfg
        _, _, _, model_env, model_default, fb_env, fb_default, _, _ = PROVIDERS[provider.lower()]
        names = _split(os.getenv(model_env, model_default)) + _split(os.getenv(fb_env, fb_default))
        for name in dict.fromkeys(names):
            models.append(Model(provider.lower(), name, url, key, effort))
    return models


def _retry_after(r: httpx.Response) -> float:
    try:
        return float(r.headers.get("retry-after", "2"))
    except ValueError:
        return 2.0


def _parse(r: httpx.Response) -> dict:
    content = r.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("content is not text")
    m = _FENCE.match(content)
    out = json.loads(m.group(1) if m else content)
    if not isinstance(out, (dict, list)):
        raise ValueError("not a JSON object")
    return out


def _post(model: Model, body: dict, timeout: float) -> httpx.Response:
    return httpx.post(model.url, headers={"Authorization": f"Bearer {model.key}"},
                      json=body, timeout=timeout)


def chat_json(system: str, user: str, deadline: float | None = None,
              models: list[Model] | None = None) -> dict:
    """Send one JSON-mode chat request down the model chain; return the parsed object.

    `deadline` is a time.monotonic() value; no request is started or waited
    for beyond it. `models` overrides the configured chain (used by probes).
    Never logs an API key.
    """
    models = chain() if models is None else models
    if not models:
        raise LLMError("no LLM provider configured (set GROQ_API_KEY and/or GEMINI_API_KEY)")
    per_call = float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
    if deadline is None:
        deadline = time.monotonic() + float(os.getenv("LLM_BUDGET_SECONDS", "20"))

    now = time.monotonic()
    # Start from remembered 429 cooldowns; per-call failures are added on top.
    cooldown: dict[str, float] = {m: t for m, t in _RATE_LIMITED.items() if t > now}
    last = "no model attempted"

    while True:
        for model in models:
            now = time.monotonic()
            left = deadline - now
            if left < 1:
                raise LLMError(f"deadline reached ({last})")
            if cooldown.get(model.id, 0) > now:
                continue
            body = {
                "model": model.name,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if model.effort:
                body["reasoning_effort"] = model.effort
            try:
                r = _post(model, body, min(per_call, left))
                # A 400 usually means this model rejects an optional parameter.
                # Retry without reasoning_effort, then without JSON mode (the
                # system prompt already demands bare JSON and _parse strips fences).
                for drop in ("reasoning_effort", "response_format"):
                    if r.status_code != 400 or drop not in body:
                        continue
                    log.info("LLM %s rejected request; retrying without %s", model.id, drop)
                    del body[drop]
                    left = deadline - time.monotonic()
                    if left < 1:
                        break
                    r = _post(model, body, min(per_call, left))
            except httpx.HTTPError as e:
                last = f"{model.id}: {type(e).__name__}"
                log.warning("LLM call failed: %s", last)
                cooldown[model.id] = deadline  # don't retry a model that timed out
                continue

            if r.status_code == 429:
                cooldown[model.id] = _RATE_LIMITED[model.id] = time.monotonic() + _retry_after(r)
                last = f"{model.id}: rate limited"
                log.warning("LLM rate limited: %s (retry after %.1fs)", model.id, _retry_after(r))
                continue
            if r.status_code != 200:
                last = f"{model.id}: HTTP {r.status_code}"
                log.warning("LLM call failed: %s", last)
                cooldown[model.id] = deadline if r.status_code < 500 else time.monotonic() + 1
                continue
            try:
                return _parse(r)
            except (KeyError, IndexError, ValueError, TypeError) as e:
                last = f"{model.id}: unparseable output ({type(e).__name__})"
                log.warning("LLM call failed: %s", last)
                continue  # next model may do better

        # Every model is cooling down: wait for the soonest one if time allows.
        soonest = min(cooldown.get(m.id, 0) for m in models)
        wait = soonest - time.monotonic()
        if soonest >= deadline or time.monotonic() + max(wait, 0) > deadline - 1:
            raise LLMError(f"all models unavailable ({last})")
        time.sleep(max(wait, 0.05))
