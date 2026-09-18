# GridWise LLM Energy Optimizer

LLM-assisted 24-hour campus energy scheduler for the BUP CSE Fest 2026 preliminary
(GridWise challenge). One FastAPI service exposes `GET /health` and `POST /optimize-energy`.
Operator notes are interpreted by a language model, validated by deterministic guardrails,
applied to a linear-programming optimizer, and the final schedule is replayed hour by hour
before it is returned.

## Contents

1. [Architecture](#architecture)
2. [Local quickstart](#local-quickstart)
3. [Environment variables](#environment-variables)
4. [API](#api)
5. [Testing](#testing)
6. [Docker fallback](#docker-fallback)
7. [Dependencies](#dependencies)
8. [Known limitations](#known-limitations)
9. [Secret handling](#secret-handling)

## Architecture

```
operator_notes ──► LLM (Groq → Gemini chain) ──► normalise ──► guardrails ──► LP optimizer ──► replay check ──► response
                    raw semantics    exact shape   §08 checks     HiGHS, 2-phase   §09/§11 rules
```

| Stage | File | What it does |
|---|---|---|
| API layer | `app/main.py`, `app/schemas.py` | FastAPI app. Pydantic validates the request (24 unique hours, 1-3 non-empty notes, battery bounds). Malformed/invalid requests → `400`; a scenario that is infeasible even with no directives → `422`; unexpected errors → `500` with a generic body (no stack traces). |
| **LLM interpreter** | `app/interpreter.py`, `app/llm.py` | The language model reads every operator note and returns *raw semantics* only: directive type, clock windows `[start, end)`, the number as stated, and what the number means (`remaining_percent`, `reduction_percent`, `kwh`, `percent_of_capacity`, `kwh_above_base_minimum`). Deterministic code then expands windows into hour lists, converts percentages/fractions, and builds the exact `structured_adjustment`. Models are called through their OpenAI-compatible chat endpoints (Groq, then Gemini) in JSON mode at temperature 0. |
| Guardrails | `app/guardrails.py` | Every entry is checked before it can reach the optimizer: allowed directive type, one entry per note in `note_index` order, hours are unique ascending integers 0-23, `factor` in [0, 1], reserve finite and ≤ capacity, grid cap finite and ≥ 0, `no_op` ⇒ `applies=false` + `null` adjustment, everything else ⇒ `applies=true`. Model text is scanned for secret-looking strings and redacted. |
| Optimizer | `app/optimizer.py` | Linear program solved with `scipy.optimize.linprog` (HiGHS). Phase 1 minimises grid cost; phase 2 keeps that cost and minimises battery throughput, so no hour both charges and discharges. Directives become per-hour bounds (effective solar, reserve floor, charge/discharge rate 0, grid cap). End-of-day neutrality is a hard equality. |
| Replay | `app/replay.py` | Independent hour-by-hour replay of the returned plan against the enforced directives and every GridWise rule (energy balance, battery transitions/bounds/rates, effective solar, grid cap, neutrality, totals). If it fails the service returns `500` instead of an invalid plan. |

**Safe failure.** The pipeline never invents a directive:

- The model is asked once for all notes; any note whose output is missing, malformed, or rejected by
  the guardrails is re-asked on its own with a corrective hint (within the request's LLM budget).
- If a note still cannot be interpreted, or the provider is unreachable, that note is returned as
  `no_op` with the explanation `"Could not be interpreted reliably; no constraint applied."`, a warning
  is logged, and the schedule is still produced (HTTP 200).
- If the interpreted directives are mutually infeasible (organizer scenarios are feasible, so this
  means a misread), the largest feasible subset is enforced and `plan_summary` names the notes that
  were not enforced. The interpretation itself is returned unchanged.
- Model chain: on rate limiting (429), timeouts, HTTP errors or unparseable output the client moves
  to the next model in `LLM_MODEL_CHAIN` (Gemini 3.5 Flash-Lite → Groq gpt-oss-120b → Gemini 3.1
  Flash-Lite). 429 cooldowns are remembered across requests so an exhausted model
  is skipped rather than re-tried on every call. A model that rejects an optional parameter (HTTP 400)
  is retried without `reasoning_effort`, then without JSON mode.

## Local quickstart

Requires Python 3.11+ and at least one LLM key: Groq (<https://console.groq.com>) and/or Gemini
(<https://aistudio.google.com/apikey>). Configure both for the most headroom; free tiers work.

```bash
git clone <repository-url> gridwise && cd gridwise
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env         # then fill in GROQ_API_KEY= and/or GEMINI_API_KEY=
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
     -H "Content-Type: application/json" \
     -d @examples/sample_request.json
```

Expected: HTTP 200 and a body shaped like [`examples/sample_response.json`](examples/sample_response.json)
(public sample `SAMPLE-01`): note 0 → `solar_reduction` with `{"hours": [12, 13], "factor": 0.25}`,
note 1 → `no_op`, `total_cost_bdt` = 38365 (± 0.01). The hourly plan may differ from the file as long as
it is valid and costs the same.

## Environment variables

Copy `.env.example` to `.env` (the file is git- and docker-ignored). Names only; never commit values.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `GROQ_API_KEY` | one of the two keys | – | Groq API key. |
| `GEMINI_API_KEY` | one of the two keys | – | Gemini API key (Google AI Studio). |
| `LLM_MODEL_CHAIN` | no | – | Exact try-order as `provider:model` entries, e.g. `gemini:gemini-3.5-flash-lite,groq:openai/gpt-oss-120b,gemini:gemini-3.1-flash-lite`. Takes precedence over the per-provider settings below. Entries whose provider has no key are skipped. |
| `LLM_PROVIDER_ORDER` | no | `groq,gemini` | Used when `LLM_MODEL_CHAIN` is unset: which provider's models are tried first. |
| `GROQ_MODEL` | no | `openai/gpt-oss-120b` | Primary Groq model. |
| `GROQ_FALLBACK_MODEL` | no | `openai/gpt-oss-20b,qwen/qwen3.8-27b` | Comma-separated Groq fallbacks. |
| `GROQ_BASE_URL` | no | `https://api.groq.com/openai/v1` | Groq endpoint. |
| `GROQ_REASONING_EFFORT` | no | `low` | `low`/`medium`/`high`; set empty to omit. |
| `GEMINI_MODEL` | no | `gemini-3.5-flash-lite` | Primary Gemini model. |
| `GEMINI_FALLBACK_MODEL` | no | `gemini-2.5-flash-lite` | Comma-separated Gemini fallbacks (submission uses `gemini-3.1-flash-lite`). |
| `GEMINI_BASE_URL` | no | `https://generativelanguage.googleapis.com/v1beta/openai` | Gemini's OpenAI-compatible endpoint. |
| `GEMINI_REASONING_EFFORT` | no | `low` | `minimal`/`low`/`medium`/`high`; set empty to omit. |
| `LLM_TIMEOUT_SECONDS` | no | `10` | Timeout per model call. |
| `LLM_BUDGET_SECONDS` | no | `20` | Total LLM time per request including retries (keeps the request under the 30 s judge limit). |
| `PORT` | no | `8000` | Port the container binds. |
| `HOST` | no | `0.0.0.0` | Bind address in the container. |
| `LOG_LEVEL` | no | `INFO` | Python logging level. |

Model chain used for submission (`LLM_MODEL_CHAIN`):

1. **Gemini `gemini-3.5-flash-lite`** — primary. Flash-Lite is Gemini's fastest, cheapest,
   highest-throughput family, which fits many short extraction requests; the task needs no long reasoning.
2. **Groq `openai/gpt-oss-120b`** — first fallback on a different provider, so a quota or outage on
   one vendor does not take the service down.
3. **Gemini `gemini-3.1-flash-lite`** — last resort.

No fine-tuning or training is performed.

Check that every configured model answers with your keys before judging:

```bash
python tests/probe_models.py
```

## API

### `GET /health`

`200` → `{"status": "ok"}`. Ready within a few seconds of process start.

### `POST /optimize-energy`

Request and response follow the Problem Statement exactly (§07 and §10). See
[`examples/sample_request.json`](examples/sample_request.json) and
[`examples/sample_response.json`](examples/sample_response.json).

| Status | When |
|---|---|
| `200` | Valid scenario; body contains `scenario_id`, `directive_interpretation`, `hourly_plan`, `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`, `plan_summary`. |
| `400` | Malformed JSON or schema violation (`{"error": "invalid request", "details": [...]}`). |
| `422` | Well-formed scenario with no feasible schedule even without directives. |
| `500` | Controlled internal error (`{"error": "internal error"}`); details stay in server logs. |

Interpretation conventions (Problem Statement §05): windows are start-inclusive/end-exclusive
(`1 PM to 3 PM` → `[13, 14]`); `factor` is the usable fraction remaining (`80% reduction` → `0.2`);
`minimum_battery_reserve` values in percent are converted using the request's `capacity_kwh`.

## Testing

Offline suite (LLM mocked, everything else real — optimizer, guardrails, replay, request validation):

```bash
python -m pytest tests -q
```

Public sample cases, optimizer only (uses the pack's ground-truth directives, no API key needed):

```bash
python tests/run_samples.py
```

Public sample cases end-to-end against a running service (real LLM calls):

```bash
python tests/run_samples.py --url http://localhost:8000
```

Expected: `10/10 passed` with interpretation, plan validity and cost all checked the way the judge does.
Other helpers in `tests/`: `probe_models.py` (one request per configured model), `check_paraphrases.py`
and `hidden_candidates.py` (paraphrase robustness against the live model), `load_test.py`
(concurrency/latency), `failure_injection.py` (provider outages).

## Docker fallback

Image: `python:3.11-slim` base, binds `0.0.0.0`, exposes `8000`, no secrets baked in (`.env` is excluded
by `.dockerignore`).

Pull the submitted image (replace with the exact reference from the submission form):

```bash
docker pull <registry>/<namespace>/gridwise-optimizer:<tag>
```

Run it (the key is passed at runtime, never stored in the image):

```bash
docker run --rm -p 8000:8000 -e GROQ_API_KEY=<groq-key> -e GEMINI_API_KEY=<gemini-key> <registry>/<namespace>/gridwise-optimizer:<tag>
```

or with a local `.env` file:

```bash
docker run --rm -p 8000:8000 --env-file .env <registry>/<namespace>/gridwise-optimizer:<tag>
```

Then `curl http://localhost:8000/health` → `{"status":"ok"}`.

Build locally instead:

```bash
docker build -t gridwise-optimizer .
docker run --rm -p 8000:8000 --env-file .env gridwise-optimizer
```

## Dependencies

All open-source, installed from `requirements.txt`:

| Package | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | HTTP API server |
| [Pydantic v2](https://docs.pydantic.dev/) | Request schema validation |
| [httpx](https://www.python-httpx.org/) | HTTP client for the Groq API |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Loads `.env` in development |
| [NumPy](https://numpy.org/) + [SciPy](https://scipy.org/) (`linprog`, HiGHS) | Linear-programming optimizer |
| [pytest](https://pytest.org/) (`requirements-dev.txt`) | Test runner |

LLM providers: [Groq](https://groq.com/) hosting `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and
`qwen/qwen3.8-27b`; [Google Gemini API](https://ai.google.dev/) (`gemini-3.5-flash-lite`,
`gemini-2.5-flash-lite`) via its OpenAI-compatible endpoint. AI coding assistants were used during development; the architecture, prompt design,
guardrails and optimizer formulation are the team's own work.

## Known limitations

- Time windows are whole hours only (as the Problem Statement requires); sub-hour times are not modelled.
- A note that gives a bare time without AM/PM is disambiguated by the model; as a safety net, a
  `solar_reduction` that lands only on hours with zero solar and no explicit AM/PM wording is read as PM.
- Percentages that are recurring decimals (33.33 %) are snapped to the exact fraction (⅓) before use.
- Interpretation results are cached per (notes, capacity, base reserve) in process memory; the cache is
  not shared across workers.
- Latency depends on the LLM provider; typical end-to-end time is 1-2 s, worst case is bounded by
  `LLM_BUDGET_SECONDS` plus a few hundred milliseconds of optimisation.

## Secret handling

- Secrets live only in environment variables. `.env` is listed in `.gitignore` and `.dockerignore`;
  `.env.example` contains names only.
- API keys are never logged. Model-written explanations are scanned for provider-style key patterns
  and redacted before they are returned.
- Error responses are generic; stack traces stay in server logs.
