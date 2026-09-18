# GridWise edge-case benchmark (unofficial)

This pack is **original synthetic testing material**, not organizer-supplied hidden tests. It is based on the uploaded BUP CSE Fest 2026 Problem Statement, Participant Guide, and public examples. The public pack's ten cases have **not** been duplicated here.

## Files

| File | Purpose |
|---|---|
| `gridwise_edge_cases.json` | 39 feasible, valid end-to-end cases, 13 invalid API requests. Each valid case has a request body, exact structured-directive ground truth (excluding prose explanations), tags, and an independently solved optimum cost. |
| `run_gridwise_edge_tests.py` | Standard-library-only test runner: checks interpretation, HTTP/schema, 24-hour energy replay, battery and directive constraints, totals, cost optimality and latency. |
| `guardrail_injection_fixtures.json` | 19 synthetic outputs to inject directly into your **local** LLM-output guardrail validator: one accepted control and 18 rejected outputs. These are *not* requests for the public API. |

## Before benchmarking

Run your own backend once with **Gemini 2.5 Flash** as its interpreter and once with **GPT-OSS-120B** as its interpreter, keeping the extraction prompt, optimizer, guardrails, hardware/deployment conditions, and non-provider logic as consistent as practical. The script tests your GridWise endpoint, **not** a Gemini/Groq model endpoint. It makes real HTTP calls and uses your provider's quota.

The target's `GET /health` must return HTTP 200 and `{"status":"ok"}`. Valid requests are sent to `POST /optimize-energy`.

## Quick start

No Python packages are required to execute the runner. Python 3.10+ is recommended.

```bash
# Start with 1 demanding case to avoid burning free API quota:
python run_gridwise_edge_tests.py --base-url https://YOUR-API-HOST --case EDGE-25

# Check all 39 valid scenarios once; save a report:
python run_gridwise_edge_tests.py --base-url https://YOUR-API-HOST --report gemini_results.json

# Add 13 malformed requests (52 POST requests in total):
python run_gridwise_edge_tests.py --base-url https://YOUR-API-HOST --include-invalid --report gemini_results.json

# Run the same suite against the other model's deployment:
python run_gridwise_edge_tests.py --base-url https://YOUR-SECOND-API-HOST --include-invalid --report gptoss_results.json

# Target the language-understanding tests:
python run_gridwise_edge_tests.py --base-url https://YOUR-API-HOST --tag percentage

# Repeat requests to probe variability (3 x 39 = 117 valid calls; check free API limits!):
python run_gridwise_edge_tests.py --base-url https://YOUR-API-HOST --repeat 3 --report stability.json
```

If the same deployment switches provider via environment variables, switch providers and rerun against the same URL. Run each model under approximately comparable traffic conditions. A side-by-side tally should focus on `directive_interpretation_passed`, `plan_passed`, `optimality_passed`, `p95_valid_latency_sec`, and `valid_over_30_sec`; the complete report includes per-case errors.

**Important:** A model that produces a correct directive can still fail schedule validation if your backend does not enforce it. Keep errors separated into interpretation, schedule, and optimum cost. The judge grades these separately.

## Ground truth and check strategy

For each `valid_cases[i]`, POST **only** its `input` object. Do not POST the surrounding test metadata or ground truth. `expected_interpretation` has four machine-checked fields: `note_index`, `applies`, `directive_type`, and `structured_adjustment`. The runner requires a string `explanation` but does not compare its wording. Time windows are start-inclusive, end-exclusive; `factor` is the remaining solar fraction. The validator checks every applicable directive from this ground truth against the *returned plan*, independently of the model's claimed interpretation.

All 39 valid scenarios were verified feasible using SciPy HiGHS linear programming. Their `reference_optimal_cost_bdt` values are the cost of a reference optimum, **not** a byte-for-byte reference schedule. The runner permits an absolute 0.01 tolerance for costs/energy. The LP uses zero-loss battery charge/discharge, following the specification. Equivalent optimum schedules are accepted.

The 13 invalid cases expect an HTTP 400 or 422, as appropriate to the user's API validation policy. The `raw_body` item tests syntax-invalid JSON and nonstandard NaN; cases with `raw_body: null` send the `input` object. They are reliability tests, *not* feasible hidden-scoring cases.

The zero-capacity battery case is explicitly an additional stress test, **not** a claim that the organizer's hidden cases include this configuration. If your backend intentionally disallows this request, inspect the policy and report the result separately. Midnight-to-midnight windows use the unambiguous written notation `00:00-24:00`; rollover windows such as `22:00-02:00` were avoided because cross-day handling is not explicitly specified. Deliberately contradictory operator notes were also avoided: the published valid scoring scenarios are feasible.

## Guardrail injection tests

`guardrail_injection_fixtures.json` provides a sample `operator_notes` array and a capacity of 180 kWh. Feed each fixture's `model_output` into your internal `validate_interpretation(...)` function without calling Gemini/Groq, then assert `expected_accept`. This requires a local test adapter to your own guardrail function and **cannot** be tested through `/optimize-energy` alone. The suite covers duplicate/incorrect note mapping, unsupported types, wrong `applies`, invalid `no_op`, unordered/repeated/out-of-range hours, bad factor, negative grid cap, excessive reserve, missing keys and malformed model output.

## Official rules used

- Problem Statement, §§4-5: six directive types, 1-3 notes, `no_op`, inclusive-start/exclusive-end hours, factor means usable fraction remaining, and cost minimization.
- Problem Statement, §§8-11: validate untrusted LLM output; grid/solar/battery balance, limits, end-of-day neutrality, returned fields, and absolute 0.01 numeric tolerance.
- Participant Guide, §§7-9: interpretation and application scored separately, timeout of 30 seconds and full p95 latency credit at 5 seconds or less, safe input/provider failure handling.

This pack does not impersonate official samples or guarantee hidden-case coverage. The test runner does not inject model failures or count cost of tokens; inspect your provider dashboard for token and quota usage.
