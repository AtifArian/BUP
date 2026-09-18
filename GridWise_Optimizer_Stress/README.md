# GridWise optimizer — reproducible randomized stress test

This pack tests the **uploaded GridWise optimizer code**, with no Groq/Gemini access, no API key, no HTTP requests, and no simulated LLM. The bundled `app/optimizer.py`, `app/replay.py`, and `app/guardrails.py` are byte-for-byte copies of the uploaded originals. **The original project was not edited.**

## Test results (default reference run)

Command: `python stress_optimizer.py --per-category 150 --seed 20260918`

- **1,200 generated scenarios**, all guaranteed feasible by a constructed witness schedule, checked before optimization.
- **1,200/1,200** passed the project's `replay.check_plan()` **and an additional independently implemented replay**. No solver exceptions or rejected plans.
- **1,198/1,200** returned grid costs within **0.01 BDT** of a *second, independently formulated* SciPy LP optimum; **2 cost-precision flags**, both among full-day directive tests.
- `RANDOM-full_day_directives-002`: output **83,363.3239 BDT**, exact LP **83,363.33848451 BDT** (output lower by ~0.01458 BDT). The independently replayed result is still valid under the statement's **per-hour 0.01 kWh tolerance**. Four-decimal rounding of solar/grid explains how a rounded solution may lie slightly below the unrounded LP cost; this is a precision diagnostic, **not proof of an invalid solution**.
- `RANDOM-full_day_directives-089`: output **34,952.7738 BDT**, LP **34,952.76343845 BDT** (output higher by ~0.01036 BDT). This is an unusually small objective-value drift introduced by four-decimal output rounding. The returned schedule replays successfully.

An experimental test **on a temporary copy only** changed `ROUND = 4` to `ROUND = 6` in `optimizer.py`. With the identical 1,200 cases and random seed, the experimental copy achieved **1,200/1,200 replay validity and 1,200/1,200 cost comparisons within 0.01 BDT**. See `precision_experiment/`. The bundle's main optimizer remains unchanged. This experiment is not a guarantee for all possible future cases; the precision change should be retested against the official 10 public cases too.

## Categories (150 each)

| Category | What it stresses | Replay-valid | Cost flags |
|---|---|---:|---:|
| normal | Mixed demand, solar, tariffs, charge/discharge | 150 | 0 |
| zero_demand | Every hour has zero demand | 150 | 0 |
| zero_rates | Both battery rate limits equal zero | 150 | 0 |
| decimals | Very small and four-decimal figures | 150 | 0 |
| full_day_directives | Exactly 3 directives, each applying to every hour | 150 | 2 |
| tight_grid | Grid caps set to a known feasible witness | 150 | 0 |
| solar_surplus | Every hour has solar exceeding demand | 150 | 0 |
| all_zero | Demand, solar, tariff, capacity and rates are all zero | 150 | 0 |

Cases may include zero tariffs, zero capacities, solar-curtailment, single-hour and full-day directives, zero rates, arbitrary reserves, no charge, no discharge, and input hours shuffled into a different order. Maximum three operator directives per case; no unknown/unpublished directive types. Each valid case includes a known-feasible **witness** with battery actions, solar and grid values. This harness sends these known structured directives **directly into** `optimize`, deliberately bypassing LLM interpretation.

## Reproduce

Python dependencies: `numpy`, `scipy` (the bundled code also imports no other required third-party libraries besides these for the optimizer/replay/guardrails). Run from the extracted directory:

```bash
python -m pip install numpy scipy
python stress_optimizer.py --per-category 150 --seed 20260918
```

The process exits with code **1** when replay validity fails **or** a cost precision diagnostic occurs. For the default unchanged original, expect exit code 1 due to the two cost flags even though **all 1,200 plans are valid**. Check `stress_report.json`, `flagged_cases.json`, and `stress_console.txt`. To test your edited implementation, replace `app/optimizer.py` / `app/replay.py` / `app/guardrails.py` with the latest corresponding project files, then re-run. The harness imports directly from its local `app/` directory; it does **not** load the service or call an LLM.

To run a different randomized campaign:

```bash
python stress_optimizer.py --seed 12345 --per-category 100 --output another_run
```

`--per-category 100` means **800** scenarios. `--no-save-cases` suppresses the large generated-case JSON. Scenario IDs are stable for a given seed and count.

## Files

- `stress_optimizer.py`: generator, witness verifier, direct optimizer call, project replay, **independent** replay and independent LP objective cross-check.
- `app/`: unchanged snapshots of the three original code modules and a minimal package initializer.
- `random_solvable_cases.json`: all 1,200 inputs, oracle directives and feasible witnesses (generated).
- `stress_report.json`: counts, categories, coverage and the two cost diagnostics.
- `flagged_cases.json`: full input, directives, witnesses for the two flagged cases.
- `stress_console.txt`: baseline console output.
- `precision_experiment/`: report and console for temporary 6-decimal experiment.

## Scope and limitations

This is a seeded finite stress campaign, **not a proof** of general correctness or a reproduction of unpublished judging tests. Independent replay verifies the mathematical constraints separately from the project's own `effective_bounds()` implementation, but both use the documented rules. The second LP removes the explicit grid variables and checks cost independently; a small discrepancy caused solely by output rounding is reported separately from a feasibility failure. The inputs are constructed to be feasible even when directives overlap, so this tests the optimizer's ability to solve and replay feasible cases, not behavior on infeasible scenarios. No provider latency, Groq/Gemini correctness, deployment or LLM failure path is exercised.
