"""Run the public sample cases and judge the results like the organizer harness.

Offline (optimizer only, uses ground-truth directives from the sample pack):
    python tests/run_samples.py

Against a running service (checks interpretation + plan + cost):
    python tests/run_samples.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.optimizer import optimize, totals  # noqa: E402
from app.replay import TOL, check_interpretation, check_plan  # noqa: E402

DEFAULT_CASES = ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def post(url: str, body: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/optimize-energy",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_case(case: dict, url: str | None) -> tuple[bool, float, list[str]]:
    inp, exp = case["input"], case["expected_output"]
    truth = exp["directive_interpretation"]
    errs: list[str] = []

    # Sanity-check our replay against the organizer's own reference plan.
    ref_errs = check_plan(inp, truth, exp)
    if ref_errs:
        errs += [f"[reference plan] {e}" for e in ref_errs]

    t0 = time.perf_counter()
    if url:
        try:
            resp = post(url, inp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            return False, time.perf_counter() - t0, [f"request failed: {e}"]
        if resp.get("scenario_id") != inp["scenario_id"]:
            errs.append(f"scenario_id={resp.get('scenario_id')!r}")
        errs += [f"[interp] {e}" for e in check_interpretation(
            resp.get("directive_interpretation"), truth)]
    else:
        plan = optimize(inp["hours"], inp["battery"], truth)
        resp = {"hourly_plan": plan, **totals(plan, inp["hours"])}
    elapsed = time.perf_counter() - t0

    # The judge replays against the TRUE directives, not the reported ones.
    errs += [f"[plan] {e}" for e in check_plan(inp, truth, resp)]

    ref_cost = exp["total_cost_bdt"]
    cost = resp.get("total_cost_bdt")
    if isinstance(cost, (int, float)) and cost > ref_cost + TOL:
        errs.append(f"[cost] {cost} worse than reference {ref_cost}")
    elif isinstance(cost, (int, float)) and cost < ref_cost - TOL:
        errs.append(f"[cost] {cost} beats reference {ref_cost} (check validity!)")
    return not errs, elapsed, errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="base URL of a running service")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("-v", "--verbose", action="store_true", help="print every error")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    mode = f"service {args.url}" if args.url else "offline optimizer"
    print(f"Running {len(cases)} cases against {mode}\n")

    passed, times = 0, []
    for case in cases:
        ok, t, errs = run_case(case, args.url)
        times.append(t)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<10} {t * 1000:7.0f} ms  {case['label']}")
        for e in errs if args.verbose else errs[:5]:
            print(f"        {e}")
        if not args.verbose and len(errs) > 5:
            print(f"        ... {len(errs) - 5} more (use -v)")

    times.sort()
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    print(f"\n{passed}/{len(cases)} passed   p95 latency {p95 * 1000:.0f} ms")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
