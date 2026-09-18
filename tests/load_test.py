"""Burst and concurrency test (Guide §8: p95 latency, 30 s timeout, failure rate).

Phase A: N distinct cases sent back-to-back to a fresh server.
Phase B: C distinct cases sent at the same moment to another fresh server.
Every response is graded against ground truth like the judge would.
Uses real Groq quota: about N + C LLM calls.

    python tests/load_test.py                    # N=20 sequential, C=10 concurrent
    python tests/load_test.py --sequential 30 --concurrent 15
    python tests/load_test.py --url https://your-deployment   # test a deployed service
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.replay import check_interpretation, check_plan  # noqa: E402
from tests.live_server import call, load_cases, server  # noqa: E402


def grade(case, status, text, secs) -> dict:
    row = {"id": case["id"], "status": status, "secs": secs,
           "interp_ok": False, "plan_ok": False, "fallback": False}
    if status != 200:
        return row
    body = json.loads(text)
    row["interp_ok"] = not check_interpretation(body.get("directive_interpretation"), case["truth"])
    applied = [e for e in case["truth"] if e["applies"]]
    row["plan_ok"] = not check_plan(case["input"], applied, body)
    row["fallback"] = any("Could not be interpreted" in e.get("explanation", "")
                          for e in body.get("directive_interpretation", []))
    return row


def p95(xs):
    xs = sorted(xs)
    return xs[math.ceil(0.95 * len(xs)) - 1] if xs else float("nan")


def report(title, rows, wall):
    lat = [r["secs"] for r in rows]
    statuses = Counter(r["status"] for r in rows)
    n = len(rows)
    print(f"\n== {title}: {n} requests in {wall:.1f}s wall time ==")
    print(f"  HTTP status        {dict(statuses)}")
    print(f"  5xx / no response  {sum(1 for r in rows if r['status'] is None or r['status'] >= 500)}")
    print(f"  latency            median {statistics.median(lat):.2f}s   p95 {p95(lat):.2f}s   max {max(lat):.2f}s")
    print(f"  over 5s / over 30s {sum(x > 5 for x in lat)} / {sum(x > 30 for x in lat)}")
    print(f"  interpretation ok  {sum(r['interp_ok'] for r in rows)}/{n}"
          f"   (LLM fallback to no_op: {sum(r['fallback'] for r in rows)})")
    print(f"  plan valid         {sum(r['plan_ok'] for r in rows)}/{n}")
    points = 3 if p95(lat) <= 5 else 2 if p95(lat) <= 15 else 1 if p95(lat) <= 30 else 0
    print(f"  => rubric latency points at this p95: {points}/3")
    bad = [r for r in rows if not (r["interp_ok"] and r["plan_ok"])]
    for r in bad:
        print(f"     {r['id']}: status={r['status']} {r['secs']:.1f}s interp_ok={r['interp_ok']} "
              f"plan_ok={r['plan_ok']} fallback={r['fallback']}")
    return rows


def run_phase(base, cases, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call, base + "/optimize-energy", c["input"], None, 35) for c in cases]
        rows = [grade(c, *f.result()) for c, f in zip(cases, futures)]
    return rows, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequential", type=int, default=20)
    ap.add_argument("--concurrent", type=int, default=10)
    ap.add_argument("--url", help="test an already-running/deployed service instead of fresh local ones")
    args = ap.parse_args()

    cases = load_cases()
    seq = cases[:args.sequential]
    par = cases[args.sequential:args.sequential + args.concurrent]
    if len(par) < args.concurrent:
        par = cases[-args.concurrent:]
    ok = True

    for title, subset, workers in [("A. back-to-back burst", seq, 1),
                                   ("B. simultaneous requests", par, len(par))]:
        if not subset:
            continue
        ctx = nullcontext((args.url.rstrip("/"), None, 0)) if args.url else server()
        with ctx as (base, _, startup):
            if startup:
                print(f"\n(fresh server ready in {startup:.1f}s)")
            rows, wall = run_phase(base, subset, workers)
        report(title, rows, wall)
        ok &= all(r["status"] == 200 and r["secs"] <= 30 for r in rows)
        if title.startswith("A") and not args.url:
            time.sleep(20)  # let the per-minute token budget recover between phases

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
