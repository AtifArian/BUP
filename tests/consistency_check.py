"""Interpretation consistency (Guide §7: 'paraphrase robustness across related hidden notes').

Interprets every public sample and edge-pack case R times with the cache cleared
and reports any note whose answer changes between runs, plus accuracy per run.
The LLM budget is raised so free-tier rate limits cause waiting, not fallbacks,
which keeps real inconsistency separate from quota problems.
Uses real Groq quota: cases x repeats LLM calls.

    python tests/consistency_check.py              # 3 repeats over all cases
    python tests/consistency_check.py --repeat 5 --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["LLM_BUDGET_SECONDS"] = os.getenv("CONSISTENCY_BUDGET_SECONDS", "120")
os.environ.setdefault("LOG_LEVEL", "ERROR")

import logging  # noqa: E402

logging.basicConfig(level=logging.ERROR)

from app import interpreter  # noqa: E402
from app.replay import check_interpretation  # noqa: E402
from tests.live_server import load_cases  # noqa: E402


def signature(entry: dict) -> str:
    return json.dumps([entry["directive_type"], entry["structured_adjustment"]], sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only the first N cases")
    args = ap.parse_args()

    cases = load_cases()[: args.limit]
    answers: dict[str, list[list[str]]] = {c["id"]: [] for c in cases}
    correct = {c["id"]: [] for c in cases}
    unavailable = 0
    t0 = time.perf_counter()

    for run in range(1, args.repeat + 1):
        interpreter._interpret_cached.cache_clear()
        ok = 0
        for c in cases:
            inp = c["input"]
            entries, warnings = interpreter.interpret(
                inp["operator_notes"], inp["battery"]["capacity_kwh"],
                [h["solar_kwh"] for h in inp["hours"]], inp["battery"]["minimum_energy_kwh"])
            if warnings:
                unavailable += 1
            answers[c["id"]].append([signature(e) for e in entries])
            good = not check_interpretation(entries, c["truth"])
            correct[c["id"]].append(good)
            ok += good
        print(f"run {run}: {ok}/{len(cases)} cases fully correct  ({time.perf_counter() - t0:.0f}s elapsed)",
              flush=True)

    print("\n== Notes whose answer changed between runs ==")
    flips = 0
    for c in cases:
        runs = answers[c["id"]]
        for i, note in enumerate(c["input"]["operator_notes"]):
            variants = {r[i] for r in runs}
            if len(variants) > 1:
                flips += 1
                print(f"  {c['id']} note {i}: {note[:90]!r}")
                for v in sorted(variants):
                    print(f"      {sum(r[i] == v for r in runs)}x  {v}")
    if not flips:
        print("  none - every note got the same answer every time")

    never = [cid for cid, rs in correct.items() if not any(rs)]
    sometimes = [cid for cid, rs in correct.items() if any(rs) and not all(rs)]
    total_notes = sum(len(c["input"]["operator_notes"]) for c in cases)
    print(f"\nnotes checked: {total_notes} x {args.repeat} runs")
    print(f"inconsistent notes: {flips}")
    print(f"cases always correct: {sum(all(rs) for rs in correct.values())}/{len(cases)}")
    print(f"cases sometimes wrong: {sometimes or 'none'}")
    print(f"cases always wrong: {never or 'none'}")
    print(f"requests that hit the LLM fallback (quota/outage, not model error): {unavailable}")
    return 1 if flips or never or sometimes else 0


if __name__ == "__main__":
    sys.exit(main())
