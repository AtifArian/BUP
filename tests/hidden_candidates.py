"""Hidden test candidates — paraphrase variations and optimizer edge cases.

This file has two independent sections:

A) LLM PARAPHRASE TESTS  (run with --paraphrase, needs GROQ_API_KEY)
   19 extra notes that might appear in the hidden judge set, covering every
   directive type plus subtle distractors.

B) OPTIMIZER EDGE-CASE SCENARIOS  (always run, offline)
   12 hand-crafted 24-hour scenarios that stress-test energy balancing,
   battery constraints, directive combinations, and boundary conditions.

Usage:
    python tests/hidden_candidates.py                # optimizer edge cases only
    python tests/hidden_candidates.py --paraphrase   # also runs LLM tests (costs quota)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.interpreter import interpret  # noqa: E402
from app.optimizer import OptimizationError, optimize, totals  # noqa: E402
from app.replay import TOL, check_interpretation, check_plan  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# A. PARAPHRASE TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def _d(kind, hours=None, **vals):
    """Shorthand to build an expected interpretation entry."""
    if kind == "no_op":
        return {"applies": False, "directive_type": "no_op", "structured_adjustment": None}
    return {"applies": True, "directive_type": kind,
            "structured_adjustment": {"hours": hours, **vals}}


# (note_text, expected_entry, battery_capacity_used_for_percent_notes)
PARAPHRASE_CASES = [
    # ── solar_reduction ──────────────────────────────────────────────────────
    # fraction words
    ("Roof panels will be jet-washed 10 AM to 12 PM cutting PV output by 70 percent.",
     _d("solar_reduction", [10, 11], factor=0.3), 300),
    ("Two-thirds of forecast solar will be unavailable from 9 AM to 11 AM due to cloud.",
     _d("solar_reduction", [9, 10], factor=round(1/3, 4)), 300),
    ("Only about a tenth of normal PV output expected from 7 to 9 in the morning.",
     _d("solar_reduction", [7, 8], factor=0.1), 300),
    ("Expect virtually zero rooftop generation between 14:00 and 17:00 — treat as negligible.",
     _d("solar_reduction", [14, 15, 16], factor=0.0), 300),
    ("Shading from scaffolding will leave only 30% of the solar forecast from 8 AM until noon.",
     _d("solar_reduction", [8, 9, 10, 11], factor=0.3), 300),
    # single hour
    ("A test blackout on the PV array at exactly 3 PM will drop output to zero for that hour.",
     _d("solar_reduction", [15], factor=0.0), 300),

    # ── minimum_battery_reserve ───────────────────────────────────────────────
    ("Ensure no less than a third of battery capacity is held from 8 PM until 11 PM for safety.",
     _d("minimum_battery_reserve", [20, 21, 22], minimum_energy_kwh=round(300 / 3, 4)), 300),
    ("Battery floor of 75 kWh applies from 6 PM until 9 PM for critical loads.",
     _d("minimum_battery_reserve", [18, 19, 20], minimum_energy_kwh=75.0), 300),
    ("Keep at least three-quarters of battery capacity available from 10 PM to midnight.",
     _d("minimum_battery_reserve", [22, 23], minimum_energy_kwh=round(300 * 0.75, 4)), 300),

    # ── no_charge_window ─────────────────────────────────────────────────────
    ("The charger is being de-energised 5 AM to 7 AM for safety inspection.",
     _d("no_charge_window", [5, 6]), 300),
    ("Battery intake must remain at zero from 23:00 until midnight tonight.",
     _d("no_charge_window", [23]), 300),
    ("Charging equipment is locked out from 11 PM to 1 AM for grid protection tests.",
     _d("no_charge_window", [0, 23]), 300),

    # ── no_discharge_window ───────────────────────────────────────────────────
    ("Battery output must remain at zero between 15:00 and 18:00 for islanding tests.",
     _d("no_discharge_window", [15, 16, 17]), 300),
    ("Keep the battery from releasing power during the 8 PM to 10 PM window.",
     _d("no_discharge_window", [20, 21]), 300),

    # ── max_grid_window ───────────────────────────────────────────────────────
    ("Utility contract caps our draw at 0.12 MWh per hour from 17:00 to 20:00.",
     _d("max_grid_window", [17, 18, 19], max_grid_kwh=120.0), 300),
    ("No more than 250 kWh from the grid is permitted during peak hours 5 PM to 8 PM.",
     _d("max_grid_window", [17, 18, 19], max_grid_kwh=250.0), 300),

    # ── no_op distractors ────────────────────────────────────────────────────
    # mentions energy but refers to a different day or has no schedule impact
    ("Solar inverter warranty service is booked for next Tuesday.",
     _d("no_op"), 300),
    ("The battery was at 80% state of charge this morning — no action needed.",
     _d("no_op"), 300),
    ("Grid tariffs will be revised starting from next Monday.",
     _d("no_op"), 300),
]


def run_paraphrase_tests() -> tuple[int, int]:
    """Run all paraphrase cases in batches of 3 (one request each)."""
    ok = total = 0
    batch_size = 3
    for start in range(0, len(PARAPHRASE_CASES), batch_size):
        batch = PARAPHRASE_CASES[start:start + batch_size]
        notes = [n for n, _, _ in batch]
        cap = batch[0][2]  # capacity (same within a batch for simplicity)
        entries, warnings = interpret(notes, cap)
        for i, (note, truth, _) in enumerate(batch):
            got = {**entries[i], "note_index": 0}
            errs = check_interpretation([got], [truth])
            ok += not errs
            total += 1
            print(f"{'PASS' if not errs else 'FAIL'}  {note[:80]}")
            for e in errs:
                print(f"        {e}")
            for w in warnings:
                print(f"        warning: {w}")
    return ok, total


# ─────────────────────────────────────────────────────────────────────────────
# B. OPTIMIZER EDGE-CASE SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def _hours(demand, solar, tariff):
    """Build 24 hour entries from three 24-element lists."""
    return [{"hour": h, "demand_kwh": d, "solar_kwh": s, "tariff_bdt_per_kwh": t}
            for h, (d, s, t) in enumerate(zip(demand, solar, tariff))]


def _bat(cap=300, init=150, min_e=30, max_c=80, max_d=80):
    return {"capacity_kwh": cap, "initial_energy_kwh": init,
            "minimum_energy_kwh": min_e, "max_charge_kwh_per_hour": max_c,
            "max_discharge_kwh_per_hour": max_d}


# Tariff patterns
CHEAP = [5] * 24
EXPENSIVE = [15] * 24
PEAK_EVE = [5]*6 + [8]*6 + [12]*6 + [15]*6       # cheap night, peak evening
VALLEY_PEAK = [4]*8 + [12]*8 + [4]*8              # valley-peak-valley

# Solar patterns (daytime only)
NO_SOLAR = [0] * 24
DAYTIME = [0]*6 + [20, 60, 120, 180, 220, 250, 250, 220, 180, 120, 60, 20] + [0]*6
LIGHT_SOLAR = [x // 4 for x in DAYTIME]           # about 25% of full
HEAVY_SOLAR = [min(x * 2, 400) for x in DAYTIME]  # solar > demand in midday

# Demand patterns
FLAT_200 = [200] * 24
FLAT_100 = [100] * 24
EVE_HEAVY = [120]*6 + [150]*6 + [140]*6 + [200]*6  # evening surge
VERY_LOW = [60] * 24


EDGE_SCENARIOS = [
    # ── 1. No solar at all — pure grid/battery scheduling ────────────────────
    {
        "id": "EDGE-01", "label": "No solar at all",
        "hours": _hours(FLAT_200, NO_SOLAR, PEAK_EVE),
        "battery": _bat(cap=300, init=150),
        "directives": [],
        "note": "Optimal: charge at night (cheap), discharge in evening (expensive).",
    },

    # ── 2. Battery starts full — no room to charge ───────────────────────────
    {
        "id": "EDGE-02", "label": "Battery starts at capacity",
        "hours": _hours(FLAT_200, DAYTIME, PEAK_EVE),
        "battery": _bat(cap=300, init=300, min_e=30),
        "directives": [],
        "note": "Cannot charge more. Must discharge first to make room.",
    },

    # ── 3. Battery starts at minimum reserve — barely any discharge ──────────
    {
        "id": "EDGE-03", "label": "Battery starts at minimum reserve",
        "hours": _hours(FLAT_200, DAYTIME, PEAK_EVE),
        "battery": _bat(cap=300, init=30, min_e=30),
        "directives": [],
        "note": "Cannot discharge at all initially. Charge first, then discharge.",
    },

    # ── 4. Solar exceeds demand — curtailment required ───────────────────────
    {
        "id": "EDGE-04", "label": "Solar exceeds demand at midday",
        "hours": _hours(FLAT_100, HEAVY_SOLAR, PEAK_EVE),
        "battery": _bat(cap=300, init=100, min_e=20, max_c=100, max_d=100),
        "directives": [],
        "note": "Excess solar is curtailed. Grid export not allowed. "
                "Battery soaks up surplus, discharges at peak.",
    },

    # ── 5. solar_reduction during main solar hours ───────────────────────────
    {
        "id": "EDGE-05", "label": "Heavy solar reduction during peak solar hours",
        "hours": _hours(EVE_HEAVY, DAYTIME, PEAK_EVE),
        "battery": _bat(),
        "directives": [{"note_index": 0, "applies": True,
                        "directive_type": "solar_reduction",
                        "structured_adjustment": {"hours": list(range(9, 16)), "factor": 0.1},
                        "explanation": "90% solar reduction 9 AM–4 PM."}],
        "note": "Only 10% of peak solar available 9-15h. Grid must fill the gap.",
    },

    # ── 6. no_charge + no_discharge in same hours (forced idle) ──────────────
    {
        "id": "EDGE-06", "label": "Battery forced idle during expensive hours",
        "hours": _hours(EVE_HEAVY, DAYTIME, PEAK_EVE),
        "battery": _bat(),
        "directives": [
            {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
             "structured_adjustment": {"hours": [18, 19, 20, 21]},
             "explanation": "No charge 6-10 PM."},
            {"note_index": 1, "applies": True, "directive_type": "no_discharge_window",
             "structured_adjustment": {"hours": [18, 19, 20, 21]},
             "explanation": "No discharge 6-10 PM."},
        ],
        "note": "Battery must be idle 6-10 PM — the most expensive hours. "
                "Grid must cover all evening demand.",
    },

    # ── 7. Very tight max_grid_window — battery must carry the load ──────────
    {
        "id": "EDGE-07", "label": "Tight grid cap forces heavy battery use",
        "hours": _hours([150] * 24, LIGHT_SOLAR, PEAK_EVE),
        "battery": _bat(cap=400, init=100, min_e=20, max_c=100, max_d=100),
        "directives": [
            {"note_index": 0, "applies": True, "directive_type": "max_grid_window",
             "structured_adjustment": {"hours": list(range(18, 24)), "max_grid_kwh": 100},
             "explanation": "Grid capped at 100 kWh/h 6 PM–midnight."},
        ],
        "note": "Demand=150, grid cap=100, so battery must supply 50 kWh/h for 6 hours "
                "(300 kWh) and still end at 100 — it must be exactly full (400) by 6 PM.",
    },

    # ── 8. High minimum reserve during discharge window ──────────────────────
    {
        "id": "EDGE-08", "label": "High reserve with no_discharge together",
        "hours": _hours(EVE_HEAVY, DAYTIME, PEAK_EVE),
        "battery": _bat(cap=300, init=150),
        "directives": [
            {"note_index": 0, "applies": True,
             "directive_type": "minimum_battery_reserve",
             "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 200},
             "explanation": "Reserve 200 kWh 6-9 PM."},
            {"note_index": 1, "applies": True,
             "directive_type": "no_discharge_window",
             "structured_adjustment": {"hours": [12, 13, 14, 15, 16]},
             "explanation": "No discharge noon–5 PM."},
        ],
        "note": "Must hold 200 kWh reserve 6-9 PM and cannot discharge noon-5 PM, "
                "then drain back to 150 in the last three hours.",
    },

    # ── 9. All tariffs equal — optimizer must still satisfy constraints ───────
    {
        "id": "EDGE-09", "label": "Flat tariff — any valid plan is optimal",
        "hours": _hours(FLAT_200, DAYTIME, CHEAP),
        "battery": _bat(),
        "directives": [
            {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
             "structured_adjustment": {"hours": list(range(8, 18))},
             "explanation": "No charge during business hours."},
        ],
        "note": "Tariff is flat so cost is the same regardless of timing. "
                "But constraints must still be satisfied and battery must neutralise.",
    },

    # ── 10. Very small battery, high demand — mostly grid-dependent ───────────
    {
        "id": "EDGE-10", "label": "Tiny battery, high demand",
        "hours": _hours(FLAT_200, DAYTIME, PEAK_EVE),
        "battery": _bat(cap=40, init=20, min_e=5, max_c=15, max_d=15),
        "directives": [],
        "note": "Battery is too small to make a big difference. Mostly grid. "
                "Optimizer should still use it optimally.",
    },

    # ── 11. minimum_battery_reserve = capacity (nearly impossible) — actually
    #        organiser says scenarios are always feasible, so cap-1 ───────────
    {
        "id": "EDGE-11", "label": "Reserve close to capacity",
        "hours": _hours(FLAT_100, DAYTIME, PEAK_EVE),
        "battery": _bat(cap=300, init=280, min_e=250, max_c=50, max_d=50),
        "directives": [
            {"note_index": 0, "applies": True,
             "directive_type": "minimum_battery_reserve",
             "structured_adjustment": {"hours": list(range(18, 24)), "minimum_energy_kwh": 260},
             "explanation": "Reserve 260 kWh all evening."},
        ],
        "note": "Very high reserve: battery level can barely move in the evening. "
                "Optimizer must plan charging early enough.",
    },

    # ── 12. Midnight-spanning no_charge + solar_reduction combo ──────────────
    {
        "id": "EDGE-12", "label": "Combined: solar cut + no-charge overnight",
        "hours": _hours(EVE_HEAVY, DAYTIME, VALLEY_PEAK),
        "battery": _bat(cap=300, init=150, max_c=100, max_d=100),
        "directives": [
            {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
             "structured_adjustment": {"hours": list(range(10, 15)), "factor": 0.2},
             "explanation": "Solar at 20% 10 AM–3 PM."},
            {"note_index": 1, "applies": True, "directive_type": "no_charge_window",
             "structured_adjustment": {"hours": list(range(0, 6)) + list(range(22, 24))},
             "explanation": "No charging midnight–6 AM and 10 PM–midnight."},
        ],
        "note": "Solar is cut during its best hours AND charging blocked overnight. "
                "Must charge in the remaining window (6 AM-10 PM minus solar-cut hours).",
    },
]


def run_optimizer_edges() -> tuple[int, int]:
    """Run all edge-case scenarios through the optimizer and replay checker."""
    ok = total = 0
    for sc in EDGE_SCENARIOS:
        total += 1
        try:
            plan = optimize(sc["hours"], sc["battery"], sc["directives"])
            tot = totals(plan, sc["hours"])
            response = {"hourly_plan": plan, **tot}
            errs = check_plan(sc, sc["directives"], response)
            if errs:
                print(f"FAIL  {sc['id']:<10} {sc['label']}")
                for e in errs[:5]:
                    print(f"        {e}")
            else:
                ok += 1
                print(f"PASS  {sc['id']:<10} {sc['label']}  "
                      f"cost={tot['total_cost_bdt']:.2f} BDT")
        except OptimizationError as e:
            print(f"FAIL  {sc['id']:<10} {sc['label']}  OptimizationError: {e}")
        except Exception as e:
            print(f"FAIL  {sc['id']:<10} {sc['label']}  {type(e).__name__}: {e}")
    return ok, total


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrase", action="store_true",
                    help="Also run LLM paraphrase tests (needs GROQ_API_KEY, costs quota)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=" * 60)
    print("B. OPTIMIZER EDGE CASES")
    print("=" * 60)
    eo, et = run_optimizer_edges()
    print(f"\n{eo}/{et} edge scenarios valid\n")

    po, pt = 0, 0
    if args.paraphrase:
        print("=" * 60)
        print("A. LLM PARAPHRASE TESTS")
        print("=" * 60)
        po, pt = run_paraphrase_tests()
        print(f"\n{po}/{pt} paraphrase cases correct\n")

    print(f"Total time: {time.perf_counter() - t0:.1f}s")
    total_pass = eo + po
    total_all = et + pt
    return 0 if total_pass == total_all else 1


if __name__ == "__main__":
    sys.exit(main())
