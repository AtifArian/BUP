"""Live LLM check: interpret reworded notes and compare with expected directives.

Needs GROQ_API_KEY (reads .env).  python tests/check_paraphrases.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.interpreter import interpret  # noqa: E402
from app.replay import check_interpretation  # noqa: E402

CAP = 400  # battery capacity used for percentage notes


def d(kind, hours=None, **values):
    if kind == "no_op":
        return {"applies": False, "directive_type": "no_op", "structured_adjustment": None}
    return {"applies": True, "directive_type": kind,
            "structured_adjustment": {"hours": hours, **values}}


CASES = [
    # solar_reduction
    ("PV production will drop to about 20% between 13:00 and 15:00.", d("solar_reduction", [13, 14], factor=0.2)),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", d("solar_reduction", [13, 14], factor=0.2)),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", d("solar_reduction", [13, 14], factor=0.2)),
    ("Heavy haze is forecast: only 40 percent of predicted solar will be usable from 9 AM to noon.", d("solar_reduction", [9, 10, 11], factor=0.4)),
    ("Solar array will be completely offline between 10 and 11 AM for rewiring.", d("solar_reduction", [10], factor=0.0)),
    ("Inverter derating will cut solar by a quarter from 11:00 to 14:00.", d("solar_reduction", [11, 12, 13], factor=0.75)),
    # minimum_battery_reserve
    ("Maintain a minimum of 150 kWh stored from 7 PM to 11 PM for the hospital wing.", d("minimum_battery_reserve", [19, 20, 21, 22], minimum_energy_kwh=150)),
    ("Battery state of charge must not fall below 25% of capacity between 17:00 and 20:00.", d("minimum_battery_reserve", [17, 18, 19], minimum_energy_kwh=100)),
    ("Security wants the battery kept at least half full from 8 PM until midnight.", d("minimum_battery_reserve", [20, 21, 22, 23], minimum_energy_kwh=200)),
    # no_charge_window
    ("The charging inverter is being serviced 3 AM to 6 AM, so no energy can go into the battery.", d("no_charge_window", [3, 4, 5])),
    ("Battery charging is prohibited from 10:00 until 12:00.", d("no_charge_window", [10, 11])),
    # no_discharge_window
    ("Discharge path is locked out for relay calibration between 4 PM and 6 PM.", d("no_discharge_window", [16, 17])),
    ("Do not draw power from the battery from 21:00 to 23:00.", d("no_discharge_window", [21, 22])),
    # max_grid_window
    ("The utility has asked us to keep grid draw under 200 kWh per hour from 6 PM to 8 PM.", d("max_grid_window", [18, 19], max_grid_kwh=200)),
    ("Feeder maintenance limits imports to 0.15 MWh each hour between 12 and 2 PM.", d("max_grid_window", [12, 13], max_grid_kwh=150)),
    # no_op distractors
    ("The cafeteria menu changes tomorrow.", d("no_op")),
    ("Solar panels will be cleaned next Monday morning.", d("no_op")),
    ("The electrical engineering club meets at 5 PM in room 402.", d("no_op")),
    ("Grid tariffs are expected to rise next quarter.", d("no_op")),
]


def main() -> int:
    ok = 0
    t0 = time.perf_counter()
    # Batch 3 notes per call, like real requests.
    for start in range(0, len(CASES), 3):
        batch = CASES[start:start + 3]
        entries, warnings = interpret([n for n, _ in batch], CAP)
        for (note, truth), got in zip(batch, entries):
            errs = check_interpretation([{**got, "note_index": 0}], [truth])
            ok += not errs
            print(f"{'PASS' if not errs else 'FAIL'}  {note}")
            for e in errs:
                print(f"        {e}   got={got['structured_adjustment']} ({got['directive_type']})")
        for w in warnings:
            print(f"        warning: {w}")
    print(f"\n{ok}/{len(CASES)} correct in {time.perf_counter() - t0:.1f}s")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
