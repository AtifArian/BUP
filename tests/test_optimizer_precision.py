"""Precision edge cases found by the randomized optimizer stress pack."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.optimizer import optimize, totals  # noqa: E402
from app.replay import check_plan  # noqa: E402


def _flat_case(demand, solar, tariff=10.0):
    return [{"hour": h, "demand_kwh": demand, "solar_kwh": solar, "tariff_bdt_per_kwh": tariff}
            for h in range(24)]


def test_cap_a_hair_below_forced_grid_is_still_solved():
    """Battery cannot move, so grid is forced to 249.396775 but the cap says 249.39677.

    Exactly infeasible by 0.000005 kWh, yet valid under the judge's 0.01 tolerance:
    the directive must be honoured, not dropped.
    """
    hours = _flat_case(demand=249.931, solar=106.845)  # 0.5% of it = 0.534225
    battery = {"capacity_kwh": 53.362, "initial_energy_kwh": 49.082, "minimum_energy_kwh": 38.443,
               "max_charge_kwh_per_hour": 120.756, "max_discharge_kwh_per_hour": 0.0}
    all_day = list(range(24))
    directives = [
        {"directive_type": "no_charge_window", "structured_adjustment": {"hours": all_day}},
        {"directive_type": "solar_reduction", "structured_adjustment": {"hours": all_day, "factor": 0.005}},
        {"directive_type": "max_grid_window", "structured_adjustment": {"hours": all_day, "max_grid_kwh": 249.39677}},
    ]
    plan = optimize(hours, battery, directives)
    resp = {"hourly_plan": plan, **totals(plan, hours)}
    assert check_plan({"hours": hours, "battery": battery}, directives, resp) == []
    assert max(p["grid_kwh"] for p in plan) <= 249.39677 + 0.005 + 1e-9


def test_six_decimal_output_keeps_cost_exact():
    """Awkward decimals must not let 24 rounded hours drift the total by 0.01 BDT."""
    hours = [{"hour": h, "demand_kwh": 123.456789 + h * 1.111111, "solar_kwh": (h % 7) * 13.131313,
              "tariff_bdt_per_kwh": 29.99 if 17 <= h <= 21 else 7.77} for h in range(24)]
    battery = {"capacity_kwh": 333.333333, "initial_energy_kwh": 111.111111, "minimum_energy_kwh": 22.222222,
               "max_charge_kwh_per_hour": 77.777777, "max_discharge_kwh_per_hour": 66.666666}
    plan = optimize(hours, battery, [])
    resp = {"hourly_plan": plan, **totals(plan, hours)}
    assert check_plan({"hours": hours, "battery": battery}, [], resp) == []
    exact = sum(p["grid_kwh"] * hours[p["hour"]]["tariff_bdt_per_kwh"] for p in plan)
    assert abs(resp["total_cost_bdt"] - exact) < 1e-4
