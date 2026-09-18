"""Independent replay of a response, mirroring the judge's checks.

Used by the service as a final self-check and by the sample test runner.
Every function returns a list of human-readable error strings (empty = valid).
"""

from __future__ import annotations

import math

from app.optimizer import H, effective_bounds

TOL = 0.01
ACTIONS = {"charge", "discharge", "idle"}
PLAN_FIELDS = ("hour", "grid_kwh", "solar_used_kwh", "battery_action",
               "battery_kwh", "battery_energy_after_kwh")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_plan(request: dict, directives: list[dict], response: dict) -> list[str]:
    """Replay hourly_plan against the scenario and the given (true) directives."""
    errs: list[str] = []
    hours = sorted(request["hours"], key=lambda h: h["hour"])
    bat = request["battery"]
    plan = response.get("hourly_plan")

    if not isinstance(plan, list) or len(plan) != H:
        return [f"hourly_plan must have {H} entries"]
    if sorted(p.get("hour") for p in plan) != list(range(H)):
        return ["hourly_plan hours must be exactly 0..23"]
    plan = sorted(plan, key=lambda p: p["hour"])

    lim = effective_bounds(hours, bat, directives)
    cap = float(bat["capacity_kwh"])
    e_prev = float(bat["initial_energy_kwh"])

    for p in plan:
        h = p["hour"]
        missing = [f for f in PLAN_FIELDS if f not in p]
        if missing:
            errs.append(f"h{h}: missing {missing}")
            continue
        nums = [p[f] for f in PLAN_FIELDS if f not in ("hour", "battery_action")]
        if not all(_num(v) for v in nums) or any(v < -TOL for v in nums):
            errs.append(f"h{h}: values must be finite and non-negative")
            continue
        act, amt = p["battery_action"], p["battery_kwh"]
        if act not in ACTIONS:
            errs.append(f"h{h}: bad battery_action {act!r}")
            continue

        charge = amt if act == "charge" else 0.0
        discharge = amt if act == "discharge" else 0.0
        if act == "idle" and abs(amt) > TOL:
            errs.append(f"h{h}: idle with battery_kwh={amt}")

        e_exp = e_prev + charge - discharge
        e = p["battery_energy_after_kwh"]
        if abs(e - e_exp) > TOL:
            errs.append(f"h{h}: battery transition {e_prev}->{e} but action implies {e_exp}")
        if e < lim["min_e"][h] - TOL:
            errs.append(f"h{h}: battery {e} below reserve {lim['min_e'][h]}")
        if e > cap + TOL:
            errs.append(f"h{h}: battery {e} above capacity {cap}")
        if charge > lim["max_c"][h] + TOL:
            errs.append(f"h{h}: charge {charge} exceeds limit {lim['max_c'][h]}")
        if discharge > lim["max_d"][h] + TOL:
            errs.append(f"h{h}: discharge {discharge} exceeds limit {lim['max_d'][h]}")
        if p["solar_used_kwh"] > lim["solar"][h] + TOL:
            errs.append(f"h{h}: solar {p['solar_used_kwh']} exceeds effective {lim['solar'][h]}")
        if p["grid_kwh"] > lim["max_g"][h] + TOL:
            errs.append(f"h{h}: grid {p['grid_kwh']} exceeds cap {lim['max_g'][h]}")

        supply = p["grid_kwh"] + p["solar_used_kwh"] + discharge
        use = float(hours[h]["demand_kwh"]) + charge
        if abs(supply - use) > TOL:
            errs.append(f"h{h}: energy balance {supply} != {use}")
        e_prev = e

    if abs(e_prev - float(bat["initial_energy_kwh"])) > TOL:
        errs.append(f"end battery {e_prev} != initial {bat['initial_energy_kwh']}")

    # Reported totals must match the plan.
    tariff = {h["hour"]: float(h["tariff_bdt_per_kwh"]) for h in hours}
    grid = [p["grid_kwh"] for p in plan]
    expect = {
        "total_grid_kwh": sum(grid),
        "total_cost_bdt": sum(p["grid_kwh"] * tariff[p["hour"]] for p in plan),
        "peak_grid_kwh": max(grid),
    }
    for k, v in expect.items():
        got = response.get(k)
        if not _num(got) or abs(got - v) > TOL:
            errs.append(f"{k}={got} but plan gives {v:.4f}")
    return errs


def check_interpretation(got: list, expected: list) -> list[str]:
    """Compare a directive_interpretation list with ground truth."""
    if not isinstance(got, list) or len(got) != len(expected):
        return [f"expected {len(expected)} interpretation entries"]
    errs = []
    for i, (g, e) in enumerate(zip(got, expected)):
        if g.get("note_index") != i:
            errs.append(f"note {i}: note_index={g.get('note_index')}")
        for k in ("applies", "directive_type"):
            if g.get(k) != e[k]:
                errs.append(f"note {i}: {k}={g.get(k)!r}, expected {e[k]!r}")
        ga, ea = g.get("structured_adjustment"), e["structured_adjustment"]
        if ea is None or ga is None:
            if ga != ea:
                errs.append(f"note {i}: adjustment={ga!r}, expected {ea!r}")
            continue
        if set(ga) != set(ea):
            errs.append(f"note {i}: adjustment keys {sorted(ga)}, expected {sorted(ea)}")
            continue
        for k, v in ea.items():
            if k == "hours":
                if ga[k] != v:
                    errs.append(f"note {i}: hours {ga[k]}, expected {v}")
            elif not _num(ga[k]) or abs(ga[k] - v) > TOL:
                errs.append(f"note {i}: {k}={ga[k]}, expected {v}")
    return errs
