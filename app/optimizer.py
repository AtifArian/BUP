"""GridWise 24-hour battery/grid scheduler, solved as a linear program.

Variables per hour h (24 of each, 120 total):
    g[h]  grid import (kWh)
    s[h]  solar used (kWh)
    c[h]  battery charge (kWh)
    d[h]  battery discharge (kWh)
    e[h]  battery energy after hour h (kWh)

Constraints:
    g + s + d - c = demand                 (energy balance)
    e[h] = e[h-1] + c[h] - d[h]            (battery transition, e[-1] = initial)
    e[23] = initial                        (end-of-day neutrality)
    bounds carry solar, rate limits, reserve, capacity and every directive.

Solved in two phases: (1) minimise grid cost, (2) keep that cost and minimise
total battery throughput so the schedule has no pointless cycling and never
charges and discharges in the same hour.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

H = 24
G, S, C, D, E = range(5)  # variable blocks
N = 5 * H

ROUND = 4  # decimals in the returned plan (judge tolerance is 0.01)


class OptimizationError(Exception):
    """Raised when no valid schedule exists for the given inputs."""


def _idx(block: int, h: int) -> int:
    return block * H + h


def effective_bounds(hours: list[dict], battery: dict, directives: list[dict]) -> dict:
    """Turn the scenario plus validated directives into per-hour limits.

    `directives` are interpretation entries (directive_type + structured_adjustment);
    no_op entries are ignored. Shared with the replay checker so both apply the
    exact same rules.
    """
    solar = [float(h["solar_kwh"]) for h in hours]
    min_e = [float(battery["minimum_energy_kwh"])] * H
    max_c = [float(battery["max_charge_kwh_per_hour"])] * H
    max_d = [float(battery["max_discharge_kwh_per_hour"])] * H
    max_g = [np.inf] * H

    for dv in directives:
        kind = dv["directive_type"]
        adj = dv.get("structured_adjustment") or {}
        for h in adj.get("hours", []):
            if kind == "solar_reduction":
                solar[h] = float(hours[h]["solar_kwh"]) * float(adj["factor"])
            elif kind == "minimum_battery_reserve":
                min_e[h] = max(min_e[h], float(adj["minimum_energy_kwh"]))
            elif kind == "no_charge_window":
                max_c[h] = 0.0
            elif kind == "no_discharge_window":
                max_d[h] = 0.0
            elif kind == "max_grid_window":
                max_g[h] = min(max_g[h], float(adj["max_grid_kwh"]))

    return {"solar": solar, "min_e": min_e, "max_c": max_c, "max_d": max_d, "max_g": max_g}


def _build(hours: list[dict], battery: dict, lim: dict):
    demand = [float(h["demand_kwh"]) for h in hours]
    init = float(battery["initial_energy_kwh"])
    cap = float(battery["capacity_kwh"])

    a_eq = np.zeros((2 * H + 1, N))
    b_eq = np.zeros(2 * H + 1)
    for h in range(H):
        # energy balance
        a_eq[h, _idx(G, h)] = 1
        a_eq[h, _idx(S, h)] = 1
        a_eq[h, _idx(D, h)] = 1
        a_eq[h, _idx(C, h)] = -1
        b_eq[h] = demand[h]
        # battery transition: e[h] - e[h-1] - c[h] + d[h] = 0 (or = init at h=0)
        r = H + h
        a_eq[r, _idx(E, h)] = 1
        a_eq[r, _idx(C, h)] = -1
        a_eq[r, _idx(D, h)] = 1
        if h == 0:
            b_eq[r] = init
        else:
            a_eq[r, _idx(E, h - 1)] = -1
    a_eq[2 * H, _idx(E, H - 1)] = 1
    b_eq[2 * H] = init

    bounds = []
    for h in range(H):
        bounds.append((0, None if np.isinf(lim["max_g"][h]) else lim["max_g"][h]))
    bounds += [(0, lim["solar"][h]) for h in range(H)]
    bounds += [(0, lim["max_c"][h]) for h in range(H)]
    bounds += [(0, lim["max_d"][h]) for h in range(H)]
    bounds += [(lim["min_e"][h], cap) for h in range(H)]
    return a_eq, b_eq, bounds


def optimize(hours: list[dict], battery: dict, directives: list[dict]) -> list[dict]:
    """Return a cost-optimal 24-entry hourly_plan, or raise OptimizationError."""
    hours = sorted(hours, key=lambda h: h["hour"])
    lim = effective_bounds(hours, battery, directives)
    for h in range(H):
        if lim["min_e"][h] > float(battery["capacity_kwh"]) + 1e-9:
            raise OptimizationError(f"reserve above capacity at hour {h}")

    a_eq, b_eq, bounds = _build(hours, battery, lim)
    tariff = np.array([float(h["tariff_bdt_per_kwh"]) for h in hours])

    cost = np.zeros(N)
    cost[G * H:(G + 1) * H] = tariff
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise OptimizationError(f"no feasible schedule ({res.message})")

    # Phase 2: same optimal cost, minimum battery throughput.
    throughput = np.zeros(N)
    throughput[C * H:(D + 1) * H] = 1
    res2 = linprog(
        throughput,
        A_ub=cost.reshape(1, -1),
        b_ub=[res.fun + 1e-4],  # far below the judge's 0.01 BDT tolerance
        A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs",
    )
    x = res2.x if res2.status == 0 else res.x

    return _to_plan(x, hours, battery, lim)


def _r(v: float) -> float:
    v = round(float(v), ROUND)
    return 0.0 if v == 0 else v  # drop -0.0


def _to_plan(x, hours: list[dict], battery: dict, lim: dict) -> list[dict]:
    """Convert the LP vector into plan entries that replay exactly.

    Battery levels are rounded first and every other value is derived from
    them, so transitions and energy balance hold after rounding.
    """
    init = float(battery["initial_energy_kwh"])
    levels = [_r(x[_idx(E, h)]) for h in range(H)]
    levels[-1] = _r(init)

    plan = []
    prev = init
    for h in range(H):
        net = _r(levels[h] - prev)  # >0 charge, <0 discharge
        if net > 0:
            action, amount = "charge", net
        elif net < 0:
            action, amount = "discharge", -net
        else:
            action, amount = "idle", 0.0

        demand = float(hours[h]["demand_kwh"])
        need = demand + net  # grid + solar must cover this
        solar = min(_r(x[_idx(S, h)]), lim["solar"][h], max(need, 0.0))
        grid = _r(need - solar)
        if grid < 0:  # rounding residue: shave solar instead of exporting
            solar, grid = _r(need), 0.0
        plan.append({
            "hour": h,
            "grid_kwh": grid,
            "solar_used_kwh": _r(solar),
            "battery_action": action,
            "battery_kwh": amount,
            "battery_energy_after_kwh": levels[h],
        })
        prev = levels[h]
    return plan


def totals(plan: list[dict], hours: list[dict]) -> dict:
    """Top-level totals recomputed from the plan (the judge's source of truth)."""
    tariff = {h["hour"]: float(h["tariff_bdt_per_kwh"]) for h in hours}
    grid = [p["grid_kwh"] for p in plan]
    return {
        "total_grid_kwh": _r(sum(grid)),
        "total_cost_bdt": _r(sum(p["grid_kwh"] * tariff[p["hour"]] for p in plan)),
        "peak_grid_kwh": _r(max(grid)),
    }
