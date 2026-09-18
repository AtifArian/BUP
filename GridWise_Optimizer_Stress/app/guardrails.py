"""Deterministic guardrails for interpreted directives (Problem Statement §08).

Nothing produced by the LLM reaches the optimizer unless it passes these checks.
"""

from __future__ import annotations

import math

ALLOWED = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
    "no_op": None,
}


def no_op(explanation: str) -> dict:
    return {"applies": False, "directive_type": "no_op",
            "structured_adjustment": None, "explanation": explanation}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_entry(entry: dict, capacity: float) -> list[str]:
    """Validate one directive_interpretation entry. Returns problems (empty = ok)."""
    kind = entry.get("directive_type")
    if kind not in ALLOWED:
        return [f"unsupported directive_type {kind!r}"]
    adj = entry.get("structured_adjustment")

    if kind == "no_op":
        ok = entry.get("applies") is False and adj is None
        return [] if ok else ["no_op must have applies=false and null adjustment"]

    problems = []
    if entry.get("applies") is not True:
        problems.append("non-no_op directive must have applies=true")
    if not isinstance(adj, dict) or set(adj) != ALLOWED[kind]:
        return problems + [f"{kind} adjustment must have keys {sorted(ALLOWED[kind])}"]

    hours = adj["hours"]
    if (not isinstance(hours, list) or not hours
            or not all(isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23 for h in hours)
            or hours != sorted(set(hours))):
        problems.append("hours must be non-empty unique ascending integers 0-23")

    if kind == "solar_reduction":
        f = adj["factor"]
        if not _finite(f) or not 0 <= f <= 1:
            problems.append("factor must be within [0, 1]")
    elif kind == "minimum_battery_reserve":
        v = adj["minimum_energy_kwh"]
        if not _finite(v) or v < 0 or v > capacity:
            problems.append("minimum_energy_kwh must be finite, >= 0 and <= capacity")
    elif kind == "max_grid_window":
        v = adj["max_grid_kwh"]
        if not _finite(v) or v < 0:
            problems.append("max_grid_kwh must be finite and >= 0")
    return problems


def check_all(entries: list[dict], n_notes: int, capacity: float) -> list[str]:
    """Validate the full list: coverage, order and every entry."""
    if len(entries) != n_notes:
        return [f"expected {n_notes} entries, got {len(entries)}"]
    problems = []
    for i, e in enumerate(entries):
        if e.get("note_index") != i:
            problems.append(f"entry {i} has note_index {e.get('note_index')}")
        problems += [f"note {i}: {p}" for p in check_entry(e, capacity)]
    return problems
