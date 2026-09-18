"""Operator-note interpretation: LLM extraction -> deterministic normalisation.

The LLM reads every note and returns *raw* semantics (directive type, time
windows as start/end clock hours, the number as stated and what it means).
Deterministic code then expands windows into hour lists, converts percentages,
and builds the exact structured_adjustment. The guardrails module validates
the result before anything reaches the optimizer.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from functools import lru_cache

from app import guardrails
from app.llm import LLMError, chat_json

log = logging.getLogger("gridwise.interpreter")

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives for a 24-hour battery/solar/grid schedule (hours 0-23 of the SAME day being planned).

Every note maps to EXACTLY ONE of these directive types:
- solar_reduction: usable rooftop solar/PV output is reduced during some hours (cleaning, washing, inspection, cloud, inverter work, shading, outage).
- minimum_battery_reserve: the battery must keep AT LEAST some amount of stored energy during some hours.
- no_charge_window: the battery cannot be CHARGED during some hours (charger/charging circuit isolated, disabled, unavailable).
- no_discharge_window: the battery cannot be DISCHARGED during some hours (discharge blocked, protection/relay testing on discharge).
- max_grid_window: grid import/intake/draw must not exceed an amount per hour during some hours (feeder, transformer, substation limit).
- no_op: anything else. Use no_op for notes about other days (tomorrow, next week, next month), notes unrelated to solar/battery/grid (menus, bookings, notices, deadlines), and notes asking for anything not in the list above (e.g. demand or tariff changes). Never invent a rule.

Time rules:
- Report each time window as [start_hour, end_hour] using 24-hour clock hours. Windows are START-INCLUSIVE and END-EXCLUSIVE, so copy the stated clock times directly: "1 PM to 3 PM" -> [13, 15]; "from noon until 2 PM" -> [12, 14]; "2 AM until 5 AM" -> [2, 5]; "between 13:00 and 15:00" -> [13, 15].
- noon = 12, midnight at the end of a window = 24, midnight at the start = 0. "the whole day"/"all day" -> [0, 24].
- A single stated hour ("at 3 PM", "during the 3 PM hour") -> [15, 16]. "for N hours starting at X" -> [X, X+N].
- Use several windows only if the note names several separate periods.

Number rules (report the number exactly as the note states it; do NOT do arithmetic):
- solar_reduction: set "value" and "value_meaning":
    "remaining_percent" if the note says what is LEFT/usable ("drop to about 20%", "roughly 25% of the forecast", "half of normal" -> 50, "one-fifth" -> 20, "no solar at all" -> 0);
    "reduction_percent" if the note says how much is LOST ("an 80% reduction", "cut by 30%", "down 40%").
- minimum_battery_reserve: set "value" and "value_meaning": "kwh" for an energy amount, or "percent_of_capacity" for a share of battery capacity ("50% of capacity", "half full" -> 50).
- max_grid_window: set "value" in kWh per hour and "value_meaning": "kwh" (convert MWh to kWh: 0.2 MWh -> 200).
- no_charge_window, no_discharge_window, no_op: "value": null, "value_meaning": null.

Return ONLY a JSON object:
{"directives": [{"note_index": 0, "directive_type": "...", "windows": [[start, end]], "value": number or null, "value_meaning": "..." or null, "explanation": "one short sentence"}]}
Return exactly one entry per note, in note_index order. no_op entries use "windows": [].

Examples:
Note: "PV generation will only be about a third of normal from 9 until 11 in the morning." -> {"note_index": 0, "directive_type": "solar_reduction", "windows": [[9, 11]], "value": 33.33, "value_meaning": "remaining_percent", "explanation": "Solar limited to one third of forecast 9-11 AM."}
Note: "Solar will be down 60% between 14:00 and 17:00 due to shading." -> {"note_index": 0, "directive_type": "solar_reduction", "windows": [[14, 17]], "value": 60, "value_meaning": "reduction_percent", "explanation": "60% solar loss 2-5 PM."}
Note: "Hold the battery at no less than 150 kWh between 7 and 10 PM." -> {"note_index": 0, "directive_type": "minimum_battery_reserve", "windows": [[19, 22]], "value": 150, "value_meaning": "kwh", "explanation": "Battery reserve of 150 kWh 7-10 PM."}
Note: "The battery must stay at least 40% full from 5 PM to 8 PM." -> {"note_index": 0, "directive_type": "minimum_battery_reserve", "windows": [[17, 20]], "value": 40, "value_meaning": "percent_of_capacity", "explanation": "Reserve of 40% of capacity 5-8 PM."}
Note: "We cannot put energy into the battery from 1 AM to 4 AM." -> {"note_index": 0, "directive_type": "no_charge_window", "windows": [[1, 4]], "value": null, "value_meaning": null, "explanation": "Charging unavailable 1-4 AM."}
Note: "Battery output is locked out 8 PM-10 PM for inverter checks." -> {"note_index": 0, "directive_type": "no_discharge_window", "windows": [[20, 22]], "value": null, "value_meaning": null, "explanation": "Discharging unavailable 8-10 PM."}
Note: "Utility asks us to cap grid draw at 120 kWh per hour from 5 to 7 PM." -> {"note_index": 0, "directive_type": "max_grid_window", "windows": [[17, 19]], "value": 120, "value_meaning": "kwh", "explanation": "Grid import capped at 120 kWh 5-7 PM."}
Note: "The auditorium projector will be replaced next Tuesday." -> {"note_index": 0, "directive_type": "no_op", "windows": [], "value": null, "value_meaning": null, "explanation": "Unrelated to today's energy schedule."}"""


def _user_prompt(notes: list[str], capacity: float) -> str:
    listed = "\n".join(f"{i}: {json.dumps(n)}" for i, n in enumerate(notes))
    return (f"Battery capacity: {capacity:g} kWh (context only).\n"
            f"Operator notes ({len(notes)}):\n{listed}")


def _expand_windows(windows) -> list[int]:
    """[[start, end), ...] clock windows -> sorted unique hours 0-23 (wraps midnight)."""
    hours: set[int] = set()
    if not isinstance(windows, list):
        raise ValueError("windows must be a list")
    for w in windows:
        if not (isinstance(w, list) and len(w) == 2):
            raise ValueError(f"bad window {w!r}")
        start, end = (float(v) for v in w)
        if not (start.is_integer() and end.is_integer()):
            raise ValueError(f"non-whole-hour window {w!r}")
        start, end = int(start), int(end)
        if not (0 <= start <= 23 and 0 <= end <= 24):
            raise ValueError(f"window out of range {w!r}")
        if end > start:
            hours.update(range(start, end))
        elif end < start:  # crosses midnight, e.g. 10 PM - 2 AM
            hours.update(range(start, 24))
            hours.update(range(0, end))
        else:
            raise ValueError(f"empty window {w!r}")
    return sorted(hours)


def _number(v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"bad number {v!r}")
    return float(v)


def _percent(v: float) -> float:
    """Accept 25 or 0.25 for 25%. Returns a fraction."""
    return v / 100 if v > 1 else v


def normalise(raw: dict, capacity: float) -> dict:
    """Convert one raw LLM entry into a spec-shaped interpretation entry."""
    kind = raw.get("directive_type")
    explanation = str(raw.get("explanation") or "")[:300]
    if kind == "no_op":
        return guardrails.no_op(explanation or "Does not affect today's energy schedule.")

    hours = _expand_windows(raw.get("windows"))
    meaning = raw.get("value_meaning")
    if kind == "solar_reduction":
        frac = _percent(_number(raw.get("value")))
        if meaning == "reduction_percent":
            factor = 1 - frac
        elif meaning == "remaining_percent":
            factor = frac
        else:
            raise ValueError(f"unknown solar value_meaning {meaning!r}")
        adj = {"hours": hours, "factor": round(factor, 4)}
    elif kind == "minimum_battery_reserve":
        v = _number(raw.get("value"))
        if meaning == "percent_of_capacity":
            v = _percent(v) * capacity
        elif meaning != "kwh":
            raise ValueError(f"unknown reserve value_meaning {meaning!r}")
        adj = {"hours": hours, "minimum_energy_kwh": round(v, 4)}
    elif kind == "max_grid_window":
        adj = {"hours": hours, "max_grid_kwh": round(_number(raw.get("value")), 4)}
    elif kind in ("no_charge_window", "no_discharge_window"):
        adj = {"hours": hours}
    else:
        raise ValueError(f"unsupported directive_type {kind!r}")

    return {"applies": True, "directive_type": kind,
            "structured_adjustment": adj, "explanation": explanation}


def _interpret_once(notes: list[str], capacity: float, deadline: float) -> list[dict | None]:
    """One LLM call; returns a validated entry per note, or None where invalid."""
    out = chat_json(SYSTEM_PROMPT, _user_prompt(notes, capacity), deadline=deadline)
    raw = out.get("directives") if isinstance(out, dict) else None
    if not isinstance(raw, list):
        raise LLMError("model output has no 'directives' list")

    by_index = {}
    for pos, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        idx = entry.get("note_index", pos)
        if isinstance(idx, int) and 0 <= idx < len(notes) and idx not in by_index:
            by_index[idx] = entry

    results: list[dict | None] = []
    for i in range(len(notes)):
        entry = None
        try:
            if i in by_index:
                entry = normalise(by_index[i], capacity)
                entry = {"note_index": i, **entry}
                problems = guardrails.check_entry(entry, capacity)
                if problems:
                    log.warning("note %d rejected by guardrails: %s", i, problems)
                    entry = None
        except (ValueError, TypeError) as e:
            log.warning("note %d could not be normalised: %s", i, e)
            entry = None
        results.append(entry)
    return results


@lru_cache(maxsize=512)
def _interpret_cached(notes: tuple[str, ...], capacity: float) -> tuple:
    """Interpret with one retry for any note the guardrails rejected."""
    deadline = time.monotonic() + float(os.getenv("LLM_BUDGET_SECONDS", "20"))
    first = _interpret_once(list(notes), capacity, deadline)
    if all(first):
        return tuple(first)
    try:
        second = _interpret_once(list(notes), capacity, deadline)
    except LLMError:
        second = [None] * len(notes)
    merged = [a or b for a, b in zip(first, second)]
    if not all(merged):
        # Do not cache partial failures; a later request may succeed.
        raise _Partial(merged)
    return tuple(merged)


class _Partial(Exception):
    def __init__(self, entries):
        self.entries = entries


def interpret(notes: list[str], capacity: float) -> tuple[list[dict], list[str]]:
    """Return (directive_interpretation, warnings).

    Safe failure: a note the model cannot interpret validly becomes no_op with
    an explanation saying so, rather than an invented constraint or a crash.
    """
    warnings: list[str] = []
    try:
        entries = list(_interpret_cached(tuple(notes), float(capacity)))
    except _Partial as p:
        entries = p.entries
    except LLMError as e:
        log.error("interpretation failed: %s", e)
        entries = [None] * len(notes)
        warnings.append("language model unavailable")

    final = []
    for i, entry in enumerate(entries):
        if entry is None:
            warnings.append(f"note {i} could not be interpreted")
            entry = {"note_index": i, **guardrails.no_op(
                "Could not be interpreted reliably; no constraint applied.")}
        final.append(dict(entry))
    return final, warnings
