"""Offline tests: the LLM is mocked, everything else is real.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import interpreter  # noqa: E402
from app.interpreter import normalise  # noqa: E402
from app.llm import LLMError  # noqa: E402
from app.main import app  # noqa: E402
from app.replay import check_interpretation, check_plan  # noqa: E402

CASES = json.loads((ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")
                   .read_text(encoding="utf-8"))["cases"]
client = TestClient(app, raise_server_exceptions=False)


def raw(kind, windows, value=None, meaning=None):
    return {"directive_type": kind, "windows": windows, "value": value,
            "value_meaning": meaning, "explanation": "x"}


@pytest.fixture(autouse=True)
def _clear_cache():
    interpreter._interpret_cached.cache_clear()


# ---- normalisation: raw LLM semantics -> exact structured_adjustment ----

@pytest.mark.parametrize("r, expected", [
    (raw("solar_reduction", [[13, 15]], 80, "reduction_percent"), {"hours": [13, 14], "factor": 0.2}),
    (raw("solar_reduction", [[13, 15]], 20, "remaining_percent"), {"hours": [13, 14], "factor": 0.2}),
    (raw("solar_reduction", [[12, 14]], 0.25, "remaining_percent"), {"hours": [12, 13], "factor": 0.25}),
    (raw("minimum_battery_reserve", [[18, 21]], 50, "percent_of_capacity"),
     {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0}),
    (raw("max_grid_window", [[18, 21]], 155, "kwh"), {"hours": [18, 19, 20], "max_grid_kwh": 155.0}),
    (raw("no_charge_window", [[22, 2]]), {"hours": [0, 1, 22, 23]}),
    (raw("no_discharge_window", [[20, 24]]), {"hours": [20, 21, 22, 23]}),
    # recurring fractions must be exact, not 99.99
    # (capacity is 200 here, so a third is 66.6667 — not 66.66)
    (raw("minimum_battery_reserve", [[20, 23]], "1/3", "percent_of_capacity"),
     {"hours": [20, 21, 22], "minimum_energy_kwh": 66.6667}),
    (raw("minimum_battery_reserve", [[20, 23]], 33.33, "percent_of_capacity"),
     {"hours": [20, 21, 22], "minimum_energy_kwh": 66.6667}),
    (raw("solar_reduction", [[9, 11]], "2/3", "reduction_percent"), {"hours": [9, 10], "factor": 0.3333}),
    (raw("max_grid_window", [[17, 19]], 0.12e3, "kwh"), {"hours": [17, 18], "max_grid_kwh": 120.0}),
])
def test_normalise(r, expected):
    assert normalise(r, capacity=200)["structured_adjustment"] == expected


@pytest.mark.parametrize("r", [
    raw("turn_off_lights", [[1, 2]]),                       # unsupported type
    raw("no_charge_window", [[5, 5]]),                      # empty window
    raw("no_charge_window", [[3, 30]]),                     # out of range
    raw("solar_reduction", [[1, 2]], 50, None),             # ambiguous meaning
    raw("max_grid_window", [[1, 2]], float("nan"), "kwh"),  # non-finite
    raw("max_grid_window", [[1, 2]], "1/0", "kwh"),         # division by zero
    raw("max_grid_window", [[1, 2]], "lots", "kwh"),        # not a number
])
def test_normalise_rejects(r):
    with pytest.raises((ValueError, TypeError)):
        normalise(r, capacity=200)


# ---- full API with a mocked LLM that answers with the ground truth ----

def _truth_as_raw(case):
    """Build what a correct LLM reply would look like for this case."""
    out = []
    for e in case["expected_output"]["directive_interpretation"]:
        adj = e["structured_adjustment"] or {}
        hours = adj.get("hours", [])
        windows = [[hours[0], hours[-1] + 1]] if hours else []
        value, meaning = None, None
        if "factor" in adj:
            value, meaning = adj["factor"] * 100, "remaining_percent"
        elif "minimum_energy_kwh" in adj:
            value, meaning = adj["minimum_energy_kwh"], "kwh"
        elif "max_grid_kwh" in adj:
            value, meaning = adj["max_grid_kwh"], "kwh"
        out.append({"note_index": e["note_index"], **raw(e["directive_type"], windows, value, meaning)})
    return {"directives": out}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_samples_end_to_end(case, monkeypatch):
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: _truth_as_raw(case))
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    body = r.json()
    truth = case["expected_output"]["directive_interpretation"]
    assert body["scenario_id"] == case["input"]["scenario_id"]
    assert check_interpretation(body["directive_interpretation"], truth) == []
    assert check_plan(case["input"], truth, body) == []
    assert abs(body["total_cost_bdt"] - case["expected_output"]["total_cost_bdt"]) <= 0.01


def test_llm_down_is_safe(monkeypatch):
    def boom(*a, **k):
        raise LLMError("down")
    monkeypatch.setattr(interpreter, "chat_json", boom)
    r = client.post("/optimize-energy", json=CASES[5]["input"])
    assert r.status_code == 200
    body = r.json()
    assert all(e["directive_type"] == "no_op" and e["applies"] is False
               for e in body["directive_interpretation"])
    assert len(body["hourly_plan"]) == 24


def test_garbage_llm_output_is_safe(monkeypatch):
    bad = {"directives": [{"note_index": 0, "directive_type": "explode_battery", "windows": [[1, 2]]}]}
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: bad)
    r = client.post("/optimize-energy", json=CASES[1]["input"])
    assert r.status_code == 200
    assert r.json()["directive_interpretation"][0]["directive_type"] == "no_op"


def test_infeasible_directive_is_dropped(monkeypatch):
    # A reserve of 100% capacity all day cannot coexist with ending at initial energy < capacity.
    reply = {"directives": [
        {"note_index": 0, **raw("minimum_battery_reserve", [[0, 24]], 100, "percent_of_capacity")},
        {"note_index": 1, **raw("no_op", [])},
    ]}
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: reply)
    r = client.post("/optimize-energy", json=CASES[0]["input"])
    assert r.status_code == 200
    assert "not enforced" in r.json()["plan_summary"]


# ---- request validation ----

def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("mutate", [
    lambda b: b.pop("battery"),
    lambda b: b.update(operator_notes=[]),
    lambda b: b.update(operator_notes=["a", "b", "c", "d"]),
    lambda b: b.update(operator_notes=["   "]),
    lambda b: b.update(hours=b["hours"][:23]),
    lambda b: b["hours"][5].update(hour=4),
    lambda b: b["hours"][0].update(demand_kwh=-5),
    lambda b: b["hours"][0].update(solar_kwh="lots"),
    lambda b: b["battery"].update(minimum_energy_kwh=10_000),
])
def test_bad_requests_400(mutate):
    body = json.loads(json.dumps(CASES[0]["input"]))
    mutate(body)
    assert client.post("/optimize-energy", json=body).status_code == 400


def test_malformed_json_400():
    r = client.post("/optimize-energy", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400


# ---- AM/PM safety net for solar notes ----

def test_solar_am_misread_is_shifted_to_pm(monkeypatch):
    """LLM reads 'one until three' as 1-3 AM; there is no solar then, so shift to 1-3 PM."""
    case = json.loads(json.dumps(CASES[0]))
    case["input"]["operator_notes"] = [
        "Panel washing from one until three will leave roughly one-fifth of normal solar output.",
        "The sports office moved next month's registration deadline."]
    reply = {"directives": [
        {"note_index": 0, **raw("solar_reduction", [[1, 3]], 20, "remaining_percent")},
        {"note_index": 1, **raw("no_op", [])},
    ]}
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: reply)
    body = client.post("/optimize-energy", json=case["input"]).json()
    assert body["directive_interpretation"][0]["structured_adjustment"]["hours"] == [13, 14]


def test_real_morning_solar_is_not_shifted(monkeypatch):
    """A reduction on morning hours that do have solar must be left alone."""
    case = CASES[0]
    reply = {"directives": [
        {"note_index": 0, **raw("solar_reduction", [[9, 11]], 20, "remaining_percent")},
        {"note_index": 1, **raw("no_op", [])},
    ]}
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: reply)
    body = client.post("/optimize-energy", json=case["input"]).json()
    assert body["directive_interpretation"][0]["structured_adjustment"]["hours"] == [9, 10]


def test_reserve_relative_to_base_minimum():
    r = raw("minimum_battery_reserve", [[14, 17]], 30, "kwh_above_base_minimum")
    assert normalise(r, capacity=180, base_min=30)["structured_adjustment"] == \
        {"hours": [14, 15, 16], "minimum_energy_kwh": 60.0}


def test_explicit_midnight_solar_is_not_shifted(monkeypatch):
    """'from midnight (00:00) until 01:00' means hour 0, even though there is no solar then."""
    case = json.loads(json.dumps(CASES[0]))
    case["input"]["operator_notes"] = ["PV output is halved from midnight (00:00) until 01:00 for a planned test."]
    reply = {"directives": [{"note_index": 0, **raw("solar_reduction", [[0, 1]], 50, "remaining_percent")}]}
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: reply)
    body = client.post("/optimize-energy", json=case["input"]).json()
    assert body["directive_interpretation"][0]["structured_adjustment"]["hours"] == [0]


@pytest.mark.parametrize("reply", [
    [{"note_index": 0, "directive_type": "no_charge_window", "windows": [[3, 5]]}],             # bare list
    {"interpretations": [{"note_index": 0, "directive_type": "no_charge_window", "windows": [[3, 5]]}]},
    {"note_index": 0, "directive_type": "no_charge_window", "windows": [[3, 5]]},               # single object
])
def test_tolerates_model_json_shape_slips(reply, monkeypatch):
    case = json.loads(json.dumps(CASES[1]))
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: reply)
    body = client.post("/optimize-energy", json=case["input"]).json()
    assert body["directive_interpretation"][0]["structured_adjustment"] == {"hours": [3, 4]}


def test_bad_shape_then_good_reply_recovers(monkeypatch):
    replies = iter([{"oops": "no list"},
                    {"directives": [{"note_index": 0, "directive_type": "no_charge_window", "windows": [[2, 5]]}]}])
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: next(replies))
    body = client.post("/optimize-energy", json=CASES[1]["input"]).json()
    assert body["directive_interpretation"][0]["directive_type"] == "no_charge_window"


def test_secret_in_model_explanation_is_redacted(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_realKEYvalue1234567890")
    reply = {"directives": [
        {"note_index": 0, **raw("no_charge_window", [[2, 5]]),
         "explanation": "Key gsk_realKEYvalue1234567890 and sk-otherSecretValue123456 here"}]}
    monkeypatch.setattr(interpreter, "chat_json", lambda *a, **k: reply)
    r = client.post("/optimize-energy", json=CASES[1]["input"])
    assert "gsk_" not in r.text and "sk-other" not in r.text
    assert "[redacted]" in r.json()["directive_interpretation"][0]["explanation"]
