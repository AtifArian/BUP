"""Unusual-but-possible request formats (Guide §7 'request validation', §8 'malformed input').

Offline: the LLM is mocked. Every case must give a deliberate status code,
never a 500 or a crash.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import interpreter  # noqa: E402
from app.main import app  # noqa: E402

CASES = json.loads((ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")
                   .read_text(encoding="utf-8"))["cases"]
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    """Every note becomes no_op, so these tests exercise only the request path."""
    interpreter._interpret_cached.cache_clear()

    def reply(system, user, **_):
        n = user.count("\n") - 1  # one line per note after the two header lines
        return {"directives": [{"note_index": i, "directive_type": "no_op", "windows": [],
                                "value": None, "value_meaning": None, "explanation": "x"}
                               for i in range(max(n, 1))]}
    monkeypatch.setattr(interpreter, "chat_json", reply)


def body():
    return json.loads(json.dumps(CASES[0]["input"]))


def post(b, **kw):
    return client.post("/optimize-energy", json=b, **kw)


# ---- accepted: tolerant of harmless variations ----

def test_extra_unknown_fields_are_ignored():
    b = body()
    b["weather"] = "sunny"
    b["hours"][0]["co2_g_per_kwh"] = 500
    b["battery"]["chemistry"] = "LFP"
    r = post(b)
    assert r.status_code == 200
    assert "weather" not in r.json()


def test_hours_out_of_order_are_sorted():
    b = body()
    b["hours"].reverse()
    r = post(b)
    assert r.status_code == 200
    assert [p["hour"] for p in r.json()["hourly_plan"]] == list(range(24))


def test_whole_numbers_written_as_floats():
    b = body()
    for h in b["hours"]:
        h["hour"] = float(h["hour"])  # 0.0, 1.0, ...
    b["battery"] = {k: float(v) for k, v in b["battery"].items()}
    assert post(b).status_code == 200


def test_unicode_scenario_id_echoed_exactly():
    b = body()
    b["scenario_id"] = "দৃশ্য-১০১ ☀️ «test» \"quoted\""
    r = post(b)
    assert r.status_code == 200
    assert r.json()["scenario_id"] == b["scenario_id"]


def test_missing_content_type_header_still_parsed():
    r = client.post("/optimize-energy", content=json.dumps(body()).encode(), headers={"Content-Type": ""})
    assert r.status_code in (200, 400)  # either is a controlled answer; never 500


def test_response_types_are_exact():
    r = post(body())
    d = r.json()
    assert r.headers["content-type"].startswith("application/json")
    for e in d["directive_interpretation"]:
        assert type(e["note_index"]) is int and type(e["applies"]) is bool
        assert isinstance(e["explanation"], str)
    for p in d["hourly_plan"]:
        assert type(p["hour"]) is int
        assert p["battery_action"] in ("charge", "discharge", "idle")
        for k in ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"):
            assert isinstance(p[k], (int, float)) and not isinstance(p[k], bool) and math.isfinite(p[k])
    for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        assert isinstance(d[k], (int, float)) and math.isfinite(d[k])
    assert isinstance(d["plan_summary"], str) and d["plan_summary"]


def test_huge_but_finite_values_do_not_crash():
    b = body()
    for h in b["hours"]:
        h["demand_kwh"] = 1e9
    assert post(b).status_code in (200, 422)


def test_zero_everything_is_fine():
    b = body()
    for h in b["hours"]:
        h.update(demand_kwh=0, solar_kwh=0, tariff_bdt_per_kwh=0)
    r = post(b)
    assert r.status_code == 200
    assert r.json()["total_cost_bdt"] == 0


# ---- rejected with a controlled 400 ----

@pytest.mark.parametrize("raw", [
    b"",                         # empty body
    b"null",
    b"[]",                       # array instead of object
    b'"just a string"',
    b"{",                        # truncated
    b'{"scenario_id": "x", "operator_notes": ["a"], "hours": [], "battery": {}, }',  # trailing comma
    '{"scenario_id":"x","operator_notes":["a"],"hours":[{"hour":0,"demand_kwh":Infinity}]}'.encode(),
    b"\xff\xfe\x00garbage",      # not UTF-8
])
def test_malformed_bodies_400(raw):
    r = client.post("/optimize-energy", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 400


@pytest.mark.parametrize("mutate", [
    lambda b: b.update(scenario_id=""),
    lambda b: b.update(scenario_id=12345),
    lambda b: b.update(operator_notes="a single string, not a list"),
    lambda b: b.update(operator_notes=[123]),
    lambda b: b.update(operator_notes=[None]),
    lambda b: b.update(operator_notes=["x" * 5000]),
    lambda b: b["hours"][3].update(hour=3.5),
    lambda b: b["hours"][3].update(demand_kwh=True),
    lambda b: b["hours"][3].update(demand_kwh="120"),       # numeric string
    lambda b: b["hours"][3].update(hour="3"),
    lambda b: b["battery"].update(capacity_kwh=False),
    lambda b: b["hours"][3].update(tariff_bdt_per_kwh=None),
    lambda b: b["hours"].append(dict(b["hours"][0])),       # 25 entries
    lambda b: b["battery"].update(capacity_kwh=-1),
    lambda b: b["battery"].pop("max_discharge_kwh_per_hour"),
    lambda b: b.update(battery=[]),
    lambda b: b.update(hours={"0": {}}),
])
def test_structurally_invalid_400(mutate):
    b = body()
    mutate(b)
    assert post(b).status_code == 400


def test_non_json_content_type_is_controlled():
    r = client.post("/optimize-energy", content=b"scenario_id=x", headers={"Content-Type": "text/plain"})
    assert r.status_code in (400, 415)


# ---- wrong method / path ----

def test_wrong_methods_are_not_500():
    assert client.get("/optimize-energy").status_code == 405
    assert client.post("/health").status_code == 405
    assert client.get("/does-not-exist").status_code == 404


def test_errors_never_leak_internals():
    r = client.post("/optimize-energy", content=b"{", headers={"Content-Type": "application/json"})
    text = r.text.lower()
    assert "traceback" not in text and "gsk_" not in text and "file \"" not in text
