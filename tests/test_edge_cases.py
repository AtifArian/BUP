"""Optimizer edge-case scenarios from hidden_candidates.py, run under pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.optimizer import optimize, totals  # noqa: E402
from app.replay import check_plan  # noqa: E402
from tests.hidden_candidates import EDGE_SCENARIOS  # noqa: E402


@pytest.mark.parametrize("sc", EDGE_SCENARIOS, ids=[s["id"] for s in EDGE_SCENARIOS])
def test_edge_scenario_is_valid(sc):
    plan = optimize(sc["hours"], sc["battery"], sc["directives"])
    response = {"hourly_plan": plan, **totals(plan, sc["hours"])}
    assert check_plan(sc, sc["directives"], response) == []


@pytest.mark.parametrize("sc", EDGE_SCENARIOS, ids=[s["id"] for s in EDGE_SCENARIOS])
def test_directives_never_make_it_cheaper(sc):
    """Adding constraints can only keep cost equal or raise it."""
    free = totals(optimize(sc["hours"], sc["battery"], []), sc["hours"])["total_cost_bdt"]
    held = totals(optimize(sc["hours"], sc["battery"], sc["directives"]), sc["hours"])["total_cost_bdt"]
    assert held >= free - 0.01
