"""Feed the edge pack's injected model outputs straight into our guardrails.

These fixtures are LLM outputs, not API requests, so they bypass Groq entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import guardrails  # noqa: E402

PACK = ROOT / "GridWise_Edge_Test_Pack" / "guardrail_injection_fixtures.json"
if not PACK.exists():
    pytest.skip("edge test pack not present", allow_module_level=True)

DATA = json.loads(PACK.read_text(encoding="utf-8"))
META = DATA["_meta"]
FIXTURES = DATA["fixtures"]


@pytest.mark.parametrize("fx", FIXTURES, ids=[f["name"] for f in FIXTURES])
def test_guardrail_fixture(fx):
    problems = guardrails.check_all(fx["model_output"], len(META["operator_notes"]),
                                    META["battery_capacity_kwh"])
    assert (not problems) == fx["expected_accept"], problems
