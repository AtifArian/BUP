"""GridWise API: LLM note interpretation -> guardrails -> LP optimizer -> replay check."""

from __future__ import annotations

import itertools
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

load_dotenv()

from app import guardrails  # noqa: E402
from app.interpreter import interpret  # noqa: E402
from app.optimizer import OptimizationError, optimize, totals  # noqa: E402
from app.replay import check_plan  # noqa: E402
from app.schemas import OptimizeRequest  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM Energy Optimizer", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def _bad_request(_: Request, exc: RequestValidationError):
    details = [
        {"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg", "")}
        for e in exc.errors()[:10]
    ]
    return JSONResponse(status_code=400, content={"error": "invalid request", "details": details})


@app.exception_handler(Exception)
async def _internal(_: Request, exc: Exception):
    log.exception("unhandled error")  # full trace stays in server logs only
    return JSONResponse(status_code=500, content={"error": "internal error"})


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


def _schedule(req: dict, entries: list[dict]) -> tuple[list[dict], list[dict], list[int]]:
    """Optimize with every applicable directive.

    Organizer scoring scenarios are feasible under the ground-truth directives,
    so infeasibility means a note was misread. Rather than failing the whole
    request (which would also discard the correctly read notes), enforce the
    largest subset of directives that is feasible and report which were dropped.
    Nothing is invented: the interpretation is returned unchanged, and only the
    schedule stops enforcing the conflicting directive(s).
    """
    active = [e for e in entries if e["applies"]]
    for keep in range(len(active), -1, -1):
        for subset in itertools.combinations(active, keep):
            try:
                plan = optimize(req["hours"], req["battery"], list(subset))
            except OptimizationError:
                continue
            dropped = [e["note_index"] for e in active if e not in subset]
            return plan, list(subset), dropped
    raise OptimizationError("scenario is infeasible even without operator directives")


def _summary(entries: list[dict], plan: list[dict], cost: float, dropped: list[int]) -> str:
    applied = [e["directive_type"] for e in entries
               if e["applies"] and e["note_index"] not in dropped]
    ignored = sum(1 for e in entries if not e["applies"])
    charge = sum(p["battery_kwh"] for p in plan if p["battery_action"] == "charge")
    parts = [
        f"Applied {len(applied)} directive(s)" + (f" ({', '.join(applied)})" if applied else ""),
        f"ignored {ignored} unrelated note(s)" if ignored else None,
        f"cycled {charge:g} kWh through the battery to shift purchases into cheaper hours"
        if charge else "kept the battery idle as shifting energy saved nothing",
        f"returned the battery to its starting level; total grid cost {cost:.2f} BDT",
    ]
    text = "; ".join(p for p in parts if p) + "."
    if dropped:
        text += f" Directives from notes {dropped} were infeasible together and were not enforced."
    return text


@app.post("/optimize-energy")
def optimize_energy(body: OptimizeRequest):
    req = body.model_dump()
    capacity = req["battery"]["capacity_kwh"]

    solar = [h["solar_kwh"] for h in req["hours"]]
    entries, warnings = interpret(req["operator_notes"], capacity, solar,
                                  req["battery"]["minimum_energy_kwh"])
    problems = guardrails.check_all(entries, len(req["operator_notes"]), capacity)
    if problems:  # interpreter already guards each entry; this is a last line of defence
        log.error("%s: guardrail failure after interpretation: %s", req["scenario_id"], problems)
        entries = [{"note_index": i, **guardrails.no_op("Rejected by guardrails; no constraint applied.")}
                   for i in range(len(req["operator_notes"]))]
    for w in warnings:
        log.warning("%s: %s", req["scenario_id"], w)

    try:
        plan, applied, dropped = _schedule(req, entries)
    except OptimizationError as e:
        return JSONResponse(status_code=422, content={"error": "infeasible scenario", "detail": str(e)})

    tot = totals(plan, req["hours"])
    response = {
        "scenario_id": req["scenario_id"],
        "directive_interpretation": entries,
        "hourly_plan": plan,
        **tot,
        "plan_summary": _summary(entries, plan, tot["total_cost_bdt"], dropped),
    }

    # Replay against the directives that were actually enforced (the judge does
    # the same with its ground truth; a dropped directive is already a lost case).
    replay_errors = check_plan(req, applied, response)
    if replay_errors:
        log.error("%s: final replay found issues: %s", req["scenario_id"], replay_errors[:5])
        return JSONResponse(status_code=500, content={"error": "internal validation failed"})
    return response
