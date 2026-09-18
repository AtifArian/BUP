#!/usr/bin/env python3
"""Deterministic GridWise optimizer stress test. No LLM, network or keys.

Test cases are feasible by construction: a witness battery schedule, curtailed solar,
and grid imports are built BEFORE compatible directives are selected. Each optimized
plan is checked by both the project's replay.py and an independent replay here.
A separately formulated LP (grid eliminated) cross-checks the achieved grid cost.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app'
if not APP.is_dir():
    raise SystemExit('Missing app/: copy original 7 project files into app/ (see README)')
sys.path.insert(0, str(ROOT))
from app import optimizer, replay, guardrails  # noqa: E402

TOL = .01
CATEGORIES = [
    'normal', 'zero_demand', 'zero_rates', 'decimals', 'full_day_directives',
    'tight_grid', 'solar_surplus', 'all_zero',
]
KINDS = ['solar_reduction', 'minimum_battery_reserve', 'no_charge_window',
         'no_discharge_window', 'max_grid_window']


def rnd(rng, lo, hi, places=3):
    return round(rng.uniform(lo, hi), places)


def window_hours(rng, full=False):
    if full or rng.random() < .15:
        return list(range(24))
    if rng.random() < .13:
        return [rng.choice([0, 1, 22, 23])]
    if rng.random() < .13:
        return sorted(rng.sample(range(24), rng.randint(1, 23)))
    start = rng.randrange(24)
    end = min(24, start + rng.randint(1, 10))
    return list(range(start, end))


def mk_directive(i, kind, hrs, **kwargs):
    return dict(note_index=i, applies=True, directive_type=kind,
                structured_adjustment={'hours': hrs, **kwargs},
                explanation='Known synthetic directive; no LLM used.')


def effective_solar(hours, solar_directive):
    factors = [1.] * 24
    if solar_directive:
        a = solar_directive['structured_adjustment']
        for h in a['hours']:
            factors[h] = a['factor']
    return [hours[h]['solar_kwh'] * factors[h] for h in range(24)]


def build_case(rng, category, case_id):
    if category == 'all_zero':
        cap = init = min_e = rate_c = rate_d = 0.
    else:
        cap = rng.choice([0., 0.001, 1., 20., 125., 300.]) if rng.random() < .25 else rnd(rng, 1, 400)
        min_e = rnd(rng, 0, cap) if cap else 0.
        init = rnd(rng, min_e, cap) if cap else 0.
        if category == 'zero_rates':
            rate_c = rate_d = 0.
        else:
            rate_c = rng.choice([0., 0.01, 1., 15., 100.]) if rng.random() < .2 else rnd(rng, 0, 130)
            rate_d = rng.choice([0., 0.01, 1., 15., 100.]) if rng.random() < .2 else rnd(rng, 0, 130)
    battery = dict(capacity_kwh=cap, initial_energy_kwh=init,
                   minimum_energy_kwh=min_e, max_charge_kwh_per_hour=rate_c,
                   max_discharge_kwh_per_hour=rate_d)
    hrs = []
    for h in range(24):
        if category in ('zero_demand', 'all_zero'):
            dem = 0.
        elif category == 'decimals':
            dem = rng.choice([0., .0001, .001, .0111, 1.0001, rnd(rng, 0, 180, 4)])
        else:
            dem = rng.choice([0., rnd(rng, 0, 250, 4)]) if rng.random() < .2 else rnd(rng, 0, 250, 3)
        if category == 'all_zero':
            sol = tariff = 0.
        elif category == 'solar_surplus':
            sol = rnd(rng, dem + 1, dem + 400, 4)
            tariff = rnd(rng, 0, 65, 4)
        elif category == 'decimals':
            sol = rng.choice([0., .0001, rnd(rng, 0, 220, 4)])
            tariff = rng.choice([0., .0001, .0033, rnd(rng, 0, 55, 4)])
        else:
            sol = rng.choice([0., rnd(rng, 0, 260, 3)])
            tariff = rng.choice([0., rnd(rng, 0, 100, 3)])
        hrs.append(dict(hour=h, demand_kwh=dem, solar_kwh=sol, tariff_bdt_per_kwh=tariff))
    if category == 'all_zero':
        assert all(h['demand_kwh'] == h['solar_kwh'] == h['tariff_bdt_per_kwh'] == 0 for h in hrs)

    # Choose up to three real directives (exactly three for full-day cases).
    n_real = 3 if category == 'full_day_directives' else rng.choices([0, 1, 2, 3], [8, 23, 29, 40])[0]
    kinds = rng.sample(KINDS, n_real)
    if category == 'tight_grid' and 'max_grid_window' not in kinds:
        if kinds:
            kinds[-1] = 'max_grid_window'
        else:
            kinds = ['max_grid_window']
    if category == 'full_day_directives':
        kinds = rng.sample(['solar_reduction', 'minimum_battery_reserve',
                            'no_charge_window', 'no_discharge_window', 'max_grid_window'], 3)
    solar_d = None
    if 'solar_reduction' in kinds:
        factor = rng.choice([0., .0001, .005, .1, .2, .3333, .5, .75, 1.])
        solar_d = mk_directive(-1, 'solar_reduction', window_hours(rng, category == 'full_day_directives'),
                               factor=factor)
    solar = effective_solar(hrs, solar_d)

    # Independent, explicitly feasible witness: disjoint adjacent charge/discharge
    # pulses that return to the initial energy after every pair. For full-day,
    # zero-demand and all-zero tests idle is a guaranteed witness.
    net = [0.] * 24
    if category not in ('zero_demand', 'all_zero', 'full_day_directives'):
        for h in range(0, 24, 2):
            if rng.random() > .60:
                continue
            charge_first = rng.random() < .5
            c_h, d_h = (h, h + 1) if charge_first else (h + 1, h)
            room = cap - init if charge_first else init - min_e
            amount = min(rate_c, rate_d, room, hrs[d_h]['demand_kwh'])
            if amount <= .0001:
                continue
            amount = round(rng.uniform(0, amount), 4)
            if amount > 0:
                net[c_h], net[d_h] = amount, -amount
    levels = []
    e = init
    for n in net:
        e = round(e + n, 7)
        levels.append(e)
    assert abs(e - init) < 1e-5

    witness = []
    for h, row in enumerate(hrs):
        need = row['demand_kwh'] + net[h]
        assert need >= -1e-7, (category, h, need)
        used = round(min(solar[h], max(0., need)) * rng.choice([0., .3, .73, 1.]), 7)
        grid = round(need - used, 7)
        witness.append(dict(grid_kwh=grid, solar_used_kwh=used,
                            battery_energy_after_kwh=levels[h], net_battery_kwh=net[h]))

    active = []
    for kind in kinds:
        full = category == 'full_day_directives'
        if kind == 'solar_reduction':
            dv = solar_d
        elif kind == 'minimum_battery_reserve':
            wh = window_hours(rng, full)
            lo = min(levels[h] for h in wh)
            threshold = rng.choice([min_e, lo, rnd(rng, min_e, lo, 4)])
            dv = mk_directive(-1, kind, wh, minimum_energy_kwh=round(threshold, 4))
        elif kind == 'no_charge_window':
            permissible = [h for h in range(24) if net[h] <= 0]
            wh = list(range(24)) if full else sorted(set(window_hours(rng)) & set(permissible))
            if not wh:
                wh = [rng.choice(permissible)]
            dv = mk_directive(-1, kind, wh)
        elif kind == 'no_discharge_window':
            permissible = [h for h in range(24) if net[h] >= 0]
            wh = list(range(24)) if full else sorted(set(window_hours(rng)) & set(permissible))
            if not wh:
                wh = [rng.choice(permissible)]
            dv = mk_directive(-1, kind, wh)
        elif kind == 'max_grid_window':
            if full:
                wh = list(range(24))
            elif category == 'tight_grid':
                discharge_hours = [h for h, n in enumerate(net) if n < 0]
                wh = [rng.choice(discharge_hours)] if discharge_hours else [rng.randrange(24)]
            else:
                wh = window_hours(rng)
            minimum_cap = max(witness[h]['grid_kwh'] for h in wh)
            cap_grid = minimum_cap if category == 'tight_grid' or rng.random() < .6 else minimum_cap + rnd(rng, 0, 100)
            dv = mk_directive(-1, kind, wh, max_grid_kwh=round(cap_grid, 5))
        else:
            raise AssertionError(kind)
        active.append(dv)
    # Spec requires at least one note; make a no_op when no operational directives.
    if not active:
        active.append(dict(note_index=0, applies=False, directive_type='no_op',
                           structured_adjustment=None, explanation='Irrelevant synthetic note.'))
    for i, d in enumerate(active):
        d['note_index'] = i
    if rng.random() < .4:
        rng.shuffle(hrs)  # valid but out-of-order input; optimizer and replay sort hours
    inp = dict(scenario_id=case_id, operator_notes=['synthetic note'] * len(active),
               hours=hrs, battery=battery)
    return dict(id=case_id, category=category, input=inp, directives=active, witness=witness)


def independent_replay(case, response):
    """Compute constraints directly, WITHOUT calling project effective_bounds()."""
    errors = []
    req, dvs = case['input'], case['directives']
    hours = sorted(req['hours'], key=lambda x: x['hour'])
    bat = req['battery']
    plan = response.get('hourly_plan')
    if not isinstance(plan, list) or len(plan) != 24:
        return ['plan must contain 24 entries']
    if any(not isinstance(p, dict) for p in plan):
        return ['non-object plan row']
    try:
        if sorted(p['hour'] for p in plan) != list(range(24)):
            return ['hour mapping invalid']
        plan = sorted(plan, key=lambda p: p['hour'])
        eff_solar = [float(h['solar_kwh']) for h in hours]
        reserve = [float(bat['minimum_energy_kwh'])] * 24
        charge_lim = [float(bat['max_charge_kwh_per_hour'])] * 24
        discharge_lim = [float(bat['max_discharge_kwh_per_hour'])] * 24
        grid_lim = [math.inf] * 24
        for dv in dvs:
            if not dv['applies']:
                continue
            a = dv['structured_adjustment']
            for h in a['hours']:
                if dv['directive_type'] == 'solar_reduction':
                    eff_solar[h] = hours[h]['solar_kwh'] * a['factor']
                elif dv['directive_type'] == 'minimum_battery_reserve':
                    reserve[h] = max(reserve[h], a['minimum_energy_kwh'])
                elif dv['directive_type'] == 'no_charge_window':
                    charge_lim[h] = 0
                elif dv['directive_type'] == 'no_discharge_window':
                    discharge_lim[h] = 0
                elif dv['directive_type'] == 'max_grid_window':
                    grid_lim[h] = min(grid_lim[h], a['max_grid_kwh'])
        e = bat['initial_energy_kwh']
        cost = grid_total = 0.
        peak = 0.
        for h, row in enumerate(plan):
            g, s, b, after = (row[k] for k in ['grid_kwh', 'solar_used_kwh', 'battery_kwh', 'battery_energy_after_kwh'])
            if any(isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z)
                   for z in (g, s, b, after)):
                errors.append(f'h{h}: invalid numbers')
                continue
            action = row['battery_action']
            if action not in ('charge', 'discharge', 'idle'):
                errors.append(f'h{h}: unknown action {action}')
                continue
            c = b if action == 'charge' else 0.
            d = b if action == 'discharge' else 0.
            checks = [
                (g >= -TOL, 'negative grid'), (s >= -TOL, 'negative solar'),
                (b >= -TOL, 'negative battery flow'), (after >= -TOL, 'negative battery energy'),
                (action != 'idle' or abs(b) <= TOL, 'idle has nonzero battery flow'),
                (abs(e + c - d - after) <= TOL, 'battery transition'),
                (reserve[h] - TOL <= after <= bat['capacity_kwh'] + TOL, 'battery bounds/reserve'),
                (c <= charge_lim[h] + TOL, 'charging forbidden/rate'),
                (d <= discharge_lim[h] + TOL, 'discharging forbidden/rate'),
                (s <= eff_solar[h] + TOL, 'overused effective solar'),
                (g <= grid_lim[h] + TOL, 'grid import cap'),
                (abs(g + s + d - hours[h]['demand_kwh'] - c) <= TOL, 'energy balance'),
            ]
            errors.extend(f'h{h}: {why}' for ok, why in checks if not ok)
            e = after
            grid_total += g
            peak = max(peak, g)
            cost += g * hours[h]['tariff_bdt_per_kwh']
        if abs(e - bat['initial_energy_kwh']) > TOL:
            errors.append('end-of-day neutrality')
        for key, reference in [('total_grid_kwh', grid_total), ('total_cost_bdt', cost),
                               ('peak_grid_kwh', peak)]:
            got = response.get(key)
            if not isinstance(got, (int, float)) or not math.isfinite(got) or abs(got-reference) > TOL:
                errors.append(f'{key}: reported {got}, recalculated {reference}')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f'replay raised {type(exc).__name__}: {exc}')
    return errors


def independently_optimal_cost(case):
    """Second LP: 96 vars (solar, charge, discharge, energy), grid eliminated.

    In this formulation g = demand + charge - discharge - solar. The constant
    demand tariff term is restored after minimizing variable grid expense.
    """
    inp, dvs = case['input'], case['directives']
    rows = sorted(inp['hours'], key=lambda x: x['hour'])
    b = inp['battery']
    solar = [row['solar_kwh'] for row in rows]
    min_e = [b['minimum_energy_kwh']] * 24
    max_c = [b['max_charge_kwh_per_hour']] * 24
    max_d = [b['max_discharge_kwh_per_hour']] * 24
    max_g = [math.inf] * 24
    for dv in dvs:
        if not dv['applies']:
            continue
        adj = dv['structured_adjustment']
        for h in adj['hours']:
            k = dv['directive_type']
            if k == 'solar_reduction': solar[h] = rows[h]['solar_kwh'] * adj['factor']
            if k == 'minimum_battery_reserve': min_e[h] = max(min_e[h], adj['minimum_energy_kwh'])
            if k == 'no_charge_window': max_c[h] = 0
            if k == 'no_discharge_window': max_d[h] = 0
            if k == 'max_grid_window': max_g[h] = min(max_g[h], adj['max_grid_kwh'])
    # Blocks: s 0:24, c 24:48, d 48:72, e 72:96.
    obj = np.zeros(96)
    aeq, beq, aub, bub = [], [], [], []
    for h in range(24):
        tariff = rows[h]['tariff_bdt_per_kwh']
        obj[h] = -tariff
        obj[24+h] = tariff
        obj[48+h] = -tariff
        transition = np.zeros(96)
        transition[72+h] = 1
        transition[24+h] = -1
        transition[48+h] = 1
        if h:
            transition[72+h-1] = -1
        aeq.append(transition)
        beq.append(b['initial_energy_kwh'] if h == 0 else 0.)
        # Demand + charge - discharge - solar >= 0.
        no_export = np.zeros(96)
        no_export[h] = 1
        no_export[24+h] = -1
        no_export[48+h] = 1
        aub.append(no_export)
        bub.append(rows[h]['demand_kwh'])
        if math.isfinite(max_g[h]):
            cap = np.zeros(96)
            cap[h] = -1
            cap[24+h] = 1
            cap[48+h] = -1
            aub.append(cap)
            bub.append(max_g[h] - rows[h]['demand_kwh'])
    terminal = np.zeros(96)
    terminal[95] = 1
    aeq.append(terminal)
    beq.append(b['initial_energy_kwh'])
    bounds = ([(0, solar[h]) for h in range(24)]
              + [(0, max_c[h]) for h in range(24)]
              + [(0, max_d[h]) for h in range(24)]
              + [(min_e[h], b['capacity_kwh']) for h in range(24)])
    res = linprog(obj, A_ub=np.array(aub), b_ub=np.array(bub),
                  A_eq=np.array(aeq), b_eq=np.array(beq), bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(f'independent LP failed: {res.message}')
    return float(res.fun + sum(h['demand_kwh']*h['tariff_bdt_per_kwh'] for h in rows))


def run(seed=20260918, per_category=75, output=ROOT, save_cases=True):
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    results, cases, failure_cases = [], [], []
    started = time.perf_counter()
    for category in CATEGORIES:
        for j in range(per_category):
            case = build_case(rng, category, f'RANDOM-{category}-{j:03d}')
            cases.append(case)
            errors, cost_diagnostics, optimum = [], [], None
            # Check witness independently; this is why the generated test is known feasible.
            witness_plan = []
            for h, w in enumerate(case['witness']):
                flow = w['net_battery_kwh']
                witness_plan.append(dict(hour=h, grid_kwh=w['grid_kwh'],
                                         solar_used_kwh=w['solar_used_kwh'],
                                         battery_action=('charge' if flow > 0 else 'discharge' if flow < 0 else 'idle'),
                                         battery_kwh=abs(flow),
                                         battery_energy_after_kwh=w['battery_energy_after_kwh']))
            witness_resp = {'hourly_plan': witness_plan, **optimizer.totals(witness_plan, case['input']['hours'])}
            w_err = independent_replay(case, witness_resp)
            if w_err:
                errors.extend('TEST GENERATOR WITNESS: ' + x for x in w_err)
            guardrail_errors = guardrails.check_all(case['directives'], len(case['directives']), case['input']['battery']['capacity_kwh'])
            if guardrail_errors:
                errors.extend('GENERATED DIRECTIVE: ' + x for x in guardrail_errors)
            if not errors:
                try:
                    plan = optimizer.optimize(case['input']['hours'], case['input']['battery'], case['directives'])
                    resp = {'hourly_plan': plan, **optimizer.totals(plan, case['input']['hours'])}
                    errors.extend('PROJECT REPLAY: '+x for x in replay.check_plan(case['input'], case['directives'], resp))
                    errors.extend('INDEPENDENT REPLAY: '+x for x in independent_replay(case, resp))
                    optimum = independently_optimal_cost(case)
                    if resp['total_cost_bdt'] > optimum + TOL + 1e-6:
                        cost_diagnostics.append(f'ABOVE OPTIMUM BY >0.01 BDT: returned={resp["total_cost_bdt"]:.7f}, independently optimum={optimum:.7f}, excess={resp["total_cost_bdt"]-optimum:.7f}')
                    if resp['total_cost_bdt'] < optimum - TOL - 1e-6:
                        cost_diagnostics.append(f'BELOW UNROUNDED LP BY >0.01 BDT: returned={resp["total_cost_bdt"]:.7f}, independent={optimum:.7f}; permitted per-hour rounding may explain this')
                except Exception as exc:
                    errors.append(f'EXCEPTION: {type(exc).__name__}: {exc}')
            outcome = dict(id=case['id'], category=category, validity_passed=not errors,
                           cost_within_0_01_bdt=not cost_diagnostics,
                           errors=errors, cost_diagnostics=cost_diagnostics, independent_optimum_bdt=optimum)
            results.append(outcome)
            if errors or cost_diagnostics:
                failure_cases.append(case)
    elapsed = time.perf_counter() - started
    counts = {c: {
        'valid': sum(r['validity_passed'] for r in results if r['category'] == c),
        'invalid': sum(not r['validity_passed'] for r in results if r['category'] == c),
        'cost_differences': sum(not r['cost_within_0_01_bdt'] for r in results if r['category'] == c),
    } for c in CATEGORIES}
    failed = [r for r in results if not r['validity_passed']]
    cost_flags = [r for r in results if not r['cost_within_0_01_bdt']]
    coverage = {
        'three_note_cases': sum(len(c['directives']) == 3 for c in cases),
        'full_day_directive_cases': sum(any(d['applies'] and len(d['structured_adjustment']['hours']) == 24
                                           for d in c['directives']) for c in cases),
        'cases_by_directive_type': dict(collections.Counter(
            d['directive_type'] for c in cases for d in c['directives'])),
        'shuffled_hours_cases': sum([h['hour'] for h in c['input']['hours']] != list(range(24)) for c in cases),
        'zero_demand_cases': sum(all(h['demand_kwh'] == 0 for h in c['input']['hours']) for c in cases),
        'zero_rate_cases': sum(c['input']['battery']['max_charge_kwh_per_hour'] == 0
                               and c['input']['battery']['max_discharge_kwh_per_hour'] == 0 for c in cases),
    }
    report = {'seed': seed, 'per_category': per_category, 'total': len(results),
              'validity_passed': len(results)-len(failed), 'validity_failed': len(failed),
              'cost_within_0_01_bdt': len(results)-len(cost_flags),
              'cost_differences': len(cost_flags),
              'elapsed_seconds': round(elapsed, 3), 'by_category': counts, 'coverage': coverage,
              'method': 'feasible witness + project replay + independent replay + independently formulated LP cost oracle; no LLM',
              'validity_failures': failed, 'cost_precision_diagnostics': cost_flags}
    (output/'stress_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (output/'flagged_cases.json').write_text(json.dumps(failure_cases, indent=2, allow_nan=False), encoding='utf-8')
    if save_cases:
        (output/'random_solvable_cases.json').write_text(json.dumps(cases, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f'REPLAY: {report["validity_passed"]}/{report["total"]} valid; {report["validity_failed"]} invalid; {elapsed:.2f}s')
    print(f'COST ORACLE: {report["cost_within_0_01_bdt"]}/{report["total"]} within 0.01 BDT; {report["cost_differences"]} rounding/precision flags')
    for category, n in counts.items():
        print(f'  {category:21s} {n["valid"]:3d} valid / {n["invalid"]:3d} invalid / {n["cost_differences"]:3d} cost flags')
    for failure in failed[:15]:
        print(f'INVALID {failure["id"]}: {failure["errors"][:4]}')
    for flagged in cost_flags[:15]:
        print(f'COST FLAG {flagged["id"]}: {flagged["cost_diagnostics"][:3]}')
    if len(failed)>15 or len(cost_flags)>15:
        print('Additional flagged case details in stress_report.json')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--per-category', type=int, default=75)
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--no-save-cases', action='store_true')
    args = p.parse_args()
    rep = run(args.seed, args.per_category, args.output, not args.no_save_cases)
    raise SystemExit(0 if rep['validity_failed']==0 and rep['cost_differences']==0 else 1)
