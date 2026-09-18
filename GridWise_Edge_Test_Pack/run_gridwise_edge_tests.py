#!/usr/bin/env python3
"""GridWise independent black-box test runner. Standard library only.

Example:
    python run_gridwise_edge_tests.py --base-url https://your-service.example --include-invalid --report results.json
This script does not call Gemini or Groq directly: run it against the deployed GridWise API.
"""
import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

TOL=0.0100001
ENUMS={'solar_reduction','minimum_battery_reserve','no_charge_window','no_discharge_window','max_grid_window','no_op'}
KEYS={'solar_reduction':{'hours','factor'},'minimum_battery_reserve':{'hours','minimum_energy_kwh'},'no_charge_window':{'hours'},'no_discharge_window':{'hours'},'max_grid_window':{'hours','max_grid_kwh'}}

def numeric(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def close(a,b):return abs(a-b)<=TOL

def request(url,method='POST',body=None,timeout=30):
    data=None if body is None else body.encode('utf-8')
    headers={'Accept':'application/json'}
    if data is not None: headers['Content-Type']='application/json'
    req=urllib.request.Request(url,data=data,headers=headers,method=method)
    start=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read().decode('utf-8',errors='replace')
            return r.status,raw,time.monotonic()-start
    except urllib.error.HTTPError as e:
        return e.code,e.read().decode('utf-8',errors='replace'),time.monotonic()-start
    except Exception as e:
        return None,str(e),time.monotonic()-start

def validate_interpretation(response,case):
    errs=[];expected=case['expected_interpretation'];found=response.get('directive_interpretation')
    if not isinstance(found,list):return ['directive_interpretation is not an array']
    if len(found)!=len(expected):errs.append(f'expected {len(expected)} interpretations, received {len(found)}')
    for i,want in enumerate(expected):
        if i>=len(found):break
        got=found[i]
        if not isinstance(got,dict):errs.append(f'note[{i}] interpretation is not an object');continue
        for key in ('note_index','applies','directive_type','structured_adjustment'):
            if key not in got:errs.append(f'note[{i}] missing {key}')
        if 'explanation' not in got or not isinstance(got.get('explanation'),str):errs.append(f'note[{i}] explanation missing/non-string')
        for key in ('note_index','applies','directive_type'):
            if got.get(key)!=want[key] or (key=='note_index' and type(got.get(key)) is not int) or (key=='applies' and type(got.get(key)) is not bool):
                errs.append(f'note[{i}] {key}: expected {want[key]!r}, got {got.get(key)!r}')
        typ=got.get('directive_type');a=got.get('structured_adjustment');w=want['structured_adjustment']
        if typ not in ENUMS:errs.append(f'note[{i}] unsupported directive {typ!r}')
        if w is None:
            if a is not None:errs.append(f'note[{i}] no_op must have null adjustment')
            continue
        if not isinstance(a,dict):errs.append(f'note[{i}] adjustment should be object');continue
        if set(a)!=KEYS[want['directive_type']]:errs.append(f'note[{i}] adjustment keys expected {sorted(KEYS[want["directive_type"]])}, got {sorted(a)}')
        hours=a.get('hours')
        if not isinstance(hours,list) or any(type(h) is not int or h<0 or h>23 for h in hours) or hours!=sorted(set(hours)):
            errs.append(f'note[{i}] hours invalid, unsorted, duplicated or out of bounds')
        if hours!=w['hours']:errs.append(f'note[{i}] hours: expected {w["hours"]}, got {hours}')
        for key,val in w.items():
            if key=='hours':continue
            gotval=a.get(key)
            if not numeric(gotval) or not close(gotval,val):errs.append(f'note[{i}] {key}: expected {val}, got {gotval}')
            if numeric(gotval):
                if key=='factor' and not 0<=gotval<=1:errs.append(f'note[{i}] solar factor outside [0,1]')
                if key=='minimum_energy_kwh' and not 0<=gotval<=case['input']['battery']['capacity_kwh']:errs.append(f'note[{i}] reserve out of battery capacity')
                if key=='max_grid_kwh' and gotval<0:errs.append(f'note[{i}] negative grid cap')
    return errs

def validate_plan(response,case):
    errs=[];inp=case['input'];battery=inp['battery'];raw=response.get('hourly_plan')
    if response.get('scenario_id')!=inp['scenario_id']:errs.append('scenario_id not echoed correctly')
    if not isinstance(response.get('plan_summary'),str):errs.append('plan_summary missing or non-string')
    if not isinstance(raw,list) or len(raw)!=24:return errs+['hourly_plan must be an array of exactly 24 entries']
    by_hour={}
    for r in raw:
        if not isinstance(r,dict) or type(r.get('hour')) is not int or not 0<=r['hour']<24:
            errs.append('hourly_plan has malformed hour entry');continue
        if r['hour'] in by_hour:errs.append(f'duplicate hour {r["hour"]}')
        by_hour[r['hour']]=r
    if set(by_hour)!=set(range(24)):return errs+['hourly_plan does not contain all unique hours 0..23']
    effective_solar=[h['solar_kwh'] for h in inp['hours']]
    floor=[battery['minimum_energy_kwh']]*24
    max_grid=[None]*24;no_charge=set();no_discharge=set()
    for di in case['expected_interpretation']:
        t=di['directive_type'];a=di['structured_adjustment']
        if t=='solar_reduction':
            for h in a['hours']:effective_solar[h]*=a['factor']
        elif t=='minimum_battery_reserve':
            for h in a['hours']:floor[h]=max(floor[h],a['minimum_energy_kwh'])
        elif t=='max_grid_window':
            for h in a['hours']:max_grid[h]=a['max_grid_kwh'] if max_grid[h] is None else min(max_grid[h],a['max_grid_kwh'])
        elif t=='no_charge_window':no_charge.update(a['hours'])
        elif t=='no_discharge_window':no_discharge.update(a['hours'])
    before=battery['initial_energy_kwh'];sum_grid=sum_cost=0;peak=0
    for h in range(24):
        r=by_hour[h]
        names=('grid_kwh','solar_used_kwh','battery_kwh','battery_energy_after_kwh')
        if any(not numeric(r.get(n)) for n in names):errs.append(f'hour {h}: missing/nonfinite numeric field');continue
        g,s,b,e=[r[n] for n in names];act=r.get('battery_action')
        if act not in ('charge','discharge','idle'):errs.append(f'hour {h}: invalid battery action');continue
        if any(v<-TOL for v in (g,s,b)):errs.append(f'hour {h}: negative energy')
        if s>effective_solar[h]+TOL:errs.append(f'hour {h}: solar {s} > effective solar {effective_solar[h]}')
        if e<floor[h]-TOL or e>battery['capacity_kwh']+TOL:errs.append(f'hour {h}: battery {e} outside bounds [{floor[h]}, {battery["capacity_kwh"]}]')
        if (act=='charge' and b>battery['max_charge_kwh_per_hour']+TOL) or (act=='discharge' and b>battery['max_discharge_kwh_per_hour']+TOL):errs.append(f'hour {h}: rate limit violated')
        if act=='idle' and abs(b)>TOL:errs.append(f'hour {h}: idle with nonzero battery_kwh')
        if h in no_charge and act=='charge' and b>TOL:errs.append(f'hour {h}: forbidden charging')
        if h in no_discharge and act=='discharge' and b>TOL:errs.append(f'hour {h}: forbidden discharging')
        if max_grid[h] is not None and g>max_grid[h]+TOL:errs.append(f'hour {h}: grid cap exceeded ({g}>{max_grid[h]})')
        delta=(b if act=='charge' else (-b if act=='discharge' else 0))
        if not close(e,before+delta):errs.append(f'hour {h}: battery state transition invalid ({before}+{delta}!={e})')
        demand=inp['hours'][h]['demand_kwh']
        if not close(g+s+(b if act=='discharge' else 0),demand+(b if act=='charge' else 0)):
            errs.append(f'hour {h}: energy balance violated')
        before=e;sum_grid+=g;sum_cost+=g*inp['hours'][h]['tariff_bdt_per_kwh'];peak=max(peak,g)
    if not close(before,battery['initial_energy_kwh']):errs.append(f'end-of-day neutrality violated (end={before}, start={battery["initial_energy_kwh"]})')
    for name,want in [('total_grid_kwh',sum_grid),('total_cost_bdt',sum_cost),('peak_grid_kwh',peak)]:
        got=response.get(name)
        if not numeric(got) or not close(got,want):errs.append(f'{name} mismatch: expected recalculated {want:.6f}, got {got!r}')
    return errs

def percentile95(xs):
    if not xs:return None
    xs=sorted(xs);return xs[math.ceil(.95*len(xs))-1]

def run(args):
    pack=json.load(open(args.pack,encoding='utf-8'));base_url=args.base_url.rstrip('/')
    valid=[c for c in pack['valid_cases'] if (not args.case or c['id'] in args.case) and (not args.tag or args.tag in c['tags'])]
    invalid=[c for c in pack['invalid_cases'] if not args.case or c['id'] in args.case] if args.include_invalid else []
    rows=[];status,body,elapsed=request(base_url+'/health',method='GET',timeout=args.timeout)
    try:health=(status==200 and json.loads(body).get('status')=='ok')
    except (ValueError,AttributeError):health=False
    print(f'HEALTH: {"PASS" if health else "FAIL"} status={status} time={elapsed:.3f}s')
    rows.append({'id':'HEALTH','name':'GET /health','passed':health,'status':status,'latency_sec':round(elapsed,5),'errors':[] if health else ['Expected HTTP 200 and JSON {"status":"ok"}']})
    for case in valid:
        for rep in range(args.repeat):
            status,body,elapsed=request(base_url+'/optimize-energy',body=json.dumps(case['input'],allow_nan=False),timeout=args.timeout)
            errs=[];interp_errs=[];plan_errs=[];cost_errs=[]
            if status!=200:errs.append(f'Expected 200, received {status}; response={body[:250]!r}')
            else:
                try:payload=json.loads(body)
                except ValueError:errs.append('Response not valid JSON')
                else:
                    if not isinstance(payload,dict):errs.append('Response JSON is not an object')
                    else:
                        interp_errs=validate_interpretation(payload,case)
                        plan_errs=validate_plan(payload,case)
                        got=payload.get('total_cost_bdt');opt=case['reference_optimal_cost_bdt']
                        if not args.skip_optimality and not plan_errs and (not numeric(got) or got>opt+TOL or got<opt-TOL):
                            cost_errs.append(f'Cost {got!r} not within 0.01 of independently optimized reference {opt}')
            errs+=interp_errs+plan_errs+cost_errs
            row={'id':case['id'],'name':case['name'],'repeat':rep+1,'tags':case['tags'],'passed':not errs,'status':status,'latency_sec':round(elapsed,5),'interpretation_ok':not interp_errs and status==200,'plan_ok':not plan_errs and status==200,'optimality_ok':not cost_errs and status==200,'errors':errs}
            rows.append(row);print(f'{"PASS" if row["passed"] else "FAIL"} {case["id"]} #{rep+1}: {case["name"]} ({elapsed:.2f}s)'+(f' | {errs[0]}' if errs else ''))
    for case in invalid:
        payload=case['raw_body'] if case['raw_body'] is not None else json.dumps(case['input'],allow_nan=False)
        status,body,elapsed=request(base_url+'/optimize-energy',body=payload,timeout=args.timeout)
        errs=[] if status in case['acceptable_http_statuses'] else [f'Expected 400 or 422, got {status}, response={body[:250]!r}']
        row={'id':case['id'],'name':case['name'],'passed':not errs,'status':status,'latency_sec':round(elapsed,5),'errors':errs}
        rows.append(row);print(f'{"PASS" if not errs else "FAIL"} {case["id"]}: {case["name"]} ({elapsed:.2f}s)'+(f' | {errs[0]}' if errs else ''))
    vr=[r for r in rows if r['id'].startswith('EDGE-')];ir=[r for r in rows if r['id'].startswith('INVALID-')]
    latency=[r['latency_sec'] for r in vr]
    overview={'health_ok':health,'valid_passed':sum(r['passed'] for r in vr),'valid_total':len(vr),'invalid_passed':sum(r['passed'] for r in ir),'invalid_total':len(ir),'directive_interpretation_passed':sum(r['interpretation_ok'] for r in vr),'plan_passed':sum(r['plan_ok'] for r in vr),'optimality_passed':sum(r['optimality_ok'] for r in vr),'p95_valid_latency_sec':percentile95(latency),'valid_over_30_sec':sum(x>30 for x in latency),'valid_over_5_sec':sum(x>5 for x in latency)}
    report={'summary':overview,'results':rows,'note':'Cases are original synthetic edge cases, not organizer hidden cases. Cost reference is a SciPy HiGHS LP computed offline.'}
    if args.report:
        Path(args.report).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f'REPORT: {args.report}')
    print('SUMMARY:',json.dumps(overview,indent=2))
    return 0 if health and all(r['passed'] for r in rows) else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True,help='Public GridWise API root, NOT a raw LLM provider URL')
    p.add_argument('--pack',default=str(Path(__file__).with_name('gridwise_edge_cases.json')))
    p.add_argument('--case',action='append',help='Select specific ID, e.g., EDGE-25; can repeat')
    p.add_argument('--tag',help='Filter valid scenarios by one tag, e.g. percentage, multi_note, zero')
    p.add_argument('--repeat',type=int,default=1,help='Repeat every selected valid request N times (uses API quota)')
    p.add_argument('--timeout',type=float,default=30,help='Seconds allowed for each request')
    p.add_argument('--skip-optimality',action='store_true',help='Skip comparing cost to the independent LP optimum')
    p.add_argument('--include-invalid',action='store_true',help='Also test 13 malformed/invalid requests')
    p.add_argument('--report',help='Write JSON results here')
    a=p.parse_args()
    if a.repeat<1 or a.timeout<=0:p.error('--repeat must be >=1 and timeout >0')
    sys.exit(run(a))
