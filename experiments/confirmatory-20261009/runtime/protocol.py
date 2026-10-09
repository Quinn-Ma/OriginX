"""Prospective assignment generation. Does not read policy outcomes."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

POLICY_ARMS = {'B': ['C','R','G','L','V'], 'base': ['C','V']}
SEED_START = 1986100900
DEVELOPMENT_START = 1986200900
SCHEDULE_SEED = 202610091838

def make_manifest(reference, development=False):
    tasks = sorted(reference['tasks'], key=lambda t:t['task'])
    assert len(tasks)==50 and len({t['task'] for t in tasks})==50
    if development:
        # Chosen by alphabetical order among shortest horizons, never outcomes.
        tasks = sorted(tasks, key=lambda t:(t['horizon'],t['task']))[:2]
    start = DEVELOPMENT_START if development else SEED_START
    count = 1 if development else 50
    cases=[]; blocks=[]
    for ti,t in enumerate(tasks):
        for si in range(count):
            seed=start+ti*50+si
            cases.append(dict(case_id=f"{'dev' if development else 'fresh'}-{ti:02d}-{si:02d}",task=t['task'],task_name=t['task'],
                              stratum=t['stratum'],seed=seed,env_seed=seed,policy_seed=seed,horizon=t['horizon'],task_index=ti,seed_index=si))
    rng=random.Random(SCHEDULE_SEED)
    # Balanced fixed task/group blocks; arms interleaved, without outcome access.
    for ti,t in enumerate(tasks):
        for group in range((count+7)//8):
            block=[c for c in cases if c['task_index']==ti and c['seed_index']//8==group]
            policy_arms=[(p,a) for p,arms in POLICY_ARMS.items() for a in arms]
            rng.shuffle(policy_arms)
            jobs=[]
            for p,a in policy_arms:
                for c in block:
                    eid=f"{c['case_id']}--{p}--{a}"
                    opaque=hashlib.sha256(('originx-confirmatory-v1:'+eid).encode()).hexdigest()[:32]
                    jobs.append(dict(c,id=eid,episode_id=eid,policy_id=p,arm=a,request_id=opaque,
                                     batch_group=f'{ti:02d}-{group:02d}-{p}-{a}'))
            blocks.append(jobs)
    rng.shuffle(blocks)
    jobs=[j for block in blocks for j in block]
    for i,j in enumerate(jobs):j['assignment_index']=i
    manifest=dict(schema='originx_confirmatory_manifest_v1',development=development,tasks=tasks,cases=cases,jobs=jobs,
                  policy_arms=POLICY_ARMS,case_count=len(cases),arm_outcome_count=len(jobs),query_interval=16,
                  seed_start=start,schedule_seed=SCHEDULE_SEED,seed_selection_uses_outcomes=False,
                  copied_trajectories=False,no_seed_replacements=True)
    validate_manifest(manifest)
    return manifest

def validate_manifest(m):
    assert m['schema']=='originx_confirmatory_manifest_v1'
    jobs=m['jobs'];cases=m['cases']
    assert len({j['id'] for j in jobs})==len(jobs)==7*len(cases)
    assert len({j['request_id'] for j in jobs})==len(jobs)
    assert len({c['seed'] for c in cases})==len(cases)
    case_map={c['case_id']:c for c in cases}
    assert len(case_map)==len(cases)
    for j in jobs:
        assert all(j[k]==v for k,v in case_map[j['case_id']].items())
        assert j['arm'] in POLICY_ARMS[j['policy_id']]
    assert all(v==7 for v in Counter(j['case_id'] for j in jobs).values())
    required={(p,a) for p,arms in POLICY_ARMS.items() for a in arms}
    combinations={cid:set() for cid in case_map}
    for j in jobs:combinations[j['case_id']].add((j['policy_id'],j['arm']))
    assert all(v==required for v in combinations.values())
    if not m['development']:
        assert len(cases)==2500 and len(jobs)==17500
        assert all(v==50 for v in Counter(c['task'] for c in cases).values())

def seed_audit(root, manifest):
    """Read only explicit historical manifests/plans; never consume result labels."""
    new={c['seed'] for c in manifest['cases']}; checked=[]; hits=[]; unreadable=[]; oversized=[]
    skip={'envs','cache','code','training','.git','originx_confirmatory_20261009'}
    def seeds(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('seed','env_seed','policy_seed') and type(v) is int:yield v
                elif k in ('seeds','env_seeds','policy_seeds') and isinstance(v,list):
                    yield from (n for n in v if type(n) is int)
                    yield from seeds(v)
                elif k in ('seed_range','env_seed_range','policy_seed_range') and isinstance(v,list) and len(v)==2 and all(type(n) is int for n in v):
                    yield from (n for n in new if v[0]<=n<=v[1])
                elif isinstance(v,(list,dict)):yield from seeds(v)
        elif isinstance(value,list):
            for v in value:yield from seeds(v)
    import os
    for base,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in skip and d not in ('episodes','requests','logs','batches')]
        for name in files:
            if not name.endswith('.json') or not any(x in name.lower() for x in ('manifest','protocol','plan','seed')):continue
            p=Path(base)/name
            if p.is_symlink():continue
            if p.stat().st_size>200_000_000:
                oversized.append(str(p));continue
            try: raw=p.read_bytes(); value=json.loads(raw)
            except (ValueError,OSError) as e:
                unreadable.append(dict(path=str(p),error=str(e)));continue
            overlap=sorted(new.intersection(seeds(value)))
            checked.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
            if overlap:hits.append(dict(path=str(p),overlap=overlap))
    if hits:raise ValueError('Fresh seeds overlap historical plans: '+str(hits))
    if unreadable or oversized:raise ValueError('Historical seed audit needs review: '+str(dict(unreadable=unreadable,oversized=oversized)))
    assert checked,'No historical manifests inspected'
    return dict(passed=True,method='explicit seed/env_seed/policy_seed fields in owned project manifest/protocol/plan/seed JSON; no outcome-based selection',checked=checked,new_seed_count=len(new))
