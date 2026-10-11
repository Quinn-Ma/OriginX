"""Outcome-blind fresh task-balanced X/A/B cohort and its B/V extension."""
import hashlib
import random

def make_manifest(reference, development=False, bv=False):
    tasks=sorted(reference['tasks'],key=lambda t:t['task'])
    assert len(tasks)==50 and len({t['task'] for t in tasks})==50
    if development:
        tasks=sorted((t for t in tasks if t['stratum']=='atomic_seen'),key=lambda t:(t['horizon'],t['task']))[:1]
        assert len(tasks)==1
    cases=[]
    for i,t in enumerate(tasks):
        seed=(2027201100 if development else 2027101100)+i
        cases.append(dict(case_id=f"sprint-{'dev' if development else 'fresh'}-{i:02d}",task=t['task'],task_name=t['task'],
                          stratum=t['stratum'],seed=seed,env_seed=seed,policy_seed=seed,horizon=t['horizon'],task_index=i,seed_index=0))
    pairs=[('B','V')] if bv else [('X','C'),('A','C'),('B','C')]
    order=list(cases);random.Random(202610111545).shuffle(order)
    jobs=[]
    for c in order:
        for p,a in pairs:
            eid=f"{c['case_id']}--{p}--{a}"
            jobs.append(dict(c,id=eid,episode_id=eid,policy_id=p,arm=a,request_id=hashlib.sha256(('originx-sprint-v1:'+eid).encode()).hexdigest()[:32],
                             batch_group=f"{c['task_index']:02d}-00-{p}-{a}",assignment_index=len(jobs)))
    return dict(schema='originx_confirmatory_manifest_v1' if bv else 'originx_sprint_xab_manifest_v1',development=development,
                tasks=tasks,cases=cases,jobs=jobs,case_count=len(cases),arm_outcome_count=len(jobs),policy_arms={'B':['V']} if bv else {'X':['C'],'A':['C'],'B':['C']},
                seed_selection_uses_outcomes=False,no_seed_replacements=True,copied_trajectories=False,query_interval=16,
                schedule_seed=202610111545,new_cohort_shared_across_extensions=True)

def validate(m):
    jobs=m['jobs'];cases=m['cases'];cm={c['case_id']:c for c in cases}
    assert len(cm)==len(cases) and len({c['seed'] for c in cases})==len(cases)
    required={(p,a) for p,arms in m['policy_arms'].items() for a in arms}
    assert len({j['id'] for j in jobs})==len(jobs)==len(cases)*len(required)
    assert len({j['request_id'] for j in jobs})==len(jobs)
    for c in cases:
        rows=[j for j in jobs if j['case_id']==c['case_id']]
        assert {(j['policy_id'],j['arm']) for j in rows}==required
        assert all(all(j[k]==v for k,v in c.items()) for j in rows)
    assert len(cases)==(1 if m['development'] else 50)
    return True
