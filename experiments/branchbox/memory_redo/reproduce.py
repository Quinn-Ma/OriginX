"""Reproduce the learned-memory follow-up from anonymized binary outcomes.

Usage: python reproduce.py [--output reproduced.json]
Requires Python 3 and NumPy. No training software or GPU is required.
The CSV is the sole outcome input; analysis.json is checked only after calculation.
"""
from pathlib import Path
from fractions import Fraction
from math import comb
import argparse
import csv
import hashlib
import itertools
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
MODES=('baseline','memory','sreset','reset')
CONTRASTS=(('memory','sreset'),('memory','baseline'),('sreset','baseline'),('sreset','reset'),('memory','reset'))

def load_outcomes(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['task_id','init',*MODES]:
            raise ValueError('Unexpected outcome columns')
        records={}
        for row in reader:
            key=(int(row['task_id']),int(row['init']))
            if key in records: raise ValueError('Duplicate paired scene')
            values=[int(row[m]) for m in MODES]
            if any(v not in (0,1) for v in values): raise ValueError('Outcomes must be binary')
            records[key]=values
    expected=list(itertools.product(range(6),range(20,50)))
    if sorted(records)!=expected: raise ValueError('Expected six tasks and initial states20-49,180complete scenes')
    return np.asarray([records[k] for k in expected],dtype=np.int64)

def exact_mcnemar(wins,losses):
    n=wins+losses
    if n==0: return 1.
    observed=comb(n,wins)
    # Exact two-sided binomial test: sum equally or less probable outcomes.
    return float(sum((Fraction(comb(n,j),2**n) for j in range(n+1) if comb(n,j)<=observed),Fraction(0)))

def compute(matrix):
    if matrix.shape!=(180,4) or not np.isin(matrix,[0,1]).all(): raise ValueError('Unexpected outcome matrix')
    rng=np.random.default_rng(1)
    B=10000
    draws=[]
    # Generate all draws in task order; same draws apply to every contrast.
    for task in range(6):
        ids=rng.integers(0,30,size=(B,30))
        counts=np.zeros((B,30),dtype=np.int64)
        np.add.at(counts,(np.repeat(np.arange(B),30),ids.ravel()),1)
        draws.append(counts)
    effects={}
    for x,y in CONTRASTS:
        d=matrix[:,MODES.index(x)]-matrix[:,MODES.index(y)]
        boot=sum(counts @ d[t*30:(t+1)*30] for t,counts in enumerate(draws))/180*100
        wins=int((d>0).sum());losses=int((d<0).sum())
        effects[f'{x}-{y}']={
            'difference_pp':round(float(d.mean()*100),2),
            'ci95_pp':[round(float(v),2) for v in np.percentile(boot,[2.5,97.5])],
            'n':180,'wins':wins,'losses':losses,'p_mcnemar':exact_mcnemar(wins,losses)}
    order=sorted(effects,key=lambda k:effects[k]['p_mcnemar'])
    running=0.
    for rank,key in enumerate(order):
        running=max(running,min(1.,effects[key]['p_mcnemar']*(len(order)-rank)))
        effects[key]['p_holm']=running
    per_task={}
    for t in range(6):
        task=matrix[t*30:(t+1)*30]
        per_task[str(t)]={'n':30,'successes':dict(zip(MODES,map(int,task.sum(0)))),
                         'memory-sreset_pp':round(float((task[:,1]-task[:,2]).mean()*100),2),
                         'memory-baseline_pp':round(float((task[:,1]-task[:,0]).mean()*100),2)}
    return {'n_per_arm':180,'rollouts':720,'tasks':list(range(6)),'initial_state_ids':[20,49],
            'training_and_rollout_seed':1,'bootstrap_draws':B,'bootstrap_seed':1,
            'bootstrap':'paired percentile bootstrap, resampling30scenes within each of6tasks; shared indices across arms',
            'primary_contrast':'memory-sreset',
            'successes':dict(zip(MODES,map(int,matrix.sum(0)))),
            'success_rates_percent':dict(zip(MODES,[round(float(v),1) for v in matrix.mean(0)*100])),
            'contrasts':effects,'per_task':per_task}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compute(load_outcomes(ROOT/'paired_outcomes.csv'))
    reference=json.loads((ROOT/'analysis.json').read_text(encoding='utf-8'))
    if result!=reference: raise SystemExit('Reproduction does not match canonical statistics')
    provenance=json.loads((ROOT/'provenance.json').read_text(encoding='utf-8'))
    for name,digest in provenance['package_file_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('Package hash mismatch: '+name)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'verified':True,'successes':result['successes'],'contrasts':result['contrasts']},indent=2))

if __name__=='__main__': main()
