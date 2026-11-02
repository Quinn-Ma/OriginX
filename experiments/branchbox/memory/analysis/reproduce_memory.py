#!/usr/bin/env python3
"""Reproduce the complete fixed20-scene memory comparison using NumPy only.

Input is the final audited paired_outcomes.csv, not partial rollout logs.
No statistic is computed until all20 unique, valid three-arm rows are present.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

TASKS=(0,3,4,5)
INITIAL_STATES=(40,41,42,43,44)
MODES=('baseline','memory','reset')
COLUMNS=('task','init',*MODES)
BOOTSTRAP_REPLICATES=100_000
BOOTSTRAP_SEED=1

def load_pairs(path:Path)->list[dict]:
    """Require exactly the fixed20 scenes; numeric0/1 are not Boolean inputs."""
    expected=set(itertools.product(TASKS,INITIAL_STATES))
    paired={}
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames)!=len(COLUMNS) or set(reader.fieldnames)!=set(COLUMNS):
            raise ValueError('CSV must have exactly these distinct columns: '+','.join(COLUMNS))
        for number,row in enumerate(reader,start=2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed CSV row at line {number}')
            if row['task'].strip() not in {str(t) for t in TASKS} or row['init'].strip() not in {str(i) for i in INITIAL_STATES}:
                raise ValueError(f'Unexpected task or initial state at line {number}')
            key=int(row['task']),int(row['init'])
            if key in paired:raise ValueError(f'Duplicate scene at line {number}: {key}')
            value={'task':key[0],'init':key[1]}
            for mode in MODES:
                literal=row[mode].strip().lower()
                if literal not in ('true','false'):
                    raise ValueError(f'{mode} must be a Boolean true/false at line {number}')
                value[mode]=literal=='true'
            paired[key]=value
    if set(paired)!=expected:
        raise ValueError(f'Complete20-scene panel required; found{len(paired)}. Missing: {sorted(expected-set(paired))}')
    return [paired[key] for key in itertools.product(TASKS,INITIAL_STATES)]

def exact_mcnemar(rescues:int,regressions:int)->float:
    """Two-sided exact binomial McNemar test, conditional on discordant pairs."""
    discordant=rescues+regressions
    if discordant==0:return 1.0
    lower=sum(math.comb(discordant,k) for k in range(min(rescues,regressions)+1))
    return min(1.0,2.0*lower/(2**discordant))

def summarize(pairs:list[dict])->dict:
    """Four fixed tasks; resample their five paired scenes independently."""
    expected=list(itertools.product(TASKS,INITIAL_STATES))
    if len(pairs)!=20 or [(r.get('task'),r.get('init')) for r in pairs]!=expected:
        raise ValueError('Validated sorted complete20-scene panel required')
    if any(type(row.get(mode)) is not bool for row in pairs for mode in MODES):
        raise ValueError('All outcomes must be Python Boolean values')
    successes={m:sum(row[m] for row in pairs) for m in MODES}
    # Generate one set of scene draws and share it across both contrasts/arms.
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    draws=[rng.integers(0,5,size=(BOOTSTRAP_REPLICATES,5)) for _ in TASKS]
    contrasts={}
    for control in ('baseline','reset'):
        differences=np.asarray([int(r['memory'])-int(r[control]) for r in pairs],dtype=np.int8).reshape(4,5)
        replicate=np.stack([differences[t][draws[t]].mean(axis=1) for t in range(4)]).mean(axis=0)*100.0
        rescue=[{'task':r['task'],'init':r['init']} for r in pairs if r['memory'] and not r[control]]
        regression=[{'task':r['task'],'init':r['init']} for r in pairs if r[control] and not r['memory']]
        contrasts['memory_vs_'+control]={
            'rescues':len(rescue),'regressions':len(regression),
            'difference_pp':float(differences.mean()*100.0),
            'ci95_unadjusted_pp':np.quantile(replicate,[.025,.975]).tolist(),
            'relative_success_change_percent':None if successes[control]==0 else 100.0*(successes['memory']/successes[control]-1.0),
            'mcnemar_exact_two_sided_p':exact_mcnemar(len(rescue),len(regression)),
            'rescue_scenes':rescue,'regression_scenes':regression,
        }
    # Holm step-down adjustment across the two specified contrasts only.
    ordered=sorted(contrasts,key=lambda name:contrasts[name]['mcnemar_exact_two_sided_p'])
    previous=0.0
    for position,name in enumerate(ordered):
        previous=max(previous,min(1.0,(2-position)*contrasts[name]['mcnemar_exact_two_sided_p']))
        contrasts[name]['holm_adjusted_p']=previous
    per_task=[]
    for task in TASKS:
        selected=[r for r in pairs if r['task']==task]
        counts={m:sum(r[m] for r in selected) for m in MODES}
        per_task.append(dict(task=task,n_scenes=5,**{m+'_successes':counts[m] for m in MODES},
            **{m+'_success_percent':20.0*counts[m] for m in MODES},
            memory_vs_baseline_pp=20.0*(counts['memory']-counts['baseline']),
            memory_vs_reset_pp=20.0*(counts['memory']-counts['reset'])))
    return dict(status='complete',successes=successes,success_rates_percent={m:5.0*n for m,n in successes.items()},
        n_per_condition=20,paired_scenes=20,total_rollouts=60,contrasts=contrasts,task_table=per_task,
        paired_outcomes=pairs,methodology=dict(bootstrap_replicates=BOOTSTRAP_REPLICATES,bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_unit='five paired initial scenes resampled within each of four fixed tasks',
            interval='unadjusted percentile95% paired difference interval',multiplicity='Holm across two exact McNemar p-values'),
        interpretation=[
            'The panel was explored in earlier pilots; this is a repeated development-panel comparison, not unseen confirmation.',
            'One training seed; intervals are conditional on the trained weights and fixed tasks.',
            'Memory versus existing baseline combines additional full fine-tuning and memory; no matched no-memory additional-training control.',
            'Memory versus same-weight reset tests combined historical associative state and first-frame anchor use; it does not isolate either component.',
            'This CSV-only reproduction does not independently verify simulator trajectories, checkpoint lineage, or training correctness.'
        ])

def compare_reference(result:dict,reference:dict)->None:
    """Require exact agreement with the final canonical JSON statistics."""
    if reference.get('status')!='complete' or reference.get('independent_recount_matches') is not True:
        raise ValueError('Reference must be the completed, independently audited canonical analysis')
    for key in ('successes','success_rates_percent','n_per_condition','paired_scenes','total_rollouts','task_table','paired_outcomes'):
        if result[key]!=reference.get(key):raise ValueError('Canonical mismatch: '+key)
    keys=('rescues','regressions','difference_pp','ci95_unadjusted_pp','relative_success_change_percent',
          'mcnemar_exact_two_sided_p','holm_adjusted_p','rescue_scenes','regression_scenes')
    for name,contrast in result['contrasts'].items():
        for key in keys:
            if contrast[key]!=reference['contrasts'][name][key]:raise ValueError('Canonical mismatch: '+name+'.'+key)

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).with_name('paired_outcomes.csv'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('memory_reproduced.json'))
    parser.add_argument('--reference',type=Path,help='Optional final canonical analysis.json for exact comparison')
    args=parser.parse_args()
    try:
        pairs=load_pairs(args.input);result=summarize(pairs)
        if args.reference is not None:
            compare_reference(result,json.loads(args.reference.read_text(encoding='utf-8')))
            result['canonical_exact_match']=True
        result['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
        result['numpy_version']=np.__version__
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except (OSError,ValueError,KeyError,TypeError) as error:
        print('No result written: '+str(error),file=sys.stderr);return 2
    print('Validated20 paired scenes (60 policy outcomes).')
    print(json.dumps({'successes':result['successes'],'contrasts':result['contrasts']},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
