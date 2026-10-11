"""Exact fixed-denominator sprint evidence ledger; no missing outcome is failure."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
sys.path[:0]=[str(HERE),str(ROOT)]
from episode import read,sha,write,require,check_sources,active
import evidence as ev

def aggregate(config_path, destination, pilot=False):
    c=read(config_path);cs=sha(config_path);out=Path(c['output']);m=read(c['manifest']);require(sha(c['manifest'])==c['manifest_sha256'],'Manifest changed')
    check_sources(c);jobs={j['request_id']:j for j in m['jobs']};records=[];audit=[];ledger=[]
    for j in m['jobs']:
        n,a=ev.normalize_episode(c,cs,j,out,jobs);records.append(n);audit.append(a)
        exits=out/'worker-exits'/(j['id']+'.json');claim=out/'claims'/(j['id']+'.json')
        state=('completed_success' if n['success'] else 'completed_failure') if n['valid'] else 'technical_error' if claim.exists() else 'not_started'
        if not n['valid'] and exits.exists() and read(exits).get('forced_reason'):state='administrative_deadline' if 'deadline' in read(exits)['forced_reason'] else 'technical_error'
        initial=out/'episodes'/j['id']/'initial_scene.json'
        ledger.append(dict(job=j,status=state,success=n['success'] if n['valid'] else None,evidence_valid=n['valid'],reason=a.get('reason'),
                           initial_fingerprint=n['initial_fingerprint'],raw_initial_scene=read(initial) if initial.exists() else None))
    pairs=[]
    for case in m['cases']:
        rows=[r for r in records if r['case_id']==case['case_id']]
        complete=all(r['valid'] for r in rows)
        fingerprints=[r['initial_fingerprint'] for r in rows if r['initial_fingerprint']]
        mismatch=len(fingerprints)>1 and any(v!=fingerprints[0] for v in fingerprints[1:])
        pairs.append(dict(case_id=case['case_id'],complete=complete,initial_match=complete and not mismatch,initial_state_deviation=mismatch,
                          outcomes={r['policy_id']:r['success'] if r['valid'] else None for r in rows},prefix_equality_required=False))
        if mismatch:
            for row in ledger:
                if row['job']['case_id']==case['case_id'] and row['evidence_valid']:row['comparison_status']='initial_state_deviation'
    report=dict(schema='originx_sprint_xab_result_v1',config_sha256=cs,manifest_sha256=c['manifest_sha256'],unique_environment_cases=m['case_count'],
                planned_arm_outcomes=len(ledger),development=c['development'],status_counts=dict(Counter(r['status'] for r in ledger)),
                per_policy={p:dict(denominator=m['case_count'],success=sum(r['job']['policy_id']==p and r['status']=='completed_success' for r in ledger),
                    failure=sum(r['job']['policy_id']==p and r['status']=='completed_failure' for r in ledger),
                    unknown=sum(r['job']['policy_id']==p and r['status'] not in ('completed_success','completed_failure') for r in ledger)) for p in m['policy_arms']},
                complete_initial_matched_cases=sum(p['initial_match'] for p in pairs),initial_state_deviation_cases=sum(p['initial_state_deviation'] for p in pairs),
                cross_policy_action_prefix_equality_required=False,historical_scores_unchanged=True,official_leaderboard_claim=False,
                paired_cases=pairs,ledger=ledger)
    if pilot:
        require(c['development'] and len(ledger)==3 and report['complete_initial_matched_cases']==1,'Engineering native-reset X/A/B initial-state admission failed')
        report.update(passed=True,outcomes_not_used_for_selection=True)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    write(destination/'report.json',report,exclusive=True);write(destination/'normalized_records.json',records,exclusive=True);write(destination/'audit.json',audit,exclusive=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    r=aggregate(a.config,a.output,a.pilot);print(json.dumps({k:v for k,v in r.items() if k not in ('ledger','paired_cases')}))
