"""Terminal-only secondary audit of the fixed sprint, never modifies raw records."""
import argparse
from collections import Counter
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
sys.path[:0]=[str(HERE),str(ROOT)]
from episode import read,write,sha,require,check_sources
from originx_confirmatory_20261009.analysis import analyze,_prepare,_pair_status
import evidence
from transport_audit import install
ev=install(evidence)

def normalize(path):
    c=read(path/'config.json');cs=sha(path/'config.json');m=read(c['manifest']);check_sources(c)
    require(sha(c['manifest'])==c['manifest_sha256'],'Manifest changed')
    require(read(path/'completion.json').get('all_children_drained') is True,'Campaign not terminal')
    jobs={j['request_id']:j for j in m['jobs']};rows=[];records=[];audits=[]
    for j in m['jobs']:
        n,a=ev.normalize_episode(c,cs,j,path,jobs)
        # Independently verify initial state even if a later technical/deadline
        # failure prevented terminal validation. Never infer it from a label.
        ep=path/'episodes'/j['id'];rp=ep/'result.json'
        if not n['initial_fingerprint'] and rp.exists():
            raw=read(rp)
            if raw.get('job')==j and raw.get('config_sha256')==cs and raw.get('manifest_sha256')==c['manifest_sha256']:
                try:n['initial_fingerprint'],_=ev._initial(ep,raw);a['initial_verified_independently']=True
                except Exception as error:a['initial_audit_error']=str(error)
        claim=path/'claims'/(j['id']+'.json');xp=path/'worker-exits'/(j['id']+'.json');exit=read(xp) if xp.exists() else {}
        n['execution_status']=('completed_success' if n['success'] else 'completed_failure') if n['valid'] else 'not_started' if not claim.exists() else 'administrative_deadline' if 'deadline' in (exit.get('forced_reason') or '') else 'technical_error'
        rows.append({k:n[k] for k in ('case_id','task_name','policy_id','arm','seed','horizon','episode_id','query_interval')})
        records.append(n);audits.append(a)
    return c,m,rows,records,audits,jobs

def run(output,broker_state=None):
    core=ROOT/'results/originx-sprint-xab-20261011-v1';bv=ROOT/'results/originx-sprint-bv-20261011-v1'
    c,m,rows,n,a,j=normalize(core);bc,bm,br,bn,ba,bj=normalize(bv)
    require(m['cases']==bm['cases'] and len(m['cases'])==50,'Fixed common50 mismatch')
    crows=[r for r in rows if r['policy_id']=='B'];cn=[r for r in n if r['policy_id']=='B']
    b_report=analyze(crows+br,cn+bn,N=50,bootstrap_samples=2000,bootstrap_seed=20261011)
    paired=[];verified_rescues=0;verified_regressions=0
    for case in m['cases']:
        cid=case['case_id'];cr=next(r for r in crows if r['case_id']==cid);vr=next(r for r in br if r['case_id']==cid)
        cv=next(r for r in cn if r['case_id']==cid);vv=next(r for r in bn if r['case_id']==cid)
        pc=_prepare(cr,cv,None);pv=_prepare(vr,vv,None);state,reason=_pair_status(pc,pv)
        verified=state=='matched' and vv.get('assistance',{}).get('delivered') is True
        rescue=verified and cv['success'] is False and vv['success'] is True
        regression=verified and cv['success'] is True and vv['success'] is False
        verified_rescues+=rescue;verified_regressions+=regression
        paired.append(dict(case_id=cid,status=state,reason=reason,control_success=cv['success'] if cv['valid'] else None,
            treatment_success=vv['success'] if vv['valid'] else None,verified_Astra_application=vv.get('assistance',{}).get('delivered') is True,
            strict_verified_Astra_rescue=rescue,strict_verified_Astra_regression=regression))
    cross=[]
    for case in m['cases']:
        rs=[r for r in n if r['case_id']==case['case_id']];fs=[r['initial_fingerprint'] for r in rs if r['initial_fingerprint']]
        deviation=len(fs)>1 and any(f!=fs[0] for f in fs[1:]);complete=all(r['valid'] for r in rs)
        cross.append(dict(case_id=case['case_id'],initial_state_deviation=deviation,all_three_valid=complete,
                          comparable=complete and not deviation,outcomes={r['policy_id']:r['success'] if r['valid'] else None for r in rs}))
    usage=ev.collect_usage(bv,bj,sha(bv/'config.json'),bc['manifest_sha256'],broker_state)
    report=dict(schema='originx_sprint_final_audit_v1',fixed_unique_environment_cases=50,core_planned_arms=150,BV_planned_arms=50,
        postprocessor_sources_sha256={str(HERE/n):sha(HERE/n) for n in ['final_audit.py','transport_audit.py','evidence.py']},
        raw_campaign_paths=dict(core=str(core),BV=str(bv)),
        core_authority=dict(config_sha256=sha(core/'config.json'),manifest_sha256=c['manifest_sha256']),BV_authority=dict(config_sha256=sha(bv/'config.json'),manifest_sha256=bc['manifest_sha256']),
        core_status_counts=dict(Counter(r['execution_status'] for r in n)),BV_status_counts=dict(Counter(r['execution_status'] for r in bn)),
        core_per_policy={p:dict(denominator=50,success=sum(r['policy_id']==p and r['valid'] and r['success'] is True for r in n),
            failure=sum(r['policy_id']==p and r['valid'] and r['success'] is False for r in n),unknown=sum(r['policy_id']==p and not r['valid'] for r in n)) for p in ['X','A','B']},
        cross_policy_complete_initial_matched=sum(r['comparable'] for r in cross),cross_policy_initial_deviation=sum(r['initial_state_deviation'] for r in cross),
        cross_policy_unresolved_cases=sum(not r['comparable'] and not r['initial_state_deviation'] for r in cross),
        cross_policy_action_prefix_equality_required=False,BV_pair_status_counts=dict(Counter(r['status'] for r in paired)),
        BV_strict_verified_Astra_rescues=verified_rescues,BV_strict_verified_Astra_regressions=verified_regressions,
        BV_verified_Astra_applications=sum(r['verified_Astra_application'] for r in paired),BV_fixed_denominator=50,
        BV_full_registered_pair_analysis=b_report,cli_usage={k:v for k,v in usage.items() if k!='calls'},
        cross_policy_cases=cross,BV_pairs=paired,development_case_ids=['sprint-dev-00'],development_seed=2027201100,
        historical_results_unchanged=True,official_leaderboard_claim=False,training=False,real_robot=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name,value in [('report.json',report),('normalized_records.json',n+bn),('episode_audits.json',a+ba),('cli_usage.json',usage)]:write(output/name,value,exclusive=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--broker-state',type=Path);a=p.parse_args();r=run(a.output,a.broker_state)
    print({k:v for k,v in r.items() if k not in ('cross_policy_cases','BV_pairs','BV_full_registered_pair_analysis','cli_usage')})
