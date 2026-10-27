"""Pure offline prospective point/regions audit; never runs models/simulation.

One complete streaming journal pass per case. Missing/partial/invalid execution
does not become a task failure. The registered 600-action task timeout is a
known failure, whereas a wall-clock/infrastructure timeout is UNKNOWN.
All world-coordinate measurements remain post-hoc, never control inputs.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[2]
PURPOSE='regions_point_unseen_seed_validation'
ARMS=('point','regions')
GATE=ROOT/'work/id60-targeted-parser/analyze_targeted.py'
spec=importlib.util.spec_from_file_location('regions_validation_complete_gate',GATE)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [0. if k==0 else max(0.,mid-half),1. if k==n else min(1.,mid+half)]

def binom_cdf(k,n,p):return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))

def clopper_pearson(k,n,alpha=.05):
    if not n:return None
    def invert(k,target):
        lo,hi=0.,1.
        for _ in range(90):
            mid=(lo+hi)/2
            if binom_cdf(k,n,mid)>target:lo=mid
            else:hi=mid
        return (lo+hi)/2
    return [0. if k==0 else invert(k-1,1-alpha/2),
            1. if k==n else invert(k,alpha/2)]

def exact_mcnemar(wins,losses):
    discord= wins+losses
    return min(1.,2*sum(math.comb(discord,i) for i in range(min(wins,losses)+1))/2**discord) if discord else 1.

def paired_effect_ci(wins,losses,n):
    if not n:return None
    # Each discordance probability receives a 97.5% exact marginal interval.
    # Bonferroni gives >=95% simultaneous coverage, hence >=95% for p_win-p_loss.
    win=clopper_pearson(wins,n,.025);loss=clopper_pearson(losses,n,.025)
    return [max(-1.,win[0]-loss[1]),min(1.,win[1]-loss[0])]

def exact_design_power(n,p_win,p_loss,alpha=.05):
    if min(p_win,p_loss)<0 or p_win+p_loss>1:raise ValueError('Invalid pair probabilities')
    total=0.
    for w in range(n+1):
        for l in range(n-w+1):
            if exact_mcnemar(w,l)>alpha:continue
            t=n-w-l
            total+=math.comb(n,w)*math.comb(n-w,l)*p_win**w*p_loss**l*(1-p_win-p_loss)**t
    return total

def distribution(values):
    return {'n':len(values),'median_cm':statistics.median(values)*100 if values else None,
            'mean_cm':statistics.mean(values)*100 if values else None,
            'minimum_cm':min(values)*100 if values else None,'maximum_cm':max(values)*100 if values else None}

def manifest_check(manifest,root):
    if manifest.get('purpose')!=PURPOSE:raise ValueError('Unexpected prospective purpose')
    if manifest.get('manifest_sha256')!=canonical({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Manifest canonical hash mismatch')
    seeds=manifest.get('seeds');cases=manifest.get('cases')
    if not isinstance(seeds,list) or len(seeds)!=12 or len(set(seeds))!=12 or not isinstance(cases,list) or len(cases)!=24:
        raise ValueError('Exactly 12 preregistered pairs/24 cases required')
    if len({c['case_id'] for c in cases})!=24:raise ValueError('Duplicate case ID')
    audit_info=manifest.get('seed_audit') or {}
    audit_path=root/audit_info.get('path','__missing_seed_audit__')
    if not audit_path.is_file() or sha(audit_path)!=audit_info.get('sha256'):
        raise ValueError('Seed audit file hash mismatch')
    audit=load(audit_path)
    if audit.get('status')!='PASS' or audit.get('ordered_seeds')!=seeds or audit.get('all_selected_absent') is not True:
        raise ValueError('Seed exclusion/matrix mismatch')
    for index,seed in enumerate(seeds):
        group=[c for c in cases if c.get('seed')==seed]
        if len(group)!=2 or {c.get('comparison_arm') for c in group}!=set(ARMS):raise ValueError('Missing pair arm')
        if {c.get('pair_position') for c in group}!={0,1} or {c.get('pair_index') for c in group}!={index}:
            raise ValueError('Invalid counterbalanced pair indices')
        expected_order=ARMS if index%2==0 else tuple(reversed(ARMS))
        if tuple(c['comparison_arm'] for c in sorted(group,key=lambda c:c['pair_position']))!=expected_order:
            raise ValueError('Counterbalanced order changed')
        if any(c.get('case_id')!=canonical({k:v for k,v in c.items() if k!='case_id'}) for c in group):
            raise ValueError('Case identity hash mismatch')
        if any(c.get('max_steps')!=600 or c.get('task')!='CoffeeSetupMug' or c.get('benchmark')!='robocasa'
            or c.get('model')!='xr1' or c.get('arm')!='full' for c in group):raise ValueError('Unexpected task/model/horizon')
    files=manifest.get('source_files')
    if not isinstance(files,dict) or not files:raise ValueError('Missing registered source hashes')
    parent=manifest.get('parent') or {};parent_path=root/parent.get('path','__missing_parent__')
    if not parent_path.is_file() or sha(parent_path)!=parent.get('sha256'):raise ValueError('Parent lineage mismatch')
    frozen=load(parent_path)
    if frozen.get('manifest_sha256')!='8d364a9d47feecbb059f8ab81fd96816bc05d12e6fea6966c23d74702fb71552':
        raise ValueError('Unexpected development parent')
    if any(files.get(name)!=digest for name,digest in frozen['source_files'].items()):raise ValueError('Frozen source omitted/changed')
    if manifest.get('checkpoint_sha256')!=frozen.get('checkpoint_sha256'):raise ValueError('Checkpoint identity changed')
    mismatches=[name for name,want in files.items() if not (root/name).is_file() or sha(root/name)!=want]
    return mismatches

def scan_journal(path):
    scan={'initial_reset_count':0,'initial_fingerprint':None,'initial_raw_rgb_sha256':None,
        'initial_proprio_sha256':None,'executed_steps_observed':0,'events':[],'terminal_observed':False,
        'records_scanned':0,'journal_present':path.is_file(),'scan_scope':'entire_journal',
        'record_counts':{},'scan_issues':[],'observation_steps':[],'action_steps':[],
        'action_next_steps':[],'proposal_steps':[],'prepare_steps':[],'terminals':[]}
    extra={'initial_comparisons':[],'initial_plan_comparisons':[],'first_observation':None,
        'initial_geometry':None,'final_contact':None,'final_geometry':None,
        'contact_steps':[],'geometry_steps':[],'first_action_sequence':None,
        'scorer_true_steps':[],'action_contract_issues':[],'diagnostic_scope_issues':[],'renderer_admission':None}
    if not path.is_file():return scan,extra
    counts=Counter();digest=hashlib.sha256()
    with path.open('rb') as f:
        for lineno,line in enumerate(f,1):
            digest.update(line)
            try:r=json.loads(line)
            except ValueError:scan['scan_issues'].append('invalid_json_line:'+str(lineno));continue
            if not isinstance(r,dict) or not isinstance(r.get('payload'),dict):
                scan['scan_issues'].append('invalid_record_shape:'+str(lineno));continue
            sequence=r.get('sequence');step=r.get('step');kind=r.get('kind');p=r['payload']
            if sequence!=scan['records_scanned']:scan['scan_issues'].append('nonconsecutive_sequence:'+str(lineno))
            scan['records_scanned']+=1;counts[kind]+=1
            if kind=='event':
                event=p.get('kind')
                if event=='initial_reset':
                    scan['initial_reset_count']+=1;scan['initial_fingerprint']=p.get('initial_fingerprint')
                elif event=='prepare' and isinstance(p.get('receipt'),dict):
                    scan['events'].append(p['receipt']);scan['prepare_steps'].append(step)
                elif event=='action_executed':
                    scan['action_steps'].append(step);scan['action_next_steps'].append(p.get('next_step'))
                    if type(p.get('next_step')) is int:scan['executed_steps_observed']=max(scan['executed_steps_observed'],p['next_step'])
                    if extra['first_action_sequence'] is None:extra['first_action_sequence']=sequence
                    if p.get('scorer_success') is True:extra['scorer_true_steps'].append(p.get('next_step'))
                    if p.get('action_modified') is not False or type(p.get('scorer_success')) is not bool:
                        extra['action_contract_issues'].append(step)
                elif event=='terminal':
                    scan['terminal_observed']=True;raw=p.get('raw_observation') or {}
                    scan['terminals'].append({'step':step,'raw_step':raw.get('step') if isinstance(raw,dict) else None,
                        'success_observed':p.get('success_observed'),'environment_done':p.get('environment_done')})
                elif event in ('initial_comparison','initial_plan_comparison'):
                    key='initial_comparisons' if event=='initial_comparison' else 'initial_plan_comparisons'
                    extra[key].append({'sequence':sequence,'step':step,'payload':p})
                elif event=='shared_hierarchical_geometry' and step==0:
                    extra['initial_geometry']={k:p.get(k) for k in ('entities','region_children','object_anchors','audit')}
                elif event=='renderer_admission':extra['renderer_admission']=p.get('identity')
                elif event=='contact_diagnostic':
                    extra['contact_steps'].append(step);extra['final_contact']={'step':step,'payload':p}
                    if p.get('logging_only') is not True or p.get('used_by_policy_or_planner') is not False:extra['diagnostic_scope_issues'].append(step)
                elif event=='placement_geometry_diagnostic':
                    extra['geometry_steps'].append(step);extra['final_geometry']={'step':step,'payload':p}
                    if p.get('logging_only') is not True or p.get('used_by_policy_or_planner') is not False:extra['diagnostic_scope_issues'].append(step)
            elif kind=='observation':
                scan['observation_steps'].append(step)
                if step==0:
                    extra['first_observation']=p;raw=p.get('raw') or {}
                    for name,key in [('rgb','initial_raw_rgb_sha256'),('proprio','initial_proprio_sha256')]:
                        v=raw.get(name);scan[key]=v.get('sha256') if isinstance(v,dict) else None
            elif kind=='action_chunk':scan['proposal_steps'].append(step)
    scan['record_counts']=dict(counts);extra['journal_sha256']=digest.hexdigest()
    return scan,extra

def initial_checks(result,scan,extra):
    issues=[];sig=result.get('initial_signature');obs=extra['first_observation'];initial=extra['initial_comparisons'];plan=extra['initial_plan_comparisons']
    if not isinstance(sig,dict) or not isinstance(obs,dict):return ['initial_signature_or_observation_missing']
    raw=obs.get('raw') or {};rgb=raw.get('rgb') or {};prop=raw.get('proprio') or {}
    actual={'initial_fingerprint':scan['initial_fingerprint'],'rgb':{k:rgb.get(k) for k in ('sha256','dtype','shape')},
        'proprio':{k:prop.get(k) for k in ('sha256','dtype','shape')},'instruction':raw.get('instruction'),
        'camera_names':raw.get('camera_names'),'step':raw.get('step'),
        'candidates_visible_sha256':canonical(obs.get('candidates_visible')),
        'candidates_all_sha256':canonical(obs.get('candidates_all')),
        'shared_augmentation_sha256':canonical(extra['initial_geometry'])}
    if sig!=actual:issues.append('initial_signature_differs_from_independent_journal_reconstruction')
    if len(initial)!=1 or initial[0]['payload'].get('actual')!=sig:issues.append('initial_comparison_missing_or_differs')
    if len(plan)!=1 or plan[0]['payload'].get('natural_plan')!=result.get('initial_plan') or plan[0]['payload'].get('injected') is not False:
        issues.append('natural_plan_journal_missing_differs_or_injected')
    for rows in (initial,plan):
        if any(r['step']!=0 or extra['first_action_sequence'] is None or r['sequence']>=extra['first_action_sequence'] for r in rows):
            issues.append('initial_admission_not_before_actions')
    return issues

def final_xy(result,extra):
    """Pure recorded geometry; does not query a scorer or select its target."""
    plan=result.get('initial_plan');steps=result.get('steps')
    if extra['diagnostic_scope_issues']:return {'status':'UNKNOWN','reason':'diagnostic_scope_not_offline'}
    if not isinstance(plan,list):return {'status':'UNKNOWN','reason':'natural_plan_missing'}
    places=[(i,s) for i,s in enumerate(plan) if s.get('operation')=='place']
    if len(places)!=1 or places[0][0]==0:return {'status':'UNKNOWN','reason':'ambiguous_place_subject'}
    i,stage=places[0];previous=plan[i-1]
    if previous.get('operation')!='grasp':return {'status':'UNKNOWN','reason':'no_preceding_grasp'}
    if extra['contact_steps']!=list(range(1,steps+1)) or extra['geometry_steps']!=list(range(1,steps+1)):
        return {'status':'UNKNOWN','reason':'incomplete_diagnostics'}
    c=extra['final_contact'];g=extra['final_geometry']
    if not c or not g or c['step']!=steps or g['step']!=steps:return {'status':'UNKNOWN','reason':'final_state_missing'}
    for r in (c,g):
        if r['payload'].get('logging_only') is not True or r['payload'].get('used_by_policy_or_planner') is not False:
            return {'status':'UNKNOWN','reason':'diagnostic_scope_not_offline'}
    d=c['payload']['diagnostic'];name=d.get('object_name');subject=previous.get('primary_target_id');destination=stage.get('primary_target_id')
    if not isinstance(name,str) or 'O'+hashlib.sha256(('objects/'+name).encode()).hexdigest()[:10]!=subject:
        return {'status':'UNKNOWN','reason':'contact_subject_identity_mismatch'}
    actual_ids={e['binding']['target_id'] for e in result.get('events',[]) if (e.get('stage') or {}).get('operation')=='place' and e.get('binding')}
    if actual_ids and actual_ids!={destination}:return {'status':'UNKNOWN','reason':'natural_and_actual_destination_mismatch'}
    obj=d.get('object_position');target=g['payload'].get('region_positions_world_m',{}).get(destination)
    if not all(isinstance(v,list) and len(v)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in v) for v in (obj,target)):
        return {'status':'UNKNOWN','reason':'invalid_or_missing_recorded_position'}
    return {'status':'RECORDED_FINAL_XY','state_step':steps,'xy_error_m':math.hypot(obj[0]-target[0],obj[1]-target[1]),
        'abs_z_error_m':abs(obj[2]-target[2]),'subject_id':subject,'destination_id':destination,
        'source':'Natural plan plus actual binding and logging-only recorded coordinates; not an oracle control input.',
        'includes_unsuccessful_episode':result.get('status')=='failure','threshold_used_for_control':False}

def classify(result,issues,scan,extra,max_steps=600):
    reported=result.get('status');reason=str(result.get('reason') or result.get('error') or '')
    if not result:return 'unknown','not_yet_recorded'
    if reported not in ('success','failure'):
        explicit=result.get('outcome_category')
        if explicit=='timeout':return 'unknown','infrastructure_timeout'
        if explicit in ('invalid_pair','prerequisite_failed','unattempted','infrastructure_error'):
            return 'unknown',explicit
        if 'timeout' in reason.lower() or 'deadline' in reason.lower():return 'unknown','infrastructure_timeout'
        return 'unknown','invalid_or_incomplete_execution'
    if issues:return 'unknown','invalid_evidence'
    steps=result['steps'];success=result.get('success_observed') is True
    if reported=='success' and success and extra['scorer_true_steps']==[steps]:return 'success','benchmark_success'
    if reported=='failure' and not success and not extra['scorer_true_steps']:
        if steps==max_steps:return 'failure','task_horizon_timeout'
        if scan['terminals'][0].get('environment_done') is True:return 'failure','environment_terminated_unsuccessful'
    return 'unknown','inconsistent_task_termination'

def analyze_case(run,case,manifest,global_issues):
    path=run/f"worker-{case['worker']}"/'run/results'/f"{case['case_id']}.json"
    trace=path.parent.parent/'traces'/case['case_id'];issues=list(global_issues);result={}
    try:result=load(path) if path.is_file() else {}
    except (OSError,ValueError) as exc:issues.append('result_parse:'+str(exc))
    scan,extra=scan_journal(trace/'records.jsonl')
    metadata=None
    try:metadata=load(trace/'trace.json') if (trace/'trace.json').is_file() else None
    except (OSError,ValueError) as exc:issues.append('trace_metadata_parse:'+str(exc))
    if result:
        for key in ('case_id','seed','task','comparison_arm','benchmark','model','arm','max_steps','pair_index','pair_position'):
            if result.get(key)!=case.get(key):issues.append('manifest_case_mismatch:'+key)
        if result.get('purpose')!=PURPOSE or result.get('manifest_sha256')!=manifest['manifest_sha256']:issues.append('result_purpose_or_manifest_mismatch')
        if result.get('parameter_updates') not in (None,0) or result.get('action_modifications') not in (None,0):issues.append('training_or_action_modifications')
        if result.get('status') in ('success','failure'):
            issues.extend(gate.trace_integrity_issues(result,metadata,scan))
            try:issues.extend(initial_checks(result,scan,extra))
            except (KeyError,TypeError,ValueError) as exc:issues.append('initial_evidence_invalid:'+str(exc))
            if extra['action_contract_issues']:issues.append('executed_action_contract_invalid')
            if result.get('parameter_updates')!=0 or result.get('action_modifications')!=0:issues.append('complete_result_missing_zero_change_contract')
    status,category=classify(result,issues,scan,extra)
    try:diagnostics=final_xy(result,extra) if status in ('success','failure') else {'status':'UNKNOWN','reason':'outcome_incomplete_or_invalid'}
    except (KeyError,ValueError,TypeError) as exc:diagnostics={'status':'UNKNOWN','reason':'invalid_diagnostic:'+str(exc)}
    counts=gate.event_metrics(result.get('events') or scan['events'])['counts']
    summary={**{k:case.get(k) for k in ('case_id','seed','task','comparison_arm','pair_key','pair_index','pair_position')},
        'outcome':status,'category':category,'reported_status':result.get('status'),'reason':result.get('reason') or result.get('error'),
        'reported_outcome_category':result.get('outcome_category'),
        'steps':result.get('steps'),'elapsed_seconds':result.get('elapsed_seconds'),'integrity_issues':issues,
        'result_path':str(path),'result_sha256':sha(path) if path.is_file() else None,
        'journal_sha256':extra.get('journal_sha256'),'journal_record_count':scan['records_scanned'],
        'initial_signature':result.get('initial_signature'),'initial_plan':result.get('initial_plan'),
        'pair_reference':result.get('pair_reference'),'natural_plan_matches_paired_reference':result.get('natural_plan_matches_paired_reference'),
        'initial_gate_events':extra['initial_comparisons'],'natural_plan_gate_events':extra['initial_plan_comparisons'],
        'native_chunk_sha256':result.get('native_chunk_sha256'),'event_counts':counts,'final_xy':diagnostics,
        'array_hash_status':'Descriptor/source integrity only here; complete NPZ byte audit is required separately before archival acceptance.'}
    return summary

def make_pairs(cases,seeds):
    pairs=[]
    for seed in seeds:
        a,b=(next(c for c in cases if c['seed']==seed and c['comparison_arm']==arm) for arm in ARMS)
        complete=all(c['outcome'] in ('success','failure') for c in (a,b))
        initial=bool(a['initial_signature']) and a['initial_signature']==b['initial_signature']
        plan=bool(a['initial_plan']) and isinstance(a['initial_plan'],list) and a['initial_plan']==b['initial_plan']
        first=min((a,b),key=lambda c:c['pair_position']);second=max((a,b),key=lambda c:c['pair_position'])
        ie=second['initial_gate_events'];pe=second['natural_plan_gate_events']
        admission=len(ie)==1 and ie[0]['payload'].get('gate',{}).get('passed') is True and len(pe)==1 and pe[0]['payload'].get('passed') is True
        ref=second.get('pair_reference') or {}
        reference_match=(ref.get('case_id')==first['case_id'] and ref.get('comparison_arm')==first['comparison_arm']
            and ref.get('result_sha256')==first['result_sha256'] and ref.get('injected') is False
            and ref.get('expected_initial')==first['initial_signature'] and ref.get('expected_plan')==first['initial_plan']
            and second.get('natural_plan_matches_paired_reference') is True
            and len(pe)==1 and pe[0]['payload'].get('expected_plan')==first['initial_plan'])
        eligible=complete and initial and plan and admission and reference_match
        xys=all(c['final_xy'].get('status')=='RECORDED_FINAL_XY' for c in (a,b))
        pairs.append({'seed':seed,'pair_index':a['pair_index'],'point_case_id':a['case_id'],'regions_case_id':b['case_id'],
            'point_outcome':a['outcome'],'regions_outcome':b['outcome'],'complete':complete,
            'initial_signature_equal':initial,'natural_plan_equal_and_nonempty':plan,'second_arm_preaction_admission':admission,
            'second_arm_reference_matches_actual_first_result_sha_and_natural_plan':reference_match,
            'eligible':eligible,'success_delta_regions_minus_point':int(b['outcome']=='success')-int(a['outcome']=='success') if eligible else None,
            'final_xy_delta_cm_regions_minus_point':100*(b['final_xy']['xy_error_m']-a['final_xy']['xy_error_m']) if eligible and xys else None,
            'native_chunk_comparison':gate.compare_chunk_hashes(a['native_chunk_sha256'],b['native_chunk_sha256'])})
    return pairs

def summaries(cases,pairs):
    eligible=[p for p in pairs if p['eligible']];selected={cid for p in eligible for cid in (p['point_case_id'],p['regions_case_id'])}
    arms={}
    for arm in ARMS:
        allrows=[c for c in cases if c['comparison_arm']==arm];rows=[c for c in allrows if c['case_id'] in selected]
        known=[c for c in allrows if c['outcome'] in ('success','failure')];success=sum(c['outcome']=='success' for c in rows)
        unknown=sum(c['outcome']=='unknown' for c in allrows);all_success=sum(c['outcome']=='success' for c in known)
        arms[arm]={'planned':len(allrows),'accounting':dict(Counter(c['category'] for c in allrows)),
            'success':all_success,'task_failure_including_horizon_timeout':sum(c['outcome']=='failure' for c in known),'unknown':unknown,
            'registered_success_fraction_identification_bounds':[all_success/len(allrows),(all_success+unknown)/len(allrows)],
            'paired_eligible_denominator':len(rows),'paired_eligible_successes':success,
            'paired_eligible_success_rate':success/len(rows) if rows else None,'paired_eligible_success_rate_wilson95':wilson(success,len(rows)),
            'final_xy_all_known_including_failures':distribution([c['final_xy']['xy_error_m'] for c in known if c['final_xy'].get('status')=='RECORDED_FINAL_XY']),
            'final_xy_paired_eligible_including_failures':distribution([c['final_xy']['xy_error_m'] for c in rows if c['final_xy'].get('status')=='RECORDED_FINAL_XY']),
            'final_xy_known_missing':sum(c['final_xy'].get('status')!='RECORDED_FINAL_XY' for c in known)}
    wins=sum(p['success_delta_regions_minus_point']==1 for p in eligible);losses=sum(p['success_delta_regions_minus_point']==-1 for p in eligible);n=len(eligible)
    xy=[p['final_xy_delta_cm_regions_minus_point'] for p in eligible if p['final_xy_delta_cm_regions_minus_point'] is not None]
    paired={'planned_pairs':len(pairs),'eligible_pairs':n,'excluded_unknown_or_mismatched_pairs':len(pairs)-n,
        'regions_only_success':wins,'point_only_success':losses,'both_success':sum(p['point_outcome']=='success' and p['regions_outcome']=='success' for p in eligible),
        'both_failure':sum(p['point_outcome']=='failure' and p['regions_outcome']=='failure' for p in eligible),
        'risk_difference_regions_minus_point':(wins-losses)/n if n else None,
        'risk_difference_conservative95':paired_effect_ci(wins,losses,n),
        'risk_difference_ci_method':'Difference of 97.5% Clopper-Pearson intervals for win/loss probabilities; Bonferroni simultaneous >=95%, conservative.',
        'mcnemar_exact_two_sided_p':exact_mcnemar(wins,losses) if n else None,
        'mcnemar_method':'Two-sided exact binomial test conditional on discordant pairs, capped at1; primary one registered contrast.',
        'final_xy_paired_delta_cm':{'n':len(xy),'median':statistics.median(xy) if xy else None,'mean':statistics.mean(xy) if xy else None,
            'interpretation':'Negative favors regions. Includes unsuccessful completed episodes; secondary descriptive endpoint.'}}
    return arms,paired

def analyze(run,root=ROOT):
    manifest=load(run/'manifest.json');mismatches=manifest_check(manifest,root)
    global_issues=['registered_source_mismatch:'+p for p in mismatches]
    cases=[analyze_case(run,c,manifest,global_issues) for c in manifest['cases']]
    pairs=make_pairs(cases,manifest['seeds']);arms,paired=summaries(cases,pairs)
    receipt=load(run/'launch-receipt.json') if (run/'launch-receipt.json').is_file() else None
    return {'status':'PROSPECTIVE_OFFLINE_ANALYSIS','purpose':PURPOSE,'manifest_sha256':manifest['manifest_sha256'],
        'source_files_match':not mismatches,'source_mismatches':mismatches,'run':str(run),'cases':cases,'pairs':pairs,
        'arms':arms,'paired':paired,'launch_cleanup':None if not receipt else {k:receipt.get(k) for k in ('status','owned_processes_stopped','cleanup_errors','elapsed_seconds')},
        'analysis_sha256':sha(Path(__file__)),'complete_trace_gate_sha256':sha(GATE),
        'new_model_generations':0,'new_env_resets':0,'new_env_steps':0,
        'limits':['Only the fixed 12 new seed pairs enter this denominator; historical development results are never pooled.',
            '600-action horizon exhaustion is a known task failure; infrastructure/wall-clock timeout remains unknown.',
            'Missing/invalid/partial traces cannot become task failures or successes; excluded pair counts remain visible.',
            'Paired rates/CI use the same eligible seed subset; if pairs are missing, no complete24-case claim is justified.',
            'Final XY is at each episode termination (success earlier than600); it is not a common-time600 trajectory comparison.',
            'Distances include known failures and are relative to naturally selected and bound regions; geometry thresholds are never control inputs.',
            'Wilson intervals are approximate; paired effect interval is conservative. Twelve seeds have limited power, and a nonsignificant result is not equivalence.',
            'One task, one model, one fixed implementation; no cross-model/benchmark or real-world generalization claim.',
            'Complete archive array-hash and GPU/renderer identity audits remain separate mandatory evidence checks.']}

def self_test():
    assert exact_mcnemar(6,0)==.03125 and exact_mcnemar(5,0)==.0625 and exact_mcnemar(1,0)==1
    assert wilson(0,12)[0]==0 and .24<wilson(0,12)[1]<.25
    assert .26<clopper_pearson(0,12)[1]<.27
    assert paired_effect_ci(0,0,12)[0]<0<paired_effect_ci(0,0,12)[1]
    assert classify({},[],{}, {})==('unknown','not_yet_recorded')
    assert classify({'status':'unknown','reason':'TimeoutError: deadline'},[],{}, {})==('unknown','infrastructure_timeout')
    assert classify({'status':'success'},['trace_partial'],{}, {})==('unknown','invalid_evidence')
    e={'scorer_true_steps':[]};s={'terminals':[{'environment_done':False}]}
    assert classify({'status':'failure','steps':600,'success_observed':False},[],s,e)==('failure','task_horizon_timeout')
    assert classify({'status':'failure','steps':100,'success_observed':False},[],s,e)[0]=='unknown'
    assert distribution([.07332475757839638,.036382468779741854,.012657587812001547])['median_cm']==3.6382468779741854
    print('CPU_SELF_TEST_PASS: exact statistics, unknown/timeout separation, all-outcome distances')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if not args.run or not args.output:parser.error('--run and --output required')
        if args.output.exists():raise FileExistsError('Never overwrite evidence')
        report=analyze(args.run);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print(json.dumps({'status':report['status'],'paired':report['paired']}))
