"""Pure offline prospective point/masks audit; never runs models/simulation.

Copied from work/id60-regions-validation/analyze_regions_validation.py
(sha256 d9cb8f468b9c493432c5a291b96355f0f693da001296a562b810de2ee57b4aff) and
its resume wrapper analyze_regions_validation_combined.py
(sha256 3c1cb1e7f85991473fda1baf29cac564a4871154a8bab11ba5dfe9b840955ff9).
Adaptations (work/id60-masks-design/harness-map-v1.md section 11.6):

* PURPOSE 'masks_point_unseen_seed_validation', ARMS ('point', 'masks'). Output
  key names derive from ARMS at call time (masks_case_id,
  success_delta_masks_minus_point, ...).
* Parent lineage is the regions validation manifest (canonical 9414bd65...):
  path, file sha256, recomputed canonical hash and every parent source hash.
* The pair count comes from manifest['planned_pairs'] (12, 24 or 36) instead
  of a hard-coded 12; planned_cases must equal 2 * planned_pairs.
* Each case is read from worker-<case['worker']> of its source directory, for
  any registered worker index; a result or trace for the same case under an
  unregistered worker directory is an integrity issue (never silently used).
* Optional registered resume directories (generic form of the combined
  wrapper): each resume must carry resume-plan.json with the zero-step,
  whole-pair rule; any chain length and pair count.
* A descriptive 'fidelity' summary from 'mask_presentation' trace events
  (draw coverage per stage and camera).  The zero requirement (gate G5:
  inside_mask_pixels_changed == 0 wherever reported) is evaluated for the
  masks (treatment) arm only.  In the point arm the frozen bbox/crosshair cue
  may cover mask pixels by design; that count is reported descriptively as
  point_cue_pixels_inside_masks (read from that field, or from the older
  point-arm field name inside_mask_pixels_changed) with no requirement.  It is
  computed after classification and never conditions outcomes, eligibility,
  pairing or statistics.
* Resume plans follow the regions resume rule and schema exactly
  (prior_output, prior_completed_result_sha256, prior_interrupted_zero_step,
  owed_case_ids), as written by scripts/run_id60_masks_validation_resume_local.py.
* Secondary descriptive episode-length summary in addition to terminal XY.
* Explicit ``--pilot-engineering`` mode accepts only the fixed three development
  pairs and pilot purpose, reports all six trace checks and descriptive outcome
  counts, and emits no prospective effect, confidence interval or significance
  claim.  The default prospective validator still rejects every pilot manifest.

Unchanged: one complete streaming journal pass per case; missing/partial/
invalid execution never becomes a task failure; the 600-action horizon is a
known failure, a wall-clock/infrastructure timeout is UNKNOWN; the 9-key
initial signature reconstruction; exact two-sided McNemar; Bonferroni
Clopper-Pearson paired risk-difference CI; Wilson per arm; terminal XY.
All world-coordinate measurements remain post-hoc, never control inputs.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[2]
PURPOSE='masks_point_unseen_seed_validation'
PILOT_PURPOSE='masks_point_dev_pilot'
ARMS=('point','masks')
ALLOWED_PAIR_COUNTS=(12,24,36)
PARENT_PATH='work/id60-regions-validation/manifest.json'
PARENT_MANIFEST_SHA256='9414bd6546022392a2c813f1b8999a050a2c5230592f6f67fcc61971964eb9e4'
DEVELOPMENT_SEEDS=(20261025,20261026,20261027)
CAMERAS=('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')
MASK_EVENT='mask_presentation'
INSIDE_FIELD='inside_mask_pixels_changed'  # masks arm: G5 requires 0 wherever reported
POINT_CUE_FIELD='point_cue_pixels_inside_masks'  # point arm: descriptive only (older name: INSIDE_FIELD)
MASK_LEAF_KEYS=('mask_px','ring_px')
RESUME_PLAN_KEYS=('prior_output','prior_completed_result_sha256','prior_interrupted_zero_step','owed_case_ids')
MASK_STRUCTURAL_KEYS=frozenset({'cameras','camera','roles','role','per_camera','by_camera','by_role',
    'masks','mask','entities','views','rings','detail','details'})
GATE=ROOT/'work/id60-targeted-parser/analyze_targeted.py'
spec=importlib.util.spec_from_file_location('masks_validation_complete_gate',GATE)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def control():return ARMS[0]
def treatment():return ARMS[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def local(path):
    """Map a WSL /mnt/<drive>/ path recorded by a launcher to this host (Windows only)."""
    text=str(path)
    if text.startswith('/mnt/') and len(text)>6 and text[6:7]=='/' and Path('C:/').exists():
        return Path(text[5].upper()+':/'+text[7:])
    return Path(text)

def result_path(source,case):return Path(source)/f"worker-{case['worker']}"/'run/results'/f"{case['case_id']}.json"
def trace_dir(source,case):return Path(source)/f"worker-{case['worker']}"/'run/traces'/case['case_id']

# ---------------------------------------------------------------- statistics

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

def plain_distribution(values):
    return {'n':len(values),'median':statistics.median(values) if values else None,
            'mean':statistics.mean(values) if values else None,
            'minimum':min(values) if values else None,'maximum':max(values) if values else None}

# ---------------------------------------------------------------- registration

def pair_count(manifest):
    n=manifest.get('planned_pairs')
    if type(n) is not int or n not in ALLOWED_PAIR_COUNTS:
        raise ValueError(f'planned_pairs must be one of {ALLOWED_PAIR_COUNTS}')
    if manifest.get('planned_cases')!=2*n:raise ValueError('planned_cases must equal 2*planned_pairs')
    return n

def manifest_check(manifest,root):
    """Prospective validator: development pilots are never admitted implicitly."""
    return _manifest_check(manifest,root,pilot=False)


def pilot_manifest_check(manifest,root):
    """Engineering-only validator for exactly the three registered development pairs."""
    return _manifest_check(manifest,root,pilot=True)


def _manifest_check(manifest,root,*,pilot):
    expected_purpose=PILOT_PURPOSE if pilot else PURPOSE
    if manifest.get('purpose')!=expected_purpose:raise ValueError('Unexpected '+('pilot' if pilot else 'prospective')+' purpose')
    if manifest.get('manifest_sha256')!=canonical({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Manifest canonical hash mismatch')
    if pilot:
        n=len(DEVELOPMENT_SEEDS)
        if manifest.get('planned_pairs')!=n or manifest.get('planned_cases')!=2*n \
                or manifest.get('seeds')!=list(DEVELOPMENT_SEEDS) or manifest.get('prospective_validation') is not False \
                or manifest.get('seed_role')!='development_seeds_already_used_not_prospective_evidence' \
                or 'seed_audit' in manifest:
            raise ValueError('Pilot must contain exactly the fixed development pairs and no prospective seed audit')
    else:
        n=pair_count(manifest)
    seeds=manifest.get('seeds');cases=manifest.get('cases')
    if not isinstance(seeds,list) or len(seeds)!=n or len(set(seeds))!=n or any(type(s) is not int or s<0 for s in seeds) \
            or not isinstance(cases,list) or len(cases)!=2*n or not all(isinstance(c,dict) for c in cases):
        raise ValueError(f'Exactly {n} preregistered pairs/{2*n} cases required')
    if len({c.get('case_id') for c in cases})!=2*n:raise ValueError('Duplicate case ID')
    if set(manifest.get('treatments') or {})!=set(ARMS):raise ValueError('Registered treatments differ from analysed arms')
    if not pilot and set(seeds)&set(DEVELOPMENT_SEEDS):raise ValueError('Development seed reused in prospective matrix')
    if any(type(c.get('worker')) is not int or c['worker']<0 for c in cases):raise ValueError('Invalid worker index')
    if len({(c['worker'],c.get('order_in_worker')) for c in cases})!=2*n:raise ValueError('Duplicate worker execution slot')
    if not pilot:
        audit_info=manifest.get('seed_audit') or {}
        audit_path=root/audit_info.get('path','__missing_seed_audit__')
        if not audit_path.is_file() or sha(audit_path)!=audit_info.get('sha256'):
            raise ValueError('Seed audit file hash mismatch')
        audit=load(audit_path)
        if audit.get('status')!='PASS' or audit.get('ordered_seeds')!=seeds or audit.get('all_selected_absent') is not True:
            raise ValueError('Seed exclusion/matrix mismatch')
        if audit.get('selected_seeds')!=seeds or ('ordered_seeds' in audit_info and audit_info['ordered_seeds']!=seeds):
            raise ValueError('Seed exclusion/matrix mismatch')
        matches=audit.get('selected_seed_prior_matches')
        if not isinstance(matches,dict) or any(matches.get(str(s))!=[] for s in seeds):
            raise ValueError('Selected seed has prior evidence')
    for index,seed in enumerate(seeds):
        group=[c for c in cases if c.get('seed')==seed]
        if len(group)!=2 or {c.get('comparison_arm') for c in group}!=set(ARMS):raise ValueError('Missing pair arm')
        if {c.get('pair_position') for c in group}!={0,1} or {c.get('pair_index') for c in group}!={index}:
            raise ValueError('Invalid counterbalanced pair indices')
        expected_order=ARMS if index%2==0 else tuple(reversed(ARMS))
        if tuple(c['comparison_arm'] for c in sorted(group,key=lambda c:c['pair_position']))!=expected_order:
            raise ValueError('Counterbalanced order changed')
        if len({c.get('pair_key') for c in group})!=1:raise ValueError('Pair key differs within pair')
        expected_key=canonical({'task':'CoffeeSetupMug','seed':seed,'model':'xr1','purpose':expected_purpose})
        if any(c.get('pair_key')!=expected_key for c in group):raise ValueError('Pair key does not bind the registered purpose')
        if len({c['worker'] for c in group})!=1:raise ValueError('A pair cannot span workers')
        if any(c.get('case_id')!=canonical({k:v for k,v in c.items() if k!='case_id'}) for c in group):
            raise ValueError('Case identity hash mismatch')
        if any(c.get('max_steps')!=600 or c.get('task')!='CoffeeSetupMug' or c.get('benchmark')!='robocasa'
            or c.get('model')!='xr1' or c.get('arm')!='full' for c in group):raise ValueError('Unexpected task/model/horizon')
    files=manifest.get('source_files')
    if not isinstance(files,dict) or not files:raise ValueError('Missing registered source hashes')
    parent=manifest.get('parent') or {}
    if parent.get('path')!=PARENT_PATH or parent.get('manifest_sha256')!=PARENT_MANIFEST_SHA256:
        raise ValueError('Unexpected parent manifest')
    parent_path=root/parent['path']
    if not parent_path.is_file() or sha(parent_path)!=parent.get('sha256'):raise ValueError('Parent lineage mismatch')
    frozen=load(parent_path)
    if frozen.get('manifest_sha256')!=PARENT_MANIFEST_SHA256 or \
            canonical({k:v for k,v in frozen.items() if k!='manifest_sha256'})!=PARENT_MANIFEST_SHA256:
        raise ValueError('Unexpected development parent')
    if not pilot and set(seeds)&set(frozen.get('seeds') or []):raise ValueError('Parent study seed reused')
    if not isinstance(frozen.get('source_files'),dict) or any(files.get(name)!=digest for name,digest in frozen['source_files'].items()):
        raise ValueError('Frozen source omitted/changed')
    if manifest.get('checkpoint_sha256')!=frozen.get('checkpoint_sha256'):raise ValueError('Checkpoint identity changed')
    mismatches=[name for name,want in files.items() if not (root/name).is_file() or sha(root/name)!=want]
    return mismatches

# ---------------------------------------------------------------- journal scan

def _count(value):return value if type(value) is int and value>=0 else None

def _mask_leaves(value,path=()):
    if isinstance(value,dict):
        if any(k in value for k in MASK_LEAF_KEYS):
            yield path,value;return
        for k,v in value.items():yield from _mask_leaves(v,path+(str(k),))
    elif isinstance(value,list):
        for v in value:yield from _mask_leaves(v,path)

def _field_values(value,field):
    if isinstance(value,dict):
        for k,v in value.items():
            if k==field:yield v
            else:yield from _field_values(v,field)
    elif isinstance(value,list):
        for v in value:yield from _field_values(v,field)

def _inside_values(value):return _field_values(value,INSIDE_FIELD)

def _operation(p):
    stage=p.get('stage')
    for value in (p.get('operation'),p.get('stage_operation'),stage.get('operation') if isinstance(stage,dict) else None):
        if isinstance(value,str) and value:return value
    return None

def mask_event_summary(step,p):
    """Compact, schema-tolerant summary of one 'mask_presentation' event (descriptive only).

    Leaves are dicts carrying 'mask_px' and/or 'ring_px'; camera and role come
    from their own 'camera'/'role' fields or from the enclosing mapping keys.
    """
    leaves=[]
    for path,leaf in _mask_leaves({k:v for k,v in p.items() if k!='kind'}):
        leaves.append({'path':list(path),'camera':leaf.get('camera') if isinstance(leaf.get('camera'),str) else None,
            'role':leaf.get('role') if isinstance(leaf.get('role'),str) else None,
            'mask_px':_count(leaf.get('mask_px')),'ring_px':_count(leaf.get('ring_px')),
            'drawn':leaf.get('drawn') if isinstance(leaf.get('drawn'),bool) else None})
    return {'step':step,'arm':p.get('arm'),'operation':_operation(p),'status':p.get('status'),
        'drawn':p.get('drawn') if isinstance(p.get('drawn'),bool) else None,
        'scope_ok':p.get('logging_only') is True and p.get('used_by_policy_or_planner') is False,
        'leaves':leaves,'inside':list(_inside_values(p)),'point_cue_inside':list(_field_values(p,POINT_CUE_FIELD))}

def scan_journal(path):
    scan={'initial_reset_count':0,'initial_fingerprint':None,'initial_raw_rgb_sha256':None,
        'initial_proprio_sha256':None,'executed_steps_observed':0,'events':[],'terminal_observed':False,
        'records_scanned':0,'journal_present':path.is_file(),'scan_scope':'entire_journal',
        'record_counts':{},'scan_issues':[],'observation_steps':[],'action_steps':[],
        'action_next_steps':[],'proposal_steps':[],'prepare_steps':[],'terminals':[]}
    extra={'initial_comparisons':[],'initial_plan_comparisons':[],'first_observation':None,
        'initial_geometry':None,'final_contact':None,'final_geometry':None,
        'contact_steps':[],'geometry_steps':[],'first_action_sequence':None,
        'scorer_true_steps':[],'action_contract_issues':[],'diagnostic_scope_issues':[],'renderer_admission':None,
        'mask_events':[]}
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
                elif event==MASK_EVENT:
                    # Descriptive fidelity only; never feeds integrity, outcome or diagnostics scope.
                    extra['mask_events'].append(mask_event_summary(step,p))
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

# ---------------------------------------------------------------- fidelity (descriptive only)

def inside_key(arm):
    """Masks (treatment) arm: the G5 field inside_mask_pixels_changed, required 0 wherever reported.
    Point (control) arm: point_cue_pixels_inside_masks, descriptive only (the frozen bbox/crosshair cue may
    cover mask pixels by design; read from that field, else from the older point-arm name)."""
    return INSIDE_FIELD if arm==treatment() else POINT_CUE_FIELD

def _new_inside():return {'events_reporting':0,'events_missing':0,'events_nonzero':0,'invalid_values':0,'max':None,
    'drawn_events_missing':0,'source_fields':{}}
def _new_camera():return {'mask_available_steps':0,'drawn_steps':0,'roles':{}}
def _new_role():return {'available_steps':0,'drawn_steps':0,'drawn_and_available_steps':0}
def _new_fidelity(arm=None):
    return {'events':0,'drawn_events':0,'status_counts':{},'arm_label_mismatch_events':0,'scope_issue_events':0,
        'events_without_mask_leaves':0,inside_key(treatment() if arm is None else arm):_new_inside(),'by_operation':{}}

def _arm_inside_values(e,arm):
    """(values, source field) of one summarized event for this arm's inside-pixel audit."""
    if arm==treatment():return list(e.get('inside') or []),INSIDE_FIELD
    values=list(e.get('point_cue_inside') or [])
    if values:return values,POINT_CUE_FIELD
    return list(e.get('inside') or []),INSIDE_FIELD  # point arm under the older field name

def fidelity_counts(events,arm,cameras=CAMERAS):
    """Integer draw-coverage counts per operation (stage), camera and role; plus this arm's inside-pixel
    audit (masks: inside_mask_pixels_changed; point: point_cue_pixels_inside_masks).
    Returns (counts, steps with nonzero inside-pixel values)."""
    cameras=list(cameras);out=_new_fidelity(arm);key=inside_key(arm);nonzero=[]
    for e in events:
        out['events']+=1
        status=str(e.get('status'));out['status_counts'][status]=out['status_counts'].get(status,0)+1
        if e.get('arm')!=arm:out['arm_label_mismatch_events']+=1
        if not e.get('scope_ok'):out['scope_issue_events']+=1
        if not e.get('leaves'):out['events_without_mask_leaves']+=1
        event_flag=e['drawn'] if isinstance(e.get('drawn'),bool) else arm==treatment()
        op=e.get('operation') or 'no_active_stage'
        row=out['by_operation'].setdefault(op,{'steps':0,'cameras':{c:_new_camera() for c in cameras}})
        row['steps']+=1
        seen={}
        for leaf in e.get('leaves') or []:
            path=leaf.get('path') or []
            cam=leaf.get('camera') or next((k for k in path if k in cameras),'_unspecified_camera')
            role=leaf.get('role') or next((k for k in reversed(path) if k not in cameras and k not in MASK_STRUCTURAL_KEYS),'_unspecified_role')
            available=(leaf.get('mask_px') or 0)>0
            leaf_flag=leaf['drawn'] if isinstance(leaf.get('drawn'),bool) else (leaf.get('ring_px') or 0)>0
            drawn=bool(event_flag and leaf_flag)
            state=seen.setdefault(cam,{}).setdefault(role,[False,False])
            state[0]|=available;state[1]|=drawn
        any_drawn=False
        for cam,roles in seen.items():
            stats=row['cameras'].setdefault(cam,_new_camera())
            if any(a for a,_ in roles.values()):stats['mask_available_steps']+=1
            if any(d for _,d in roles.values()):stats['drawn_steps']+=1;any_drawn=True
            for role,(a,d) in roles.items():
                r=stats['roles'].setdefault(role,_new_role())
                r['available_steps']+=int(a);r['drawn_steps']+=int(d);r['drawn_and_available_steps']+=int(a and d)
        out['drawn_events']+=int(any_drawn)
        inside=out[key];values,source=_arm_inside_values(e,arm)
        if not values:
            inside['events_missing']+=1;inside['drawn_events_missing']+=int(any_drawn);continue
        inside['events_reporting']+=1;inside['source_fields'][source]=inside['source_fields'].get(source,0)+1
        valid=[v for v in values if type(v) is int and v>=0];inside['invalid_values']+=len(values)-len(valid)
        if valid:
            peak=max(valid);inside['max']=peak if inside['max'] is None else max(inside['max'],peak)
            if peak>0:inside['events_nonzero']+=1;nonzero.append(e.get('step'))
    return out,nonzero

def merge_fidelity(total,counts):
    for key in ('events','drawn_events','arm_label_mismatch_events','scope_issue_events','events_without_mask_leaves'):
        total[key]+=counts[key]
    for status,n in counts['status_counts'].items():total['status_counts'][status]=total['status_counts'].get(status,0)+n
    for field in (INSIDE_FIELD,POINT_CUE_FIELD):
        if field not in counts:continue
        a,b=total.setdefault(field,_new_inside()),counts[field]
        for key in ('events_reporting','events_missing','events_nonzero','invalid_values','drawn_events_missing'):a[key]+=b[key]
        for source,n in b.get('source_fields',{}).items():a['source_fields'][source]=a['source_fields'].get(source,0)+n
        if b['max'] is not None:a['max']=b['max'] if a['max'] is None else max(a['max'],b['max'])
    for op,row in counts['by_operation'].items():
        dst=total['by_operation'].setdefault(op,{'steps':0,'cameras':{}});dst['steps']+=row['steps']
        for cam,stats in row['cameras'].items():
            d=dst['cameras'].setdefault(cam,_new_camera())
            d['mask_available_steps']+=stats['mask_available_steps'];d['drawn_steps']+=stats['drawn_steps']
            for role,r in stats['roles'].items():
                x=d['roles'].setdefault(role,_new_role())
                for key in r:x[key]+=r[key]
    return total

def with_fractions(counts):
    out=copy.deepcopy(counts)
    for row in out['by_operation'].values():
        steps=row['steps']
        for stats in row['cameras'].values():
            stats['drawn_fraction_of_stage_steps']=stats['drawn_steps']/steps if steps else None
            stats['mask_available_fraction_of_stage_steps']=stats['mask_available_steps']/steps if steps else None
            for r in stats['roles'].values():
                r['drawn_fraction_of_stage_steps']=r['drawn_steps']/steps if steps else None
                r['drawn_fraction_when_mask_available']=r['drawn_and_available_steps']/r['available_steps'] if r['available_steps'] else None
    if INSIDE_FIELD in out:  # masks arm only: gate G5
        inside=out[INSIDE_FIELD]
        inside['requirement']='zero wherever reported (gate G5, masks arm only)'
        inside['all_reported_zero']=inside['events_reporting']>0 and inside['events_nonzero']==0 and inside['invalid_values']==0
        inside['all_drawn_events_reported_zero']=inside['all_reported_zero'] and inside['drawn_events_missing']==0
    if POINT_CUE_FIELD in out:
        out[POINT_CUE_FIELD]['requirement']=None
        out[POINT_CUE_FIELD]['interpretation']=('Role-mask pixels changed by the frozen point-arm bbox/crosshair cue; '
            'descriptive only, never a fidelity requirement.')
    return out

def case_fidelity(extra,arm,observation_steps):
    obs=extra.get('first_observation') or {};raw=obs.get('raw') if isinstance(obs.get('raw'),dict) else {}
    names=raw.get('camera_names');cameras=names if isinstance(names,list) and names and all(isinstance(c,str) for c in names) else list(CAMERAS)
    events=extra.get('mask_events') or []
    counts,nonzero=fidelity_counts(events,arm,cameras)
    return {**with_fractions(counts),'raw_counts':counts,'camera_names':list(cameras),
        'event_steps_match_observation_steps':sorted(e.get('step') for e in events if type(e.get('step')) is int)==sorted(
            s for s in observation_steps if type(s) is int) and len(events)==len(observation_steps),
        f'{inside_key(arm)}_nonzero_steps':nonzero[:20]}

def fidelity_summary(cases):
    arms={};t_arm=treatment()
    for arm in ARMS:
        key=inside_key(arm)
        rows=[c for c in cases if c['comparison_arm']==arm];total=_new_fidelity(arm);nonzero=[];mismatched=[]
        for c in rows:
            f=c.get('mask_fidelity') or {}
            if 'raw_counts' in f:merge_fidelity(total,f['raw_counts'])
            if f.get(key,{}).get('events_nonzero'):nonzero.append(c['case_id'])
            if f.get('events') and not f.get('event_steps_match_observation_steps'):mismatched.append(c['case_id'])
        row={**with_fractions(total),'cases':len(rows),
            'cases_with_mask_events':sum(bool((c.get('mask_fidelity') or {}).get('events')) for c in rows),
            'cases_with_complete_task_outcome':sum(c['outcome'] in ('success','failure') for c in rows),
            'cases_event_steps_differ_from_observation_steps':mismatched}
        row['cases_with_nonzero_inside_mask_pixels_changed' if arm==t_arm else 'cases_with_point_cue_pixels_inside_masks']=nonzero
        arms[arm]=row
    g5=arms[t_arm][INSIDE_FIELD]
    return {'descriptive_only':True,'conditions_primary_analysis':False,'event_kind':MASK_EVENT,
        'arms':arms,
        'inside_mask_gate':{'gate':'G5','arm':t_arm,'field':INSIDE_FIELD,'requirement':'0 wherever reported',
            'events_reporting':g5['events_reporting'],'events_missing':g5['events_missing'],
            'events_nonzero':g5['events_nonzero'],'invalid_values':g5['invalid_values'],'max':g5['max'],
            'all_reported_zero':g5['all_reported_zero'],'all_drawn_events_reported_zero':g5['all_drawn_events_reported_zero'],
            'cases_with_nonzero':arms[t_arm]['cases_with_nonzero_inside_mask_pixels_changed'],
            'control_arm':control(),'control_arm_field':POINT_CUE_FIELD,'control_arm_requirement':None},
        'reading_rule':('Leaves are dicts with mask_px and/or ring_px inside the event payload; camera and role come from '
            'their camera/role fields or enclosing mapping keys; operation from payload operation, stage_operation or '
            'stage.operation (else no_active_stage). A leaf counts as drawn when its own drawn flag (else ring_px>0) and the '
            'event drawn flag (else arm==treatment) hold. Masks arm: inside_mask_pixels_changed is collected at any depth. '
            'Point arm: point_cue_pixels_inside_masks is collected at any depth, else the older point-arm name '
            'inside_mask_pixels_changed; source_fields counts which name each event used.'),
        'interpretation':('Coverage counts steps per operation and camera on which any outline of that camera was drawn. '
            'Gate G5 applies to the masks (treatment) arm only: inside_mask_pixels_changed is 0 wherever reported. '
            'The point-arm count of mask pixels covered by the frozen point cue is descriptive and has no requirement. '
            'These numbers never alter outcomes, pair eligibility or the registered contrast.')}

# ---------------------------------------------------------------- cases, pairs, summaries

def unregistered_worker_copies(source,case):
    source=Path(source);own=f"worker-{case['worker']}";found=[]
    for pattern in (f"worker-*/run/results/{case['case_id']}.json",f"worker-*/run/traces/{case['case_id']}"):
        for p in sorted(source.glob(pattern)):
            name=p.relative_to(source).parts[0]
            if name!=own:found.append(name)
    return sorted(set(found))

def validate_task_time(row):
    """A missing historical timer is not reconstructed from whole-run elapsed."""
    timing=row.get('timing')
    if not isinstance(timing,dict):return ['task_time_v1_missing']
    issues=[]
    if timing.get('contract')!='composebrain_task_time_v1':issues.append('unsupported_contract')
    if timing.get('clock')!='monotonic':issues.append('unsupported_clock')
    if timing.get('start_boundary')!='reset_return_first_observation_before_augmentation_or_planning':issues.append('start_boundary')
    if timing.get('stop_boundary')!='terminal_env_step_return_before_logging':issues.append('stop_boundary')
    if timing.get('terminal_observed') is not True:issues.append('not_terminal')
    if timing.get('engineering_only') is not False:issues.append('engineering_cutoff_or_fixture')
    steps=row.get('steps')
    if type(steps) is not int or steps<1 or timing.get('terminal_control_steps')!=steps or timing.get('executed_control_steps')!=steps:
        issues.append('step_count_mismatch')
    elapsed=timing.get('task_elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:issues.append('invalid_task_elapsed')
    success_time=timing.get('success_completion_seconds')
    if row.get('outcome')=='success':
        if type(success_time) not in (int,float) or not math.isfinite(success_time) or success_time!=elapsed:
            issues.append('success_time_mismatch')
        if timing.get('termination_reason')!='success':issues.append('success_termination_mismatch')
    elif row.get('outcome')=='failure':
        if success_time is not None:issues.append('failure_has_success_time')
        if timing.get('termination_reason') not in ('environment_done','task_horizon'):issues.append('failure_termination_mismatch')
    else:issues.append('no_valid_task_outcome')
    return issues


def task_time_summary(cases):
    """Descriptive within-run measurements, with no inferred speed benefit."""
    def describe(values):
        values=sorted(values)
        if not values:return {'n':0,'median':None,'p90':None,'mean':None,'min':None,'max':None}
        position=.9*(len(values)-1);lo=math.floor(position);hi=math.ceil(position)
        return {**plain_distribution(values),'p90':values[lo]+(values[hi]-values[lo])*(position-lo)}
    arms={}
    for arm in ARMS:
        terminal=[c for c in cases if c.get('comparison_arm')==arm and c.get('outcome') in ('success','failure')
                  and not c.get('integrity_issues')]
        valid=[c for c in terminal if not validate_task_time(c)]
        arms[arm]={'eligible_terminal_cases':len(terminal),'valid_timing_cases':len(valid),
            'unavailable':[{'case_id':c.get('case_id'),'issues':validate_task_time(c)} for c in terminal if validate_task_time(c)],
            'success_completion_seconds':describe([c['timing']['success_completion_seconds'] for c in valid if c['outcome']=='success']),
            'all_terminal_task_elapsed_seconds':describe([c['timing']['task_elapsed_seconds'] for c in valid]),
            'all_terminal_control_steps':describe([c['steps'] for c in terminal if type(c.get('steps')) is int]),
            'success_control_steps':describe([c['steps'] for c in terminal if c['outcome']=='success' and type(c.get('steps')) is int])}
    return {'contract':'composebrain_task_time_v1','descriptive_only':True,'comparison_claim':None,
        'percentile_method':'linear interpolation at p*(n-1)','arms':arms,
        'limits':['Success time is conditioned on success; it is not a standalone efficiency comparison.',
            'Failures retain terminal elapsed time and control steps, but have null success completion time.',
            'Whole-run elapsed_seconds is never relabeled or substituted for missing task_time_v1.',
            'Hardware, model service placement, and concurrency form an execution cohort; no cross-cohort speedup is claimed.']}


def analyze_case(run,case,manifest,global_issues,*,expected_purpose=PURPOSE):
    path=result_path(run,case)
    trace=trace_dir(run,case);issues=list(global_issues);result={}
    for name in unregistered_worker_copies(run,case):issues.append('case_present_under_unregistered_worker:'+name)
    try:result=load(path) if path.is_file() else {}
    except (OSError,ValueError) as exc:issues.append('result_parse:'+str(exc))
    scan,extra=scan_journal(trace/'records.jsonl')
    metadata=None
    try:metadata=load(trace/'trace.json') if (trace/'trace.json').is_file() else None
    except (OSError,ValueError) as exc:issues.append('trace_metadata_parse:'+str(exc))
    if result:
        for key in ('case_id','seed','task','comparison_arm','benchmark','model','arm','max_steps','pair_index','pair_position'):
            if result.get(key)!=case.get(key):issues.append('manifest_case_mismatch:'+key)
        if result.get('purpose')!=expected_purpose or result.get('manifest_sha256')!=manifest['manifest_sha256']:issues.append('result_purpose_or_manifest_mismatch')
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
    summary={**{k:case.get(k) for k in ('case_id','seed','task','comparison_arm','pair_key','pair_index','pair_position','worker')},
        'outcome':status,'category':category,'reported_status':result.get('status'),'reason':result.get('reason') or result.get('error'),
        'reported_outcome_category':result.get('outcome_category'),
        'steps':result.get('steps'),'elapsed_seconds':result.get('elapsed_seconds'),'timing':result.get('timing'),'integrity_issues':issues,
        'result_path':str(path),'result_sha256':sha(path) if path.is_file() else None,
        'journal_sha256':extra.get('journal_sha256'),'journal_record_count':scan['records_scanned'],
        'initial_signature':result.get('initial_signature'),'initial_plan':result.get('initial_plan'),
        'pair_reference':result.get('pair_reference'),'natural_plan_matches_paired_reference':result.get('natural_plan_matches_paired_reference'),
        'initial_gate_events':extra['initial_comparisons'],'natural_plan_gate_events':extra['initial_plan_comparisons'],
        'native_chunk_sha256':result.get('native_chunk_sha256'),'event_counts':counts,'final_xy':diagnostics,
        'array_hash_status':'Descriptor/source integrity only here; complete NPZ byte audit is required separately before archival acceptance.'}
    # Computed after classification from the same pass; descriptive only.
    summary['mask_fidelity']=case_fidelity(extra,case.get('comparison_arm'),scan['observation_steps'])
    return summary

def make_pairs(cases,seeds):
    c_arm,t_arm=ARMS;pairs=[]
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
        steps_known=all(type(c.get('steps')) is int for c in (a,b))
        pairs.append({'seed':seed,'pair_index':a['pair_index'],f'{c_arm}_case_id':a['case_id'],f'{t_arm}_case_id':b['case_id'],
            f'{c_arm}_outcome':a['outcome'],f'{t_arm}_outcome':b['outcome'],'complete':complete,
            'initial_signature_equal':initial,'natural_plan_equal_and_nonempty':plan,'second_arm_preaction_admission':admission,
            'second_arm_reference_matches_actual_first_result_sha_and_natural_plan':reference_match,
            'eligible':eligible,f'success_delta_{t_arm}_minus_{c_arm}':int(b['outcome']=='success')-int(a['outcome']=='success') if eligible else None,
            f'final_xy_delta_cm_{t_arm}_minus_{c_arm}':100*(b['final_xy']['xy_error_m']-a['final_xy']['xy_error_m']) if eligible and xys else None,
            f'episode_steps_delta_{t_arm}_minus_{c_arm}':b['steps']-a['steps'] if eligible and steps_known else None,
            'native_chunk_comparison':gate.compare_chunk_hashes(a['native_chunk_sha256'],b['native_chunk_sha256'])})
    return pairs

def summaries(cases,pairs):
    c_arm,t_arm=ARMS
    eligible=[p for p in pairs if p['eligible']];selected={cid for p in eligible for cid in (p[f'{c_arm}_case_id'],p[f'{t_arm}_case_id'])}
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
            'final_xy_known_missing':sum(c['final_xy'].get('status')!='RECORDED_FINAL_XY' for c in known),
            'episode_steps_known_successes':plain_distribution([c['steps'] for c in known if c['outcome']=='success' and type(c.get('steps')) is int]),
            'episode_steps_paired_eligible':plain_distribution([c['steps'] for c in rows if type(c.get('steps')) is int])}
    delta=f'success_delta_{t_arm}_minus_{c_arm}';xykey=f'final_xy_delta_cm_{t_arm}_minus_{c_arm}';stepkey=f'episode_steps_delta_{t_arm}_minus_{c_arm}'
    wins=sum(p[delta]==1 for p in eligible);losses=sum(p[delta]==-1 for p in eligible);n=len(eligible)
    xy=[p[xykey] for p in eligible if p[xykey] is not None]
    step_deltas=[p[stepkey] for p in eligible if p[stepkey] is not None]
    paired={'planned_pairs':len(pairs),'eligible_pairs':n,'excluded_unknown_or_mismatched_pairs':len(pairs)-n,
        f'{t_arm}_only_success':wins,f'{c_arm}_only_success':losses,'both_success':sum(p[f'{c_arm}_outcome']=='success' and p[f'{t_arm}_outcome']=='success' for p in eligible),
        'both_failure':sum(p[f'{c_arm}_outcome']=='failure' and p[f'{t_arm}_outcome']=='failure' for p in eligible),
        f'risk_difference_{t_arm}_minus_{c_arm}':(wins-losses)/n if n else None,
        'risk_difference_conservative95':paired_effect_ci(wins,losses,n),
        'risk_difference_ci_method':'Difference of 97.5% Clopper-Pearson intervals for win/loss probabilities; Bonferroni simultaneous >=95%, conservative.',
        'mcnemar_exact_two_sided_p':exact_mcnemar(wins,losses) if n else None,
        'mcnemar_method':'Two-sided exact binomial test conditional on discordant pairs, capped at1; primary one registered contrast.',
        'final_xy_paired_delta_cm':{'n':len(xy),'median':statistics.median(xy) if xy else None,'mean':statistics.mean(xy) if xy else None,
            'interpretation':f'Negative favors {t_arm}. Includes unsuccessful completed episodes; secondary descriptive endpoint.'},
        'episode_steps_paired_delta':{**plain_distribution(step_deltas),
            'interpretation':f'{t_arm} minus {c_arm} executed steps at termination; secondary descriptive endpoint, success ends early.'}}
    return arms,paired

# ---------------------------------------------------------------- resume support

def resolve_sources(run,resumes,manifest):
    """Assign every registered case to exactly one directory: the run, or the latest
    registered resume that owes it. Enforces the pre-registered resume rule:
    whole pairs only, completed results unchanged, and an owed case may have no
    prior result and at most a zero-step (records 0, last_step -1) partial trace."""
    run=Path(run);cases=manifest['cases'];by_id={c['case_id']:c for c in cases};ids=set(by_id)
    source={cid:run for cid in ids};chain=[run];plans=[];reattempted={}
    for index,resume in enumerate(resumes,1):
        resume=Path(resume);plan=load(resume/'resume-plan.json')
        if not isinstance(plan,dict) or set(plan)!=set(RESUME_PLAN_KEYS):
            raise ValueError('Malformed resume plan: keys differ from the registered regions resume schema')
        prior=local(plan.get('prior_output','')).resolve()
        if prior not in {d.resolve() for d in chain}:raise ValueError('Resume plan names a different prior run')
        if load(resume/'manifest.json')!=manifest:raise ValueError('Prior and resume manifests differ')
        owed=plan.get('owed_case_ids');completed=plan.get('prior_completed_result_sha256');zero=plan.get('prior_interrupted_zero_step') or {}
        if not isinstance(owed,list) or not isinstance(completed,dict) or not isinstance(zero,dict) or len(set(owed))!=len(owed):
            raise ValueError('Malformed resume plan')
        owed=set(owed)
        if owed&set(completed) or owed|set(completed)!=ids:
            raise ValueError(f'Resume plan does not partition the {len(ids)} registered cases')
        for cid,digest in completed.items():
            path=result_path(source[cid],by_id[cid])
            if not path.is_file() or sha(path)!=digest:raise ValueError('Prior result changed after resume registration: '+cid)
        pair_flags={}
        for c in cases:pair_flags.setdefault(c['pair_key'],set()).add(c['case_id'] in owed)
        if any(len(flags)!=1 for flags in pair_flags.values()):raise ValueError('Resume must cover whole pairs only')
        if set(zero)-owed:raise ValueError('Interrupted case not owed by resume')
        for cid in sorted(owed):
            case=by_id[cid]
            if result_path(source[cid],case).exists():raise ValueError('Owed case already has a recorded result: '+cid)
            meta_path=trace_dir(source[cid],case)/'trace.json'
            if cid in zero:
                info=zero[cid]
                if not meta_path.is_file() or sha(meta_path)!=info.get('trace_sha256'):raise ValueError('Interrupted trace changed: '+cid)
                meta=load(meta_path)
                if meta.get('status')!='partial' or meta.get('records')!=0 or meta.get('last_step')!=-1:
                    raise ValueError('Interrupted case progressed; it must stay UNKNOWN: '+cid)
                reattempted[cid]=index
            elif trace_dir(source[cid],case).exists():
                raise ValueError('Owed case has an unregistered prior trace: '+cid)
        for cid in owed:source[cid]=resume
        chain.append(resume)
        plans.append({'index':index,'path':str(resume),'plan_sha256':canonical(plan),'owed_cases':len(owed),
            'reattempted_zero_step_case_ids':sorted(zero),'plan':plan})
    labels={str(d.resolve()):('prior' if i==0 else f'resume-{i}') for i,d in enumerate(chain)}
    return source,chain,plans,reattempted,labels

def cleanup(directory):
    receipt=load(directory/'launch-receipt.json') if (directory/'launch-receipt.json').is_file() else None
    return None if not receipt else {k:receipt.get(k) for k in ('status','owned_processes_stopped','cleanup_errors','elapsed_seconds')}

def base_limits(n):
    return [f'Only the fixed {n} new seed pairs enter this denominator; historical development results are never pooled.',
        '600-action horizon exhaustion is a known task failure; infrastructure/wall-clock timeout remains unknown.',
        'Missing/invalid/partial traces cannot become task failures or successes; excluded pair counts remain visible.',
        f'Paired rates/CI use the same eligible seed subset; if pairs are missing, no complete {2*n}-case claim is justified.',
        'Final XY is at each episode termination (success earlier than 600); it is not a common-time 600 trajectory comparison.',
        'Distances include known failures and are relative to naturally selected and bound regions; geometry thresholds are never control inputs.',
        f'Wilson intervals are approximate; paired effect interval is conservative. {n} seed pairs have limited power, and a nonsignificant result is not equivalence.',
        'One task, one model, one fixed implementation; no cross-model/benchmark or real-world generalization claim.',
        'Masks are oracle simulator segmentation and a geometric destination proxy: an upper-bound presentation cue, not perception.',
        'Mask-presentation fidelity is descriptive only; it never conditions outcomes, pair eligibility or the registered contrast.',
        'Complete archive array-hash and GPU/renderer identity audits remain separate mandatory evidence checks.']

def analyze(run,root=ROOT,resumes=()):
    run=Path(run);resumes=[Path(r) for r in resumes or ()]
    manifest=load(run/'manifest.json');mismatches=manifest_check(manifest,root);n=pair_count(manifest)
    source,chain,plans,reattempted,labels=resolve_sources(run,resumes,manifest)
    global_issues=['registered_source_mismatch:'+p for p in mismatches]
    cases=[]
    for case in manifest['cases']:
        directory=source[case['case_id']]
        row=analyze_case(directory,case,manifest,global_issues)
        row['source_run']=labels[str(directory.resolve())]
        row['reattempt_after_zero_step_kill']=case['case_id'] in reattempted
        cases.append(row)
    pairs=make_pairs(cases,manifest['seeds']);arms,paired=summaries(cases,pairs)
    c_arm,t_arm=ARMS
    report={'status':'PROSPECTIVE_OFFLINE_ANALYSIS','purpose':PURPOSE,'manifest_sha256':manifest['manifest_sha256'],
        'planned_pairs':n,'arms_order':list(ARMS),
        'source_files_match':not mismatches,'source_mismatches':mismatches,'run':str(run),
        'resume_runs':[str(r) for r in resumes],'resume_plans':plans,'cases':cases,'pairs':pairs,
        'arms':arms,'paired':paired,'fidelity':fidelity_summary(cases),'task_time':task_time_summary(cases),
        'launch_cleanup':cleanup(run),'resume_launch_cleanup':[cleanup(r) for r in resumes],
        'analysis_sha256':sha(Path(__file__)),'complete_trace_gate_sha256':sha(GATE),
        'new_model_generations':0,'new_env_resets':0,'new_env_steps':0,'limits':base_limits(n)}
    if resumes:
        registered=[{**c,'outcome':'unknown','category':'registered_rule_interrupted_trace_unknown'}
            if c['reattempt_after_zero_step_kill'] else c for c in cases]
        registered_pairs=make_pairs(registered,manifest['seeds'])
        registered_arms,registered_paired=summaries(registered,registered_pairs)
        report['registered_rule_sensitivity']={'reattempted_case_ids':sorted(reattempted),
            'arms':registered_arms,'paired':registered_paired,
            'pairs':[{k:p[k] for k in ('seed','eligible',f'{c_arm}_outcome',f'{t_arm}_outcome',f'success_delta_{t_arm}_minus_{c_arm}')}
                for p in registered_pairs]}
        if reattempted:
            report['limits'].insert(0,f'{len(reattempted)} registered case(s) were killed by an infrastructure shutdown before '
                'their first environment step and were re-attempted once under the pre-registered zero-step resume rule; '
                'registered_rule_sensitivity keeps them UNKNOWN.')
    return report


def analyze_pilot(run,root=ROOT):
    """Read the complete six-case development pilot without calculating a prospective treatment effect."""
    run=Path(run);root=Path(root)
    manifest=load(run/'manifest.json');mismatches=pilot_manifest_check(manifest,root)
    if (run/'resume-plan.json').exists():
        raise ValueError('Pilot engineering mode does not admit prospective resume plans')
    global_issues=['registered_source_mismatch:'+p for p in mismatches]
    cases=[analyze_case(run,case,manifest,global_issues,expected_purpose=PILOT_PURPOSE)
           for case in manifest['cases']]
    pairs=make_pairs(cases,manifest['seeds'])
    valid=[c for c in cases if c['outcome'] in ('success','failure') and not c['integrity_issues']]
    fidelity=fidelity_summary(cases)
    cadence=[c['case_id'] for c in cases if not c['mask_fidelity']['event_steps_match_observation_steps']
             or not c['mask_fidelity']['events']]
    return {'status':'PILOT_ENGINEERING_OFFLINE_ANALYSIS','purpose':PILOT_PURPOSE,
        'prospective_evidence':False,'formal_freeze_performed':False,'effect_estimation_performed':False,
        'manifest_sha256':manifest['manifest_sha256'],'planned_pairs':3,'planned_cases':6,
        'source_files_match':not mismatches,'source_mismatches':mismatches,'run':str(run),
        'cases':cases,'pair_integrity':[{k:p[k] for k in ('seed','pair_index','complete','eligible')}
                                      for p in pairs],
        'g9_complete_trace_gate':{'passed':len(valid)==6 and all(p['eligible'] for p in pairs) and not cadence,
            'expected_traces':6,'complete_valid_traces':len(valid),
            'invalid_or_missing_case_ids':[c['case_id'] for c in cases if c not in valid],
            'mask_event_cadence_issues':cadence},
        'descriptive_outcome_counts':{arm:dict(Counter(c['outcome'] for c in cases if c['comparison_arm']==arm))
                                      for arm in ARMS},
        'fidelity':fidelity,'task_time':task_time_summary(cases),'launch_cleanup':cleanup(run),'analysis_sha256':sha(Path(__file__)),
        'complete_trace_gate_sha256':sha(GATE),'new_model_generations':0,'new_env_resets':0,'new_env_steps':0,
        'limits':['Six development episodes validate engineering and trace fidelity only; success counts are descriptive.',
            'No prospective success-rate improvement, statistical significance, generalization or leaderboard claim.',
            'All six complete valid traces and paired initial conditions are required for gate G9; task failures may pass engineering checks.',
            'Pixel geometry, crop fidelity, full array hashes, GPU renderer identity and cleanup require their separate gates.',
            'This analysis neither performs a seed audit nor freezes a formal prospective experiment.']}

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
    ev=mask_event_summary(3,{'kind':MASK_EVENT,'arm':treatment(),'operation':'grasp','drawn':True,'logging_only':True,
        'used_by_policy_or_planner':False,'cameras':{CAMERAS[0]:{'target':{'mask_px':40,'ring_px':30,'inside_mask_pixels_changed':0}}}})
    counts,nonzero=fidelity_counts([ev],treatment())
    assert counts['by_operation']['grasp']['cameras'][CAMERAS[0]]['drawn_steps']==1 and not nonzero
    pt=mask_event_summary(4,{'kind':MASK_EVENT,'arm':control(),'operation':'grasp','drawn':False,'logging_only':True,
        'used_by_policy_or_planner':False,'cameras':{CAMERAS[0]:{'target':{'mask_px':40,'ring_px':0,POINT_CUE_FIELD:7}}}})
    counts,nonzero=fidelity_counts([pt],control())
    assert INSIDE_FIELD not in counts and counts[POINT_CUE_FIELD]['max']==7 and nonzero==[4]
    assert 'all_reported_zero' not in with_fractions(counts)[POINT_CUE_FIELD]
    print('CPU_SELF_TEST_PASS: exact statistics, unknown/timeout separation, all-outcome distances, fidelity reader, '
          'masks-arm-only G5')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Offline prospective point/masks analysis (never runs models or simulation).')
    parser.add_argument('--run',type=Path,help='registered run directory (contains manifest.json and worker-N/)')
    parser.add_argument('--resume',type=Path,action='append',default=[],
        help='registered resume directory with resume-plan.json; repeat in execution order')
    parser.add_argument('--output',type=Path,help='new JSON report path (never overwritten)')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--pilot-engineering',action='store_true',
        help='explicit six-case development-pilot trace/fidelity audit; no prospective effect estimation')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if not args.run or not args.output:parser.error('--run and --output required')
        if args.output.exists():raise FileExistsError('Never overwrite evidence')
        if args.pilot_engineering and args.resume:parser.error('--pilot-engineering does not accept --resume')
        report=analyze_pilot(args.run) if args.pilot_engineering else analyze(args.run,resumes=args.resume)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print(json.dumps({'status':report['status'],
                          'g9_complete_trace_gate':report['g9_complete_trace_gate']} if args.pilot_engineering
                         else {'status':report['status'],'paired':report['paired']}))
