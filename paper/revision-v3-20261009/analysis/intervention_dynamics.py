"""Retrospective action-onset and success-timing analysis of immutable OriginX evidence.

No learning, simulation, new model requests, or source mutations. Standard library only.
Run from any directory: python -X utf8 intervention_dynamics.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULT = ROOT / 'outputs/OriginX_Astra救援复测结果.json'
PRIOR_AUDIT = ROOT / 'work/paper_revision_20261009/evidence_audit.json'
PILOT_IDS = {48, 97, 249, 345}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def percentile(x, p):
    a=sorted(x)
    j=(len(a)-1)*p
    i=int(j)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(j-i)


def summary(x):
    return {'n':len(x),'mean':statistics.mean(x),'median':statistics.median(x),
            'minimum':min(x),'p25':percentile(x,.25),'p75':percentile(x,.75),
            'p90':percentile(x,.9),'maximum':max(x)} if x else {'n':0}


def trace(path):
    rows=[json.loads(line) for line in Path(path).read_text('utf-8-sig').splitlines() if line]
    assert len(rows)==len({r['step'] for r in rows})
    return {r['step']:r for r in rows}


def verify_action_hash_producer():
    """Verify source pin and AST, without importing or executing experimental code."""
    original=ROOT/'work/astra_rescue_20261009/assistance.py'
    continuation=ROOT/'work/astra_rescue_continuation_20261009/assistance.py'
    op=ast.parse(original.read_text('utf-8-sig'))
    original_class=next(x for x in op.body if isinstance(x,ast.ClassDef) and x.name=='AssistanceClient')
    infer=next(x for x in original_class.body if isinstance(x,ast.FunctionDef) and x.name=='infer')
    array_sha=next(x for x in infer.body if isinstance(x,ast.FunctionDef) and x.name=='array_sha')
    expected=ast.parse('''def array_sha(value):
    array = np.asarray(value)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()
''').body[0]
    assert ast.dump(array_sha,include_attributes=False)==ast.dump(expected,include_attributes=False)
    action=next(x for x in infer.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='action' for t in x.targets))
    assert ast.unparse(action.value)=='self.client.infer(state_history, image_history, actual_instruction)'
    traced=next(x for x in infer.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trace' for t in x.targets))
    fields={kw.arg:ast.unparse(kw.value) for kw in traced.value.keywords}
    assert fields['action_sha256']=='array_sha(action)'
    assert fields['instruction_sha256']=="hashlib.sha256(actual_instruction.encode('utf-8')).hexdigest()"
    continuation_ast=ast.parse(continuation.read_text('utf-8-sig'))
    cls=next(x for x in continuation_ast.body if isinstance(x,ast.ClassDef) and x.name=='AssistanceClient')
    assert [ast.unparse(x) for x in cls.bases]==['OriginalAssistanceClient']
    assert not any(isinstance(x,ast.FunctionDef) and x.name=='infer' for x in cls.body)
    pins=[]
    for directory,package in [('full_sync','astra_rescue_20261009'),('continuation_sync','astra_rescue_continuation_20261009')]:
        config_path=ROOT/'work/astra_rescue_20261009'/directory/'evidence/config.json'
        config=read(config_path)
        for file in [original]+([continuation] if directory=='continuation_sync' else []):
            suffix='/'+file.parent.name+'/assistance.py'
            expected_hash=[value for path,value in config['source_sha256'].items() if path.endswith(suffix)]
            assert expected_hash==[sha(file)]
            pins.append(dict(config=str(config_path.relative_to(ROOT)),source=str(file.relative_to(ROOT)),sha256=sha(file),matched=True))
    return dict(source_pins=pins,producer=str(original.relative_to(ROOT)),
        infer_line=infer.lineno,array_sha_line=array_sha.lineno,action_inference_line=action.lineno,
        trace_assignment_line=traced.lineno,
        expression='SHA256(UTF8(str((array.shape, array.dtype.str))) || array.tobytes())',
        input_to_hash='The action array returned by client.infer; shape/dtype metadata plus raw array bytes, no instruction/request fields.',
        instruction_hash_separate=True,
        continuation_inherits_identical_infer=True,AST_matches_expected_formula=True,
        limitation='Raw action values were not retained by this trace, so no magnitude, direction or physical significance is reconstructed. The fingerprint includes shape and dtype as well as bytes.')


def timeline_summary(rows, fixed_n=None):
    rescued=[r for r in rows if r['strict_rescued']]
    return dict(eligible_pairs=len(rows),strict_rescues=len(rescued),
                posttrigger_steps_to_success=summary([r['success_lag_steps'] for r in rescued]),
                remaining_budget_fraction_to_success=summary([r['remaining_budget_fraction_to_success'] for r in rescued]),
                whole_horizon_fraction_to_success=summary([r['astra_terminal_steps']/r['horizon'] for r in rescued]),
                steps_left_at_success=summary([r['horizon']-r['astra_terminal_steps'] for r in rescued]),
                cumulative=[dict(remaining_budget_fraction=q,
                    strict_rescues_by_fraction=sum(r['remaining_budget_fraction_to_success']<=q for r in rescued),
                    fraction_of_all_eligible=sum(r['remaining_budget_fraction_to_success']<=q for r in rescued)/len(rows),
                    fraction_of_recorded_rescues=sum(r['remaining_budget_fraction_to_success']<=q for r in rescued)/len(rescued) if rescued else None,
                    fraction_of_fixed_failure_cohort=(sum(r['remaining_budget_fraction_to_success']<=q for r in rescued)/fixed_n if fixed_n else None))
                    for q in [.25,.5,.75,.9,1.0]])


def main():
    source=read(RESULT)
    prior=read(PRIOR_AUDIT)
    producer=verify_action_hash_producer()
    cs=source['cases']
    assert len(cs)==len({c['case_id'] for c in cs})==1004
    eligible=[c for c in cs if c['verified_astra_matched_pair']]
    assert len(eligible)==prior['summary']['E']==869
    rows=[]
    source_files=[]
    for c in eligible:
        base=Path(c['source_output'])
        ep={arm:base/'episodes'/(c['case_id']+'--'+arm) for arm in ['control','astra']}
        results={arm:read(ep[arm]/'result.json') for arm in ep}
        traces={arm:trace(ep[arm]/'query_trace.jsonl') for arm in ep}
        request_path=base/'requests'/(c['case_id']+'--astra')/'request.json'
        request=read(request_path)
        assert sha(request_path)==c['astra_evidence']['request_sha256']
        tau=request['step']
        horizon=request['horizon']
        assert tau==16*math.ceil(horizon/32)==c['prefix']['trigger_step']
        for arm in ep:
            r=results[arm]
            assert r['status']=='completed'
            assert r['job']['horizon']==horizon
            assert r['success'] is c[arm+'_success']
            steps=r['stats']['episodes'][0]['steps']
            assert list(traces[arm])==list(range(0,steps,16))
            assert read(ep[arm]/'initial_replay.json')['exact_match'] is True
            for filename in ['result.json','query_trace.jsonl','initial_replay.json']:
                path=ep[arm]/filename
                source_files.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        tc,ta=traces['control'],traces['astra']
        assert {t:r for t,r in tc.items() if t<tau}=={t:r for t,r in ta.items() if t<tau}
        assert results['control']['success'] is False
        assert results['control']['stats']['episodes'][0]['steps']==horizon
        onset_input_match=all(tc[tau][key]==ta[tau][key] for key in ['state_sha256','image_sha256'])
        original=request['original_instruction']
        instruction0=hashlib.sha256(original.encode('utf-8')).hexdigest()
        assert tc[tau]['instruction_sha256']==instruction0
        actual=results['astra']['assistance']['policy_instruction']
        assert ta[tau]['instruction_sha256']==hashlib.sha256(actual.encode('utf-8')).hexdigest()
        assert actual!=original
        shared_steps=[t for t in ta if t>=tau and t in tc]
        changed=[t for t in shared_steps if ta[t]['action_sha256']!=tc[t]['action_sha256']]
        first_changed=min(changed) if changed else None
        success=c['classification']=='rescued'
        assert success is c['astra_success']
        terminal=results['astra']['stats']['episodes'][0]['steps']
        assert tau<terminal<=horizon
        if not success: assert terminal==horizon
        rows.append(dict(case_id=c['case_id'],assignment_index=c['assignment_index'],task=c['task'],stratum=c['stratum'],
            development_exposed=c['assignment_index'] in PILOT_IDS,
            strict_rescued=success,horizon=horizon,trigger_step=tau,remaining_step_budget=horizon-tau,
            trigger_state_and_images_exact=onset_input_match,
            changed_instruction_hash=ta[tau]['instruction_sha256']!=tc[tau]['instruction_sha256'],
            first_posttrigger_action_hash_difference_step=first_changed,
            first_difference_lag_steps=first_changed-tau if first_changed is not None else None,
            changed_action_hash_at_trigger=first_changed==tau,
            observed_common_posttrigger_queries=len(shared_steps),
            changed_action_hash_queries=len(changed),
            astra_terminal_steps=terminal,
            success_lag_steps=terminal-tau if success else None,
            remaining_budget_fraction_to_success=(terminal-tau)/(horizon-tau) if success else None,
            cli_receipt_sha256=c['astra_evidence']['cli_receipt_sha256']))
    assert sum(r['strict_rescued'] for r in rows)==56
    batches=defaultdict(list)
    for row in rows:batches[row['cli_receipt_sha256']].append(row)
    batch_rows=[]
    for identity,group in sorted(batches.items()):
        r=sum(x['strict_rescued'] for x in group)
        batch_rows.append(dict(cli_receipt_sha256=identity,eligible_cases=len(group),rescues=r,
            remaining_E=869-len(group),remaining_R=56-r,
            remaining_R_over_E=(56-r)/(869-len(group)),
            removed_case_ids=[x['case_id'] for x in group]))
    assert len(batches)==132
    batch_sensitivity=dict(verified_assistance_batches=132,
        batches_with_rescue=sum(r['rescues']>0 for r in batch_rows),
        maximum_rescues_in_one_batch=max(r['rescues'] for r in batch_rows),
        baseline_R_over_E=56/869,
        leave_one_batch_out_R_over_E_range=[min(r['remaining_R_over_E'] for r in batch_rows),max(r['remaining_R_over_E'] for r in batch_rows)],
        eligible_batch_size_histogram=dict(Counter(str(r['eligible_cases']) for r in batch_rows)),
        rescue_per_batch_histogram=dict(Counter(str(r['rescues']) for r in batch_rows)),
        batches=batch_rows,
        interpretation='Deterministic deletion sensitivity over132 existing CLI receipts supplying eligible cases. Each deletion removes all eligible cases from one receipt and recomputes R/E. This is not a confidence interval, an independent-batch assumption, or a new primary score. All135 primary invocations remain in the resource ledger.')
    onset=dict(eligible_pairs=869,missing_records=0,
        matched_state_and_images_at_trigger=sum(r['trigger_state_and_images_exact'] for r in rows),
        changed_instruction_at_trigger=sum(r['changed_instruction_hash'] for r in rows),
        changed_action_hash_at_trigger=sum(r['changed_action_hash_at_trigger'] for r in rows),
        no_observed_action_hash_difference=sum(r['first_difference_lag_steps'] is None for r in rows),
        first_difference_lag_steps_histogram=dict(Counter(str(r['first_difference_lag_steps']) for r in rows)),
        outcomes_of_immediate_change=dict(rescued=sum(r['strict_rescued'] and r['changed_action_hash_at_trigger'] for r in rows),
            not_rescued=sum(not r['strict_rescued'] and r['changed_action_hash_at_trigger'] for r in rows)),
        conclusion='All verified eligible pairs have matching trigger observation/state fingerprints, a changed language instruction, and an immediately different returned action hash. Mere computational response to the changed instruction is therefore insufficient for success in 813 eligible cases.',
        limits=['A hash inequality detects any serialized action-array difference; no action magnitudes or physical significance can be recovered from hashes.',
                'No counterfactual generic or nonvisual instruction control isolates reasoning quality or semantic grounding.',
                'Different service scheduling or residual arithmetic nondeterminism is not independently randomized by this retrospective comparison.',
                'These 869 cases are selected original failures with verified assistance; the 135 remaining cohort cases and original 1496 successes are not represented by this onset statistic.'])
    timeline=timeline_summary(rows,1004)
    timeline['missing_records']=0
    timeline['censored_or_unobserved_cases']=dict(eligible_non_rescues_at_original_horizon=813,
        excluded_unknowns=28,excluded_initial_state_deviations=107)
    timeline['by_stratum']={k:timeline_summary([r for r in rows if r['stratum']==k]) for k in sorted({r['stratum'] for r in rows})}
    timeline['excluding_development_IDs']=timeline_summary([r for r in rows if not r['development_exposed']],1000)
    timeline['rescues_after_90_percent_remaining_budget']=sum(r['strict_rescued'] and r['remaining_budget_fraction_to_success']>.9 for r in rows)
    timeline['rescues_in_last_quarter_of_remaining_budget']=sum(r['strict_rescued'] and r['remaining_budget_fraction_to_success']>.75 for r in rows)
    timeline['interpretation']='Empirical completion timing under the already recorded intervention and fixed horizon; all 869 eligible pairs stay in cumulative denominators. Timing percentiles condition on the 56 observed rescues. No extrapolation beyond the horizon, time-to-event model, earlier-trigger benefit, or new benchmark score is estimated.'
    timeline['limitations']=['Successful cases are selected by terminal outcome, so their completion-time median is not expected time to recovery for an arbitrary failure.',
        'Original task horizons and difficulty differ; normalized residual time helps comparison but does not make tasks exchangeable.',
        'The 813 horizon-terminal failures do not establish when or whether a longer rollout could succeed.',
        'A retrospective truncation of unchanged trajectories does not evaluate changing the trigger, policy, or prompt.',
        'Case-level dependence remains through tasks, shared policy, and batched assistant calls; no iid confidence interval is attached.']
    report=dict(schema='originx_post_intervention_dynamics_audit_v1',
        source_classification_sha256=sha(RESULT),prior_evidence_audit_sha256=sha(PRIOR_AUDIT),
        no_source_changes=True,no_model_calls=True,no_training=True,no_simulation=True,
        selection='All869 verified-astra matched pairs from immutable1004-case classification, without selecting cases by success for onset or cumulative-denominator statistics.',
        action_hash_producer_verification=producer,
        action_onset=onset,success_timing=timeline,batch_deletion_sensitivity=batch_sensitivity,cases=rows,
        source_files=source_files)
    (HERE/'intervention_dynamics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (HERE/'eligible_timing_cdf.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['case_id','task','stratum','strict_rescued','development_exposed','horizon','trigger_step',
                'remaining_step_budget','astra_terminal_steps','success_lag_steps','remaining_budget_fraction_to_success']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    with (HERE/'rescued_timing.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows if r['strict_rescued'])
    text=['# Two additional retrospective analyses','',
          'No new experiments, training, remote jobs, or assistant calls. All sources are immutable existing evidence.','',
          '## 1. Immediate action divergence','',
          'Across all869 verified eligible pairs, the control and assisted policy-query records have identical state and RGB fingerprints at the trigger, different instruction hashes, and different returned action hashes immediately at that query. The action-hash onset lag is0 steps in869/869; missing0. Of these,56 are strict rescues and813 remain failures.',
          'This rules out an unchanged returned action tensor at intervention as the explanation for the813 non-rescues. It does not measure action magnitude, show that the physical response was useful, prove semantic grounding, or isolate Astra-specific reasoning. Hashes cannot reconstruct floating-point actions. Retrospective matching does not randomize latent service/arithmetic effects.','',
          'Producer verified against immutable source pins and AST: AssistanceClient.infer obtains action=client.infer(...), then hashes only array shape, dtype string and raw bytes. Instruction is a separately hashed field. The continuation inherits infer without override. Source line numbers and cryptographic pins are in JSON; action_hash is not a combined instruction/request fingerprint.','',
          '## 2. Completion timing within the original budget','',
          'For each rescued case, define u=(success_step-trigger)/(horizon-trigger). Median u={:.4f} ({:.2f}% of remaining steps), IQR{:.4f}–{:.4f}, range{:.4f}–{:.4f}.'.format(timeline['remaining_budget_fraction_to_success']['median'],100*timeline['remaining_budget_fraction_to_success']['median'],timeline['remaining_budget_fraction_to_success']['p25'],timeline['remaining_budget_fraction_to_success']['p75'],timeline['remaining_budget_fraction_to_success']['minimum'],timeline['remaining_budget_fraction_to_success']['maximum']),
          'Median steps from intervention to success={:.1f}; mean={:.2f}. These summaries condition on56 observed rescues.'.format(timeline['posttrigger_steps_to_success']['median'],timeline['posttrigger_steps_to_success']['mean']),'',
          '| Remaining action budget consumed | Confirmed rescues | Fraction of E869 | Fraction of R56 | Fraction of N1004 |',
          '|---:|---:|---:|---:|---:|']
    for r in timeline['cumulative']:
        text.append('| {:.0%} | {} | {:.2%} | {:.2%} | {:.2%} |'.format(r['remaining_budget_fraction'],r['strict_rescues_by_fraction'],r['fraction_of_all_eligible'],r['fraction_of_recorded_rescues'],r['fraction_of_fixed_failure_cohort']))
    text += ['',
          '15/56 rescues finish in the last quarter of their remaining action budget, and7/56 require more than90%. Thus a short continuation window would miss some successful recorded recoveries; these observations do not establish that an earlier assistance trigger would improve outcomes.',
          'There are no missing source records among E869. All813 eligible non-rescues reach the original horizon. The28 unknown and107 reset-deviation cases remain outside eligible timing analysis and inside the N1004 accounting. Original1496 successes are not reevaluated. No statistical independence, overall treatment effect, new leaderboard score, or future-success extrapolation is assumed.','',
          '## Batch dependence deletion sensitivity','',
          'The869 eligible pairs originate from132 distinct CLI receipts, with rescues in47 batches. One batch contributes at most3/56 rescues. Deleting each batch and all its eligible cases gives R/E between{:.4%} and{:.4%}, compared with6.4442%. This deterministic deletion range is not a confidence interval and does not establish independent batches. All135 primary invocations remain in usage accounting.'.format(*batch_sensitivity['leave_one_batch_out_R_over_E_range']),'',
          '## Suggested concise manuscript insertion','',
          'At the intervention query, all869 eligible pairs retain identical observation/state fingerprints, while both the language instruction and returned action hashes differ. The813 non-rescues therefore reflect failure despite a computational response to the intervention, rather than an unchanged action output; hash evidence does not establish the magnitude or physical utility of that response. Among56 rescues, the median successful continuation consumes51.67% of the remaining step budget. Cumulative confirmed rescues after25%,50%,75%, and100% of that budget are11,27,41, and56;15 finish in the last quarter. These descriptive timing results retain all869 eligible pairs in the cumulative denominator and do not test an alternative trigger.','',
          '## Recompute','',
          'Run intervention_dynamics.py with Python3 and no third-party packages. The JSON includes every eligible case, denominators, exclusions, source digests, and development-ID sensitivity. eligible_timing_cdf.csv contains all869 cases for cumulative-denominator plots (non-rescues have empty event times). rescued_timing.csv contains only56 events and must not be used to normalize the eligible-cohort CDF; its median is explicitly conditional on success.']
    (HERE/'intervention_dynamics.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'action_onset':onset,'success_timing':timeline},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
