"""Read-only OriginX retrospective analysis. No models, rollouts, source edits, or dependencies.

Run: python -X utf8 work/paper_revision_20261009/audit_analysis.py
All estimands are descriptive of a postselected cohort. Task resampling is an
exploratory stability analysis, not a confidence interval for a fixed census.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs'
STUDY = ROOT / 'work/astra_rescue_20261009'
PILOTS = {48, 97, 249, 345}
LABELS = ('rescued', 'not_rescued', 'unknown', 'initial_state_deviation')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q(values, prob):
    a = sorted(values)
    i = (len(a) - 1) * prob
    k = int(i)
    return a[k] + (a[min(k + 1, len(a) - 1)] - a[k]) * (i - k)


def descr(a):
    return dict(n=len(a), mean=statistics.mean(a), median=statistics.median(a),
                minimum=min(a), p25=q(a, .25), p75=q(a, .75), p90=q(a, .9),
                p95=q(a, .95), maximum=max(a)) if a else {'n': 0}


def table(rows):
    pairs = [c for c in rows if c['matched_complete_pair']]
    return dict(both_success=sum(c['control_success'] and c['astra_success'] for c in pairs),
                control_only=sum(c['control_success'] and not c['astra_success'] for c in pairs),
                astra_only=sum(not c['control_success'] and c['astra_success'] for c in pairs),
                both_failure=sum(not c['control_success'] and not c['astra_success'] for c in pairs))


def summarize(rows):
    counts = {k: sum(c['classification'] == k for c in rows) for k in LABELS}
    n = len(rows)
    m = sum(c['matched_complete_pair'] for c in rows)
    e = sum(c['verified_astra_matched_pair'] for c in rows)
    r = counts['rescued']
    return dict(n=n, **counts, M=m, E=e, R_over_N=r/n if n else None,
                R_over_M=r/m if m else None, R_over_E=r/e if e else None,
                paired_M=table(rows), paired_E=table([c for c in rows if c['verified_astra_matched_pair']]))


def cluster_stability(task_rows, repeats=30000, seed=20261009):
    """Resample whole tasks within recorded strata; preserve number of tasks per stratum.

    An equal-task macro rate covers the 49 tasks with at least one original
    failure (CloseFridge contributes no failure cohort). The failure-weighted
    ratio changes denominator with the resampled clusters, retaining every
    unknown and reset deviation within each selected task cluster.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for t in task_rows:
        groups[t['stratum']].append(t)
    micro, macro, valid = [], [], []
    for _ in range(repeats):
        draw = [rng.choice(g) for g in groups.values() for _ in range(len(g))]
        micro.append(sum(t['rescued'] for t in draw) / sum(t['n'] for t in draw))
        macro.append(statistics.mean(t['R_over_N'] for t in draw))
        valid.append(sum(t['rescued'] for t in draw) / sum(t['E'] for t in draw))
    return dict(seed=seed, repeats=repeats,
                method='Percentile resampling of whole tasks within original split, fixed task counts 17/16/16.',
                interpretation='Exploratory task-composition stability only; exchangeable-task superpopulation assumption is unverified. Not finite-cohort sampling uncertainty or causal confidence.',
                fixed_denominator_rate_percentile_95=[q(micro,.025),q(micro,.975)],
                equal_task_rate_percentile_95=[q(macro,.025),q(macro,.975)],
                verified_pair_rate_percentile_95=[q(valid,.025),q(valid,.975)])


def episode_dir(c, arm='astra'):
    return Path(c['source_output']) / 'episodes' / (c['case_id']+'--'+arm)


def evidence_example(c, rule):
    base = Path(c['source_output'])
    folder = base / 'requests' / (c['case_id']+'--astra')
    request, response = read(folder/'request.json'), read(folder/'response.json')
    assist = read(episode_dir(c)/'assistance.json')
    assets=[]
    for cam in request['cameras']:
        path=folder/cam['file']
        assert sha(path)==cam['sha256']
        assets.append(dict(name=cam['name'],path=str(path),sha256=sha(path)))
    outcomes={}
    traces={}
    for arm in ['control','astra']:
        ep=episode_dir(c,arm)
        r=read(ep/'result.json')
        outcomes[arm]=r['stats']['episodes'][0]
        traces[arm]=[json.loads(x) for x in (ep/'query_trace.jsonl').read_text('utf-8-sig').splitlines() if x]
    prefix={arm:[x for x in rows if x['step']<request['step']] for arm,rows in traces.items()}
    assert prefix['control']==prefix['astra']
    assert sha(folder/'request.json')==c['astra_evidence']['request_sha256']
    assert sha(folder/'response.json')==c['astra_evidence']['response_sha256']
    return dict(case_id=c['case_id'], selection_rule=rule, classification=c['classification'],
                developmental_id=c['assignment_index'] in PILOTS, image_assets=assets,
                original_instruction=request['original_instruction'], trigger_step=request['step'],
                horizon=request['horizon'], remaining_steps=request['remaining_steps'],
                subgoal=response['subgoal_instruction'], generated_rationale=response.get('rationale'),
                rationale_is_hypothesis_not_ground_truth=True,
                matched_prefix_queries=len(prefix['control']), outcomes=outcomes,
                wait_seconds=assist['wait_seconds'], request_path=str(folder/'request.json'),
                response_path=str(folder/'response.json'))


def main():
    cp=OUT/'OriginX_Astra救援复测结果.json'
    up=OUT/'OriginX_Astra救援CLI用量最终.json'
    fp=STUDY/'failure_manifest.json'
    d,u,f=read(cp),read(up),read(fp)
    cs=d['cases']
    original_by_id={r['id']:r for r in f['failure_records']}
    assert len(cs)==len({c['case_id'] for c in cs})==1004
    assert all(x['original_success'] is False for x in f['failure_records'])
    assert all(x['id'].startswith('native-B-') for x in f['jobs'])
    assert set(c['case_id'] for c in cs)==set(x['id'] for x in f['jobs'])
    assert sha(fp)==d['failure_manifest_sha256']
    for source in d['source_evidence'].values():
        base=Path(source['directory'])
        assert sha(base/'analysis/case-classification.json')==source['classification_sha256']
        assert sha(base/'config.json')==source['config_sha256']
        assert sha(base/'manifest.json')==source['manifest_sha256']
    for c in cs:
        valid=all(c['arm_status'][a]['valid_completed'] is True for a in ['control','astra'])
        m=valid and not c['initial_deviation_arms'] and c['prefix']['exact'] is True
        e=m and c['assistance_status']=='applied' and c['astra_evidence']['valid'] is True
        assert c['matched_complete_pair']==m and c['verified_astra_matched_pair']==e
        if c['classification']=='rescued':
            assert e and c['control_success'] is False and c['astra_success'] is True
        if c['classification']=='not_rescued':
            assert e and c['control_success'] is False and c['astra_success'] is False
    s=summarize(cs)
    assert (s['rescued'],s['not_rescued'],s['unknown'],s['initial_state_deviation'],s['M'],s['E'])==(56,813,28,107,874,869)
    task_rows=[]
    for task in sorted({c['task'] for c in cs}):
        rows=[c for c in cs if c['task']==task]
        task_rows.append(dict(task=task,stratum=rows[0]['stratum'],**summarize(rows)))
    tasks_sorted=sorted(task_rows,key=lambda t:(-t['rescued'],t['task']))
    no_pilots=[c for c in cs if c['assignment_index'] not in PILOTS]
    reason_counts=Counter(c['classification']+':'+c['reason'] for c in cs)
    deviation_case_fields=Counter()
    deviation_arm_fields=Counter()
    wait_e,wait_all=[],[]
    applied_statuses=Counter()
    outcome_wall=defaultdict(list)
    prefix_total=0
    e_receipts=set()
    for c in cs:
        a=read(episode_dir(c)/'assistance.json')
        applied_statuses[a.get('status')]+=1
        if isinstance(a.get('wait_seconds'),(int,float)):
            wait_all.append(a['wait_seconds'])
            if c['verified_astra_matched_pair']:wait_e.append(a['wait_seconds'])
        fields=set()
        for arm in c['initial_deviation_arms']:
            initial=read(episode_dir(c,arm)/'initial_replay.json')
            fields.update(initial['differences'])
            deviation_arm_fields.update(initial['differences'].keys())
        deviation_case_fields.update(fields)
        if c['verified_astra_matched_pair']:
            request_dir=Path(c['source_output'])/'requests'/(c['case_id']+'--astra')
            for file,key in [('request.json','request_sha256'),('response.json','response_sha256'),('cli_receipt.json','cli_receipt_sha256')]:
                assert sha(request_dir/file)==c['astra_evidence'][key]
            request=read(request_dir/'request.json')
            for image in request['cameras']:
                assert sha(request_dir/image['file'])==image['sha256']
            assert request['oracle_inputs_included'] is False
            assert request['step']==16*math.ceil(request['horizon']/32)
            e_receipts.add(c['astra_evidence']['cli_receipt_sha256'])
            traces={}
            for arm in ['control','astra']:
                ep=episode_dir(c,arm)
                traces[arm]=[json.loads(x) for x in (ep/'query_trace.jsonl').read_text('utf-8-sig').splitlines() if x]
                initial=read(ep/'initial_replay.json')
                assert initial['exact_match'] is True
                result=read(ep/'result.json')
                assert read(ep/'initial_scene.json')==result['initial_scene']==original_by_id[c['case_id']]['original_initial_scene']
                assert result['success']==c[arm+'_success']
                assert [x['step'] for x in traces[arm]]==list(range(0,result['stats']['episodes'][0]['steps'],16))
                traces[arm]=[x for x in traces[arm] if x['step']<request['step']]
                outcome_wall[arm].append(result['wall_seconds'])
            assert traces['control']==traces['astra'] and traces['control']
            prefix_total+=len(traces['control'])
    calls=u['primary']['calls']
    assert len(calls)==len({x['cli_receipt_sha256'] for x in calls})==135
    totals={k:sum(x['usage'][k] or 0 for x in calls) for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_tokens']}
    assert totals['input_tokens']==2592713 and totals['output_tokens']==195237 and totals['reasoning_tokens']==75511
    receipt_ids={x['cli_receipt_sha256'] for x in calls}
    assert e_receipts<=receipt_ids
    zero_original=['GatherTableware','HeatKebabSandwich','PanTransfer']
    hard5=zero_original+['CategorizeCondiments','SeparateFreezerRack']
    positive=next(c for c in cs if c['assignment_index']==186)
    same_task_negative=next(c for c in cs if c['assignment_index']==586)
    negative=min((c for c in cs if c['classification']=='not_rescued' and c['assignment_index'] not in PILOTS),key=lambda c:c['assignment_index'])
    exact_p=2*sum(math.comb(100,k) for k in range(48))/(2**100)
    report=dict(schema='originx_retrospective_paper_evidence_audit_v1',
        sources={str(p.relative_to(ROOT)):sha(p) for p in [cp,up,fp]},
        no_source_modifications=True,no_rollouts=True,no_model_calls=True,
        verified_checks=dict(fixed_B_failure_cohort=True,unique_cases=1004,source_hashes=True,
            strict_case_flags=True,verified_E_request_response_receipt_and_image_hashes=869,
            paired_exact_prefixes_recomputed=869,paired_prefix_query_count=prefix_total,
            original_initial_fingerprint_records_recompared=1738,full_query_step_sequences_checked=1738,
            result_success_bits_checked=1738,actual_receipt_dedup=True),
        summary=s,
        strict_definition='Original B failure; both arms valid completed; both native-reset initial fingerprints match original; exact complete query prefix before trigger; applied assistance with linked request/images/response/CLI evidence; control failure and assisted success.',
        M_definition='Both arms valid completed, no initial-state deviation, exact preintervention prefix; may include unapplied assistance.',
        E_definition='M plus assistance_status=applied and verified Astra evidence.',
        by_stratum={k:summarize([c for c in cs if c['stratum']==k]) for k in sorted(d['by_stratum'])},
        task_equal=dict(tasks_with_original_failures=49,tasks_with_verified_pairs=48,
            rate_fixed_failure_cohort=statistics.mean(t['R_over_N'] for t in task_rows),
            rate_conditional_verified=statistics.mean(t['R_over_E'] for t in task_rows if t['E']),
            note='49-task macro conditions on having at least one original failure; CloseFridge has zero failures and no defined rescue rate. 48-task E macro additionally omits OpenStandMixerHead with no valid pairs; not directly comparable.'),
        task_concentration=dict(tasks_with_rescue=sum(t['rescued']>0 for t in task_rows),
            top_two_rescues=sum(t['rescued'] for t in tasks_sorted[:2]),
            top_five_rescues=sum(t['rescued'] for t in tasks_sorted[:5]),
            top_five_tasks=[t['task'] for t in tasks_sorted[:5]],
            original_zero_success_tasks=zero_original,
            original_zero_success_summary=summarize([c for c in cs if c['task'] in zero_original]),
            original_bottom_five_tasks=hard5,
            original_bottom_five_summary=summarize([c for c in cs if c['task'] in hard5])),
        pilot_sensitivity=dict(development_assignment_indices=sorted(PILOTS),
            primary_development_rows=[dict(case_id=c['case_id'],classification=c['classification']) for c in cs if c['assignment_index'] in PILOTS],
            excluded_development_ids=summarize(no_pilots),
            note='Development execution outcomes 0/4 were not merged. The same four IDs remained in main study with new primary execution outcomes 1/4. Excluding identities gives 55/1000, not 56/1000.'),
        missingness=dict(reasons=dict(reason_counts),deviation_arms=dict(Counter('+'.join(c['initial_deviation_arms']) for c in cs if c['classification']=='initial_state_deviation')),
            deviation_fields_case_counts=dict(deviation_case_fields),deviation_fields_arm_counts=dict(deviation_arm_fields),
            overlapping_fields=True,
            technical_unknown_assistance=dict(Counter(c['assistance_status'] for c in cs if c['classification']=='unknown')),
            note='23 invalid/infrastructure pairs (22 timeout, one applied intervention with invalid outcome); 5 complete pairs with broker_error. Deviation category precedes all outcome classification; not a policy failure.'),
        assistance_accounting=dict(all_statuses=dict(applied_statuses),verified_E=869,receipts_with_verified_E=len(e_receipts),
            wait_verified_E_seconds=descr(wait_e),wait_all_records_seconds=descr(wait_all),
            wait_note='Observed request-to-response waiting includes batching, queue and transport; waiting pauses simulation. It is distinct from per-batch CLI latency and not real-time robot performance.',
            paired_wall_seconds={k:descr(v) for k,v in outcome_wall.items()}),
        CLI=dict(calls=len(calls),statuses=dict(Counter(x['status'] for x in calls)),known_tokens=totals,
            incomplete_totals=True,missing_usage_calls=sum(x['usage']['input_tokens'] is None for x in calls),
            latency_seconds=descr([x['latency_seconds'] for x in calls]),
            latency_sum=sum(x['latency_seconds'] for x in calls),
            cost_usd=None,reasoning_in_output_not_additive=True,
            mean_batch_requests=statistics.mean(len(x['request_ids']) for x in calls),
            total_request_references=sum(len(x['request_ids']) for x in calls)),
        task_resampling=cluster_stability(task_rows),
        hypothesis_tests=dict(earlier_600_pair_McNemar_exact_two_sided=exact_p,
            rescue_McNemar_recommended=False,
            reason='Controls are selected original failures under a fixed deterministic replay. Thus absence of control success and reverse discordance is expected by design. A McNemar calculation or binomial test would not establish an unbiased full-population effect. No individual-episode iid interval is promoted.'),
        examples=[evidence_example(positive,'One member of the complete two-case OpenDrawer failure subset; not chosen as typical. Excludes pilot IDs.'),
                  evidence_example(same_task_negative,'The only other member of the complete two-case OpenDrawer failure subset. This task was selected illustratively, not claimed representative.'),
                  evidence_example(negative,'Smallest original assignment index among strict non-rescues excluding four development IDs.')],
        by_task=task_rows)
    (HERE/'evidence_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# OriginX retrospective evidence audit','',
        'Read-only analysis of the frozen study. No training, rollout, source mutation, new model call, or outcome selection for aggregate statistics.','',
        '## Verified primary result','',
        '1,004 unique original B failures: 56 strict rescues, 813 strict non-rescues, 28 technical unknowns, 107 reset deviations. M=874 matched complete pairs; E=869 with verified applied assistance. All 869 linked request/response/CLI receipts, RGB hashes, initial-state equality flags, preintervention query traces and 1,738 result success bits were independently checked.','',
        'M paired table: both success 0; control only 0; assistance only 56; both fail 818. E table: 0, 0, 56, 813. The five extra complete pairs have broker errors, not verified assistance.','',
        f"Failure-weighted R/N={s['R_over_N']:.6%}; R/E={s['R_over_E']:.6%}. Equal-task macro over 49 tasks with original failures={report['task_equal']['rate_fixed_failure_cohort']:.6%}. Conditional E macro over 48 tasks={report['task_equal']['rate_conditional_verified']:.6%}; its population differs.",'',
        '## Findings that sharpen the paper','',
        'Rescues occur in 27/49 failure-containing tasks, so they are not confined to one example. BreadSelection and PrepareCoffee account for 13/56 (23.21%). The first five by rescue count (ties alphabetically) account for 24/56 (42.86%); this ranking is descriptive, not a selected test set.',
        'The three tasks with 0/50 original successes yield only one strict rescue among 150 original failures (1/135 verified pairs). The five worst original tasks yield one rescue among 237 original failures (1/219 verified pairs). Text assistance therefore does not resolve the dominant difficult-task bottleneck in these observations.',
        'Atomic-Seen 12/153 (7.84%), Composite-Seen 25/321 (7.79%), Composite-Unseen 19/530 (3.58%). Conditional verified-pair rates are 12/137 (8.76%), 25/243 (10.29%), 19/489 (3.89%). Different reset validity and task mixtures prevent attributing these differences to task novelty alone.','',
        '## Development exposure sensitivity','',
        'The pilot executions were 0/4 and excluded as outcomes, but their IDs remained in the main cohort. Primary index97 is rescued; indices48/249/345 are not. Excluding all four identities gives 55 rescues / 1,000 cases (5.50%), M870, E865, 810 strict non-rescues, 28 unknowns, 107 deviations. This distinction must be explicit.','',
        '## Infrastructure and timing','',
        'Unknowns are 23 invalid/infrastructure pairs (22 timeout, one applied but invalid outcome) plus five completed pairs with broker error. Reset deviations affect 107 cases: control only40, Astra only27, both40. Their overlapping field counts appear in JSON; these are not all merely renderer differences.',
        f"For E869 actual waiting is median {statistics.median(wait_e):.2f}s, IQR {q(wait_e,.25):.2f}–{q(wait_e,.75):.2f}s, range {min(wait_e):.2f}–{max(wait_e):.2f}s. Median per-CLI batch latency {statistics.median([x['latency_seconds'] for x in calls]):.2f}s excludes most queue time. The method pauses simulation and is not demonstrated real-time assistance.",
        'Primary CLI calls135=134 successful+one quota error. Known tokens input2,592,713/output195,237, including reasoning75,511. Error-call counters and dollar cost unknown. Reasoning must not be added to output.','',
        '## Statistical interpretation','',
        'The recorded 1,004-case cohort is a census, so its realized 5.58% is exact; the missing/deviation records remain distinct. No iid binomial confidence interval or small paired p-value can establish full-population causal improvement for a cohort selected as original failures. Controls all failing is expected under faithful deterministic replay. The study does not measure regressions on the original1,496 successes.',
        'Optional split-stratified whole-task bootstrap sensitivity is in JSON (30,000 replicates, seed20261009). It describes sensitivity to task composition under an unverified exchangeable-task assumption, not finite-cohort sampling uncertainty. Use only clearly labeled exploratory stability ranges, if at all.',
        f"Earlier 600-pair McNemar exact two-sided p recomputes to {exact_p:.10f} from 53/47 discordances. Its reported [-2.17,4.17]pp interval is not independently verifiable without the raw pairs and bootstrap method; an ordinary iid paired Wald interval differs and should not silently replace it.",'',
        '## Missing controls / inferences not supported','',
        '- Same-budget generic instruction restatement, deterministic subgoal prompt, nonvisual language, and alternative model controls are absent; Astra-specific reasoning is not isolated.',
        '- No full-cohort assisted evaluation includes original successes, so overall gains and regressions are unknown.',
        '- No matched Xiaomi/A-only/B/zero-branch/stage/retention ablations on final manifest; branch gains are unproven.',
        '- One observed RGB triplet and an assistant rationale do not establish a physical failure cause. Generated rationales are hypotheses, not annotations.',
        '- Initial-state exactness and prefix checks support controlled within-case contrasts, but not equivalent initial conditions for the107 deviations.',
        '- No new benchmark score, significance versus leaderboard models, cross-robot transfer, or conference-acceptance guarantee follows.','',
        '## Examples and figure assets','',
        'Exact source PNG paths, hashes, response text, step budgets and matched outcome records are in evidence_audit.json for the complete two-case OpenDrawer failure subset (186 rescued; 586 not rescued), plus the deterministic earliest nonpilot negative example PickPlaceDrawerToCounter38. OpenDrawer is an illustrative task, not asserted representative. Source images were not altered or generated.','',
        '## Per-task census','',
        '| Task | Split | N | R | Not rescued | Unknown | Deviation | M | E |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in task_rows:
        lines.append('| {task} | {stratum} | {n} | {rescued} | {not_rescued} | {unknown} | {initial_state_deviation} | {M} | {E} |'.format(**t))
    (HERE/'evidence_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['summary','task_equal','task_concentration','pilot_sensitivity','task_resampling']},indent=2))


if __name__=='__main__':
    main()
