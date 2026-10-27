"""Analyze the registered 24 cases split across the interrupted run and its resume.

Each case is read from exactly one directory: the prior run for the cases it
completed, the resume run for the cases listed in its resume plan. Pair, arm and
statistic logic is the frozen analyze_regions_validation.py, imported unchanged.
"""
from pathlib import Path
import argparse
import ast
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = HERE / 'analyze_regions_validation.py'
spec = importlib.util.spec_from_file_location('regions_validation_frozen', FROZEN)
frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)


def local(path):
    """Map a WSL /mnt/<drive>/ path recorded by the resume launcher to this host."""
    text = str(path)
    if text.startswith('/mnt/') and len(text) > 6 and Path('C:/').exists():
        return Path(text[5].upper() + ':/' + text[7:])
    return Path(text)


def analyze(prior, resume, root=ROOT):
    plan = frozen.load(resume / 'resume-plan.json')
    if local(plan['prior_output']).resolve() != prior.resolve():
        raise ValueError('Resume plan names a different prior run')
    manifest = frozen.load(prior / 'manifest.json')
    if frozen.load(resume / 'manifest.json') != manifest:
        raise ValueError('Prior and resume manifests differ')
    owed = set(plan['owed_case_ids'])
    for cid, digest in plan['prior_completed_result_sha256'].items():
        case = next(c for c in manifest['cases'] if c['case_id'] == cid)
        path = prior / f"worker-{case['worker']}" / 'run/results' / f'{cid}.json'
        if frozen.sha(path) != digest:
            raise ValueError('Prior result changed after resume registration: ' + cid)
    if owed & set(plan['prior_completed_result_sha256']) or len(owed) + len(plan['prior_completed_result_sha256']) != 24:
        raise ValueError('Resume plan does not partition the 24 registered cases')
    mismatches = frozen.manifest_check(manifest, root)
    global_issues = ['registered_source_mismatch:' + p for p in mismatches]
    cases = []
    for case in manifest['cases']:
        source = resume if case['case_id'] in owed else prior
        row = frozen.analyze_case(source, case, manifest, global_issues)
        row['source_run'] = 'resume' if source is resume else 'prior'
        row['reattempt_after_zero_step_kill'] = case['case_id'] in plan['prior_interrupted_zero_step']
        cases.append(row)
    pairs = frozen.make_pairs(cases, manifest['seeds']); arms, paired = frozen.summaries(cases, pairs)
    # Registered-rule sensitivity: the re-attempted zero-step case stays UNKNOWN, as the
    # registered driver would have recorded it, so its pair cannot enter paired inference.
    registered = [{**c, 'outcome': 'unknown', 'category': 'registered_rule_interrupted_trace_unknown'}
                  if c['reattempt_after_zero_step_kill'] else c for c in cases]
    registered_pairs = frozen.make_pairs(registered, manifest['seeds'])
    registered_arms, registered_paired = frozen.summaries(registered, registered_pairs)
    receipt = frozen.load(resume / 'launch-receipt.json') if (resume / 'launch-receipt.json').is_file() else None
    return {'status': 'PROSPECTIVE_OFFLINE_ANALYSIS', 'purpose': frozen.PURPOSE, 'manifest_sha256': manifest['manifest_sha256'],
        'source_files_match': not mismatches, 'source_mismatches': mismatches,
        'run': str(prior), 'resume_run': str(resume), 'resume_plan': plan, 'cases': cases, 'pairs': pairs,
        'arms': arms, 'paired': paired,
        'registered_rule_sensitivity': {'reattempted_case_ids': sorted(plan['prior_interrupted_zero_step']),
            'arms': registered_arms, 'paired': registered_paired,
            'pairs': [{k: p[k] for k in ('seed', 'eligible', 'point_outcome', 'regions_outcome', 'success_delta_regions_minus_point')} for p in registered_pairs]},
        'launch_cleanup': None if not receipt else {k: receipt.get(k) for k in ('status', 'owned_processes_stopped', 'cleanup_errors', 'elapsed_seconds')},
        'prior_launch_cleanup': 'Prior launcher was killed with WSL at 13:16 CDT; see interruption-record-20261024.json.',
        'analysis_sha256': frozen.sha(FROZEN), 'combined_wrapper_sha256': frozen.sha(Path(__file__)),
        'complete_trace_gate_sha256': frozen.sha(frozen.GATE),
        'new_model_generations': 0, 'new_env_resets': 0, 'new_env_steps': 0,
        'limits': [
            'One registered case (seed 20261034, regions, first arm of pair 3) was killed by an infrastructure shutdown before its first environment step and was re-attempted once in the resume run; no outcome had been observed.',
            *frozen_limits()]}


def frozen_limits():
    # The frozen analyzer embeds its limit statements inside analyze(); recover them verbatim.
    source = FROZEN.read_text(encoding='utf-8')
    start = source.index("'limits':[") + len("'limits':")
    end = source.index(']}', start) + 1
    return ast.literal_eval(source[start:end])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite evidence')
    report = analyze(args.prior, args.resume)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'paired': report['paired']}))
