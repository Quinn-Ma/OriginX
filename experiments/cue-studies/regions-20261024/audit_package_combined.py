"""Execution-evidence audit and byte-verified backup for the split 24-case run.

The registered cases live in two directories: the interrupted run (six completed
cases, no launcher receipt) and its resume (the 18 owed cases). The frozen
package_regions_validation.py functions are reused unchanged; only the receipt
gate, the per-case run directory and the resume driver identity differ.
No model, simulator or live run is touched.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = HERE / 'package_regions_validation.py'
spec = importlib.util.spec_from_file_location('regions_package_frozen', FROZEN)
pkg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pkg)
require, read, sha256_file, canonical_hash, atomic_json = pkg.require, pkg.read, pkg.sha256_file, pkg.canonical_hash, pkg.atomic_json

COMBINED_ANALYZER = HERE / 'analyze_regions_validation_combined.py'
RESUME_DRIVER = ROOT / 'scripts/run_id60_regions_validation_resume_local.py'
REGISTERED_DRIVER = 'scripts/run_id60_regions_validation_local.py'
INTERRUPTION = 'interruption-record-20261024.json'
AUDIT = HERE / 'execution-evidence-audit-combined-v1.json'
OUT = ROOT / 'outputs/id60-regions-validation'
DELIVERABLES = ('paper/main.pdf', '实验结论.txt', 'validation-results.json', 'paper-sources.zip')


def local(path):
    """Map a WSL /mnt/<drive>/ path recorded by the resume launcher to this host."""
    text = str(path)
    if text.startswith('/mnt/') and len(text) > 6 and Path('C:/').exists():
        return Path(text[5].upper() + ':/' + text[7:])
    return Path(text)


def results(run):
    return {p.stem for p in (run / 'worker-0/run/results').glob('*.json')}


def traces(run):
    return {p.name for p in (run / 'worker-0/run/traces').glob('*') if p.is_dir()}


def gate(prior, resume, manifest):
    """All receipt/partition checks; must precede any journal traversal."""
    receipt = read(resume / 'launch-receipt.json')
    require(receipt.get('owned_processes_stopped') is True, 'All resume-owned processes must be stopped first')
    require(not receipt.get('cleanup_errors'), 'Resume cleanup has unresolved errors')
    require(receipt.get('purpose') == pkg.PURPOSE, 'Wrong resume launcher purpose')
    require(receipt.get('status') in ('all_owed_cases_accounted', 'partial'), 'Resume launcher is not terminal')
    require(not (prior / 'launch-receipt.json').exists(), 'Prior run has a receipt; it was not an interrupted run')
    record = read(prior / INTERRUPTION)
    require(record.get('launch_receipt_present') is False and record.get('interrupted_case_trace_records') == 0,
            'Interruption record does not describe a zero-step kill without cleanup')
    plan = read(resume / 'resume-plan.json')
    execution = read(resume / 'local-execution.json')
    require(execution.get('resume_plan_sha256') == canonical_hash(plan), 'Resume plan changed after registration')
    require(execution.get('driver_sha256') == sha256_file(RESUME_DRIVER), 'Resume driver changed after launch')
    require(execution.get('registered_driver_sha256') == manifest['source_files'][REGISTERED_DRIVER], 'Registered driver changed')
    require(local(plan['prior_output']).resolve() == prior.resolve(), 'Resume plan names a different prior run')
    ids = {c['case_id'] for c in manifest['cases']}
    completed, owed, zero = plan['prior_completed_result_sha256'], plan['owed_case_ids'], plan['prior_interrupted_zero_step']
    require(len(owed) == len(set(owed)) and set(completed).isdisjoint(owed) and set(completed) | set(owed) == ids,
            'Resume plan does not partition the 24 registered cases')
    for cid, digest in completed.items():
        require(sha256_file(prior / 'worker-0/run/results' / (cid + '.json')) == digest, 'Prior result changed: ' + cid)
    for cid, value in zero.items():
        folder = prior / 'worker-0/run/traces' / cid
        require(cid in owed and record.get('interrupted_case_trace_sha256') == value['trace_sha256'], 'Zero-step case record mismatch')
        require({p.name for p in folder.iterdir()} == {'trace.json'} and sha256_file(folder / 'trace.json') == value['trace_sha256'],
                'Zero-step trace changed or gained records: ' + cid)
    require(results(prior) == set(completed) and results(resume) == set(owed), 'Unregistered or missing result slots')
    require(traces(prior) <= set(completed) | set(zero) and traces(resume) <= set(owed), 'Unregistered trace directory')
    return receipt, record, plan


def audit(prior, resume, analysis_path):
    manifest = pkg.load_manifest(prior / 'manifest.json', ROOT)
    require(read(resume / 'manifest.json') == read(prior / 'manifest.json'), 'Prior and resume manifests differ')
    receipt, record, plan = gate(prior, resume, manifest)
    require(manifest['purpose'] == pkg.PURPOSE and len(manifest['cases']) == 24
            and len(manifest['source_files']) == 138 and manifest['historical_source_count'] == 133, 'Wrong study/source scope')
    require(sha256_file(pkg.ANALYZER) == pkg.ANALYZER_SHA and sha256_file(pkg.ANALYSIS_GATE) == pkg.ANALYSIS_GATE_SHA, 'Final analyzer source changed')
    analysis = read(analysis_path)
    require(analysis.get('analysis_sha256') == pkg.ANALYZER_SHA and analysis.get('complete_trace_gate_sha256') == pkg.ANALYSIS_GATE_SHA
            and analysis.get('combined_wrapper_sha256') == sha256_file(COMBINED_ANALYZER), 'Analysis generated by unregistered analyzer')
    require(analysis.get('status') == 'PROSPECTIVE_OFFLINE_ANALYSIS' and analysis.get('purpose') == pkg.PURPOSE
            and analysis.get('manifest_sha256') == manifest['manifest_sha256'] and analysis.get('resume_plan') == plan, 'Wrong final analysis source')
    require(analysis.get('source_files_match') is True and not analysis.get('source_mismatches'), 'Analysis source verification failed')
    rows = analysis.get('cases')
    require(isinstance(rows, list) and [r['case_id'] for r in rows] == [c['case_id'] for c in manifest['cases']], 'Analysis accounting changed or incomplete')
    owed = set(plan['owed_case_ids'])
    runs = {'prior': prior, 'resume': resume}
    services, configs = {}, {}
    for name, run in runs.items():
        view = manifest
        if name == 'resume':
            # The frozen check compares the launcher hash to the registered driver; the resume
            # launch is bound to the resume driver instead (checked in gate()).
            view = {**manifest, 'source_files': {**manifest['source_files'], REGISTERED_DRIVER: sha256_file(RESUME_DRIVER)}}
        planner, policy, config = pkg.audit_configuration(run, view)
        services[name], configs[name] = (planner, policy), config
    require(services['prior'] == services['resume'], 'Planner/policy identities differ between the two launches')
    gpu = pkg.load_cuda_tool()
    checks, known, unknown, inputs = {'prior': [], 'resume': []}, [], [], {}
    for case, analyzed in zip(manifest['cases'], rows):
        name = 'resume' if case['case_id'] in owed else 'prior'
        run = runs[name]; planner, policy = services[name]
        require(analyzed.get('source_run') == name, 'Analysis read a case from the wrong run')
        result_path = run / 'worker-0/run/results' / (case['case_id'] + '.json')
        result = read(result_path)
        require(result.get('manifest_sha256') == manifest['manifest_sha256'] and all(result.get(k) == v for k, v in case.items()), 'Result case identity mismatch')
        require(result.get('status') in ('success', 'failure', 'unknown'), 'Unknown result status')
        require(analyzed.get('result_sha256') == sha256_file(result_path), 'Analysis result hash mismatch')
        require(analyzed.get('outcome') == result['status'], 'Analysis changed result outcome')
        if result['status'] != 'unknown':
            require(planner is not None and policy is not None, 'Known case requires both verified service identities')
            require(not analyzed.get('integrity_issues'), 'Known analysis reports trace integrity issues')
            require(result.get('planner_identity') == planner and result.get('policy_identity') == policy, 'Result service identities differ')
            known.append(case['case_id'])
        else:
            unknown.append({'case_id': case['case_id'], 'source_run': name, 'category': result.get('outcome_category'), 'reason': result.get('reason')})
        check = pkg.audit_trace(run / 'worker-0/run/traces' / case['case_id'], case, result, planner, gpu)
        if check.get('journal_sha256'):
            require(analyzed.get('journal_sha256') == check['journal_sha256'], 'Analysis journal hash mismatch')
        for file, ref in check.pop('verified_trace_files', {}).items():
            inputs[(run / 'worker-0/run/traces' / case['case_id'] / file).relative_to(ROOT).as_posix()] = ref
        check['source_run'] = name
        checks[name].append(check)
        print(json.dumps({'phase': 'trace_arrays_verified', 'source_run': name, 'case_id': case['case_id'],
                          'status': result['status'], 'npz': check.get('npz_records', 0)}), flush=True)
    # The prior launcher was killed without cleanup, so an unmapped native suffix is possible there.
    native = {'prior': pkg.audit_native_evidence(prior / 'native-cuda-evidence.jsonl', checks['prior'], True, gpu),
              'resume': pkg.audit_native_evidence(resume / 'native-cuda-evidence.jsonl', checks['resume'],
                                                  any(u['source_run'] == 'resume' for u in unknown) or receipt['status'] == 'partial', gpu)}
    archives = pkg.audit_prior_archives()
    extra = [Path(analysis_path), pkg.ANALYZER, pkg.ANALYSIS_GATE, pkg.CUDA_TOOL, pkg.PRIOR_REFS, FROZEN,
             COMBINED_ANALYZER, RESUME_DRIVER, Path(__file__)]
    for path in [*pkg.retained_files(prior), *pkg.retained_files(resume), *extra]:
        key = path.relative_to(ROOT).as_posix()
        if key not in inputs:
            inputs[key] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    for name, digest in manifest['source_files'].items():
        inputs[name] = {'sha256': digest, 'bytes': (ROOT / name).stat().st_size}
    return {'status': 'PASS',
        'execution_evidence_status': 'ALL_CASES_EXECUTION_EVIDENCE_VERIFIED' if not unknown else 'PARTIAL_EXECUTION_EVIDENCE_VERIFIED',
        'purpose': pkg.PURPOSE, 'manifest_sha256': manifest['manifest_sha256'],
        'analysis_path': str(analysis_path), 'analysis_sha256': sha256_file(analysis_path),
        'prior_run': prior.relative_to(ROOT).as_posix(), 'resume_run': resume.relative_to(ROOT).as_posix(),
        'resume_launch_receipt_sha256': sha256_file(resume / 'launch-receipt.json'),
        'interruption_record_sha256': sha256_file(prior / INTERRUPTION), 'resume_plan_sha256': canonical_hash(plan),
        'reattempted_zero_step_case_ids': sorted(plan['prior_interrupted_zero_step']),
        'source_script_sha256': sha256_file(Path(__file__)), 'frozen_package_script_sha256': sha256_file(FROZEN),
        'audited_input_files': inputs, 'owned_processes_stopped': True,
        'source_files_verified': 138, 'historical_sources_verified': 133, 'old_archives_unchanged': archives,
        'configuration': configs, 'service_identities_equal_across_launches': True,
        'known_case_ids': known, 'unknown': unknown, 'trace_checks': checks['prior'] + checks['resume'],
        'native_cuda': native,
        'planner_generations_verified': sum(c['planner_generations_verified'] for c in checks['prior'] + checks['resume']),
        'all_persisted_planner_generations_passed': not any(c.get('partial_cuda_errors') for c in checks['prior'] + checks['resume']),
        'native_encoded_input_tensors_instrumented': False, 'complete_physics_replay_claimed': False,
        'new_model_generations': 0, 'new_simulator_calls': 0, 'resume_launcher_elapsed_seconds': receipt.get('elapsed_seconds')}


def package(prior, resume, analysis_path):
    value = read(AUDIT)
    checksum = AUDIT.with_suffix('.sha256')
    require(checksum.is_file() and checksum.read_text().split()[0] == sha256_file(AUDIT), 'Execution audit checksum missing or changed')
    require(value.get('status') == 'PASS' and value.get('source_script_sha256') == sha256_file(Path(__file__)), 'Unverified or stale combined audit')
    require(value.get('analysis_sha256') == sha256_file(analysis_path), 'Analysis changed after audit')
    require(value.get('resume_launch_receipt_sha256') == sha256_file(resume / 'launch-receipt.json'), 'Resume receipt changed after audit')
    require(pkg.audit_prior_archives() == value['old_archives_unchanged'], 'Historical archives changed after audit')
    for name in DELIVERABLES:
        require((OUT / name).is_file() and (OUT / name).stat().st_size > 0, 'Final deliverable is not ready: ' + name)
    require(sha256_file(OUT / 'validation-results.json') == value['analysis_sha256'], 'User-facing result JSON differs from audited analysis')
    archive, inventory_path, verify = OUT / 'id60-regions-validation-combined-full.zip', OUT / 'backup-inventory.json', OUT / 'backup-verification.json'
    require(not archive.exists() and not verify.exists(), 'Backup target already exists')
    manifest = pkg.load_manifest(prior / 'manifest.json', ROOT)
    files = {p.relative_to(ROOT).as_posix(): p for folder in (prior, resume, HERE, OUT) for p in pkg.retained_files(folder)}
    for name in manifest['source_files']:
        files[name] = ROOT / name
    for path in (Path(analysis_path), ROOT / pkg.PARENT, pkg.CUDA_TOOL, pkg.ANALYSIS_GATE, RESUME_DRIVER):
        files[path.relative_to(ROOT).as_posix()] = path
    started = time.monotonic()
    inventory = pkg.write_verified_archive(files, archive, inventory_path, value['audited_input_files'])
    atomic_json(verify, {'status': 'PASS_BYTE_VERIFIED_BACKUP', 'archive': str(archive), 'archive_sha256': sha256_file(archive),
        'archive_bytes': archive.stat().st_size, 'member_count': len(inventory), 'inventory_sha256': sha256_file(inventory_path),
        'manifest_sha256': manifest['manifest_sha256'], 'analysis_file_sha256': value['analysis_sha256'],
        'execution_audit_sha256': sha256_file(AUDIT), 'execution_evidence_status': value['execution_evidence_status'],
        'known_cases': len(value['known_case_ids']), 'unknown_cases': len(value['unknown']),
        'runs_included': [value['prior_run'], value['resume_run']], 'weights_included': False,
        'elapsed_seconds': time.monotonic() - started, 'source_script_sha256': sha256_file(Path(__file__)),
        'new_model_generations': 0, 'new_simulator_calls': 0})
    print(json.dumps({'status': 'PASS_BYTE_VERIFIED_BACKUP', 'archive': str(archive)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('audit', 'package'), required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    args = p.parse_args()
    prior, resume, analysis = args.prior.resolve(), args.resume.resolve(), args.analysis.resolve()
    require(prior.is_relative_to(ROOT / 'runs') and resume.is_relative_to(ROOT / 'runs') and analysis.is_relative_to(ROOT),
            'Unexpected run/analysis scope')
    if args.phase == 'package':
        return package(prior, resume, analysis)
    require(not AUDIT.exists() and not AUDIT.with_suffix('.sha256').exists(), 'Never replace a prior execution audit')
    value = audit(prior, resume, analysis)
    atomic_json(AUDIT, value)
    AUDIT.with_suffix('.sha256').write_text(sha256_file(AUDIT) + '  ' + AUDIT.name + '\n', encoding='utf-8')
    print(json.dumps({'status': value['status'], 'evidence': value['execution_evidence_status'], 'audit': str(AUDIT)}), flush=True)


if __name__ == '__main__':
    main()
