"""External GR00T v2 development admission; never launches models/rollouts/Astra.

After the local development broker exits, --snapshot-broker-terminal seals its
complete durable inventory. Upload that directory, reaggregate to a NEW analysis
directory with --broker-state, and run --check on Linux. --approve repeats all
checks and exclusively writes the independent development admission-review.json.
All own workers AND the development model must have terminated; an otherwise
successful but still live service is not waived. Negative outcomes may pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NAMESPACE = 'originx_gr00t_confirmation_20261009_v2'
SERVICE_NAMESPACE = 'originx_gr00t_confirmation_20261009'
DEV = 'originx-gr00t-confirmatory-development-20261009-v2'
PRIOR_DEV = 'originx-gr00t-confirmatory-development-20261009-v1'
FULL = 'originx-gr00t-confirmatory-20261009-v1'
HELPER_SHA = 'a9cc287845d188be23f2bf26cbb94b82d4915be5134669cd5415537b9a4fa170'
POLICY_ID = 'gr00t_n15_robocasa_multitask120000'


def helper(path):
    import hashlib
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError('Frozen common admission helper changed')
    spec = importlib.util.spec_from_file_location('gr00t_admission_common', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    # Only the generic local broker-terminal snapshot's allowed output label is
    # specialized in memory. No frozen file or experiment record is changed.
    module.DEV_NAME = DEV
    return module


def runtime_modules(root):
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(NAMESPACE + '.aggregate'), importlib.import_module(NAMESPACE + '.campaign')
    finally:
        sys.path.remove(str(root))


def snapshot_broker_terminal(h, state, source, output):
    """Empty owned batch paths have no matches; avoid PowerShell null coercion.

    Nonempty invocations retain the frozen helper's exact process scan. The
    empty-set case is allowed only with zero recorded calls everywhere and no
    durable CLI starts or receipts. No running process is ignored by identity.
    """
    state=Path(state); starts=list((state/'batches').glob('*/cli_started.json'))
    if starts:return h.snapshot_broker_terminal(state,source,output)
    h.require(h.read(state/'broker_status.json').get('cli_calls')==0,'Zero-start snapshot has nonzero call count')
    for path in state.glob('*.json'):
        value=h.read(path)
        if isinstance(value,dict) and 'cli_calls' in value:h.require(value['cli_calls']==0,'Inconsistent zero-call inventory')
    h.require(not list((state/'batches').glob('*/receipt.json')),'Zero starts with CLI receipt is not empty')
    def empty_set_scan(paths):
        h.require(not paths and not list((state/'batches').glob('*/cli_started.json')),'CLI inventory changed during empty scope scan')
        return []
    return h.snapshot_broker_terminal(state,source,output,process_scan=empty_set_scan)


def verify_raw(config, config_sha, manifest, output, broker_state):
    aggregate, _ = runtime_modules(Path(config['root']))
    aggregate.validate_manifest(manifest)
    projected = aggregate.make_manifest(aggregate.read(config['reference_manifest']), True)
    aggregate.require({k:v for k,v in manifest.items() if k not in ('source_manifest_path','source_manifest_sha256')} == projected,
                      'Development differs from the exact disclosed two-case projection')
    jobs = {j['request_id']: j for j in manifest['jobs']}
    normalized, audits = [], []
    for job in manifest['jobs']:
        row, audit = aggregate.normalize_episode(config, config_sha, job, output, jobs, True)
        normalized.append(row); audits.append(audit)
    aggregate.require(all(row['valid'] for row in audits), 'Fresh raw verification failed: ' + repr(audits))
    usage = aggregate.collect_usage(output, jobs, config_sha, config['manifest_sha256'], broker_state)
    analyze = aggregate.load_analysis(Path(config['root']) / 'originx_confirmatory_20261009/analysis.py')
    rows = [dict(case_id=j['case_id'],task_name=j['task'],policy_id=j['policy_id'],arm=j['arm'],
                 seed=j['env_seed'],horizon=j['horizon'],episode_id=j['id'],query_interval=16) for j in manifest['jobs']]
    report = analyze(rows, normalized, N=2)
    comparison = report['policies'][POLICY_ID]['comparisons']['V_vs_C']
    pairs = {key: comparison[key] for key in ('categories','valid_paired_2x2','unresolved_comparisons')}
    return normalized, audits, usage, pairs


def verify_parity(config, runtime_admission, evidence, h):
    """Recheck direct actions, six client streams and exact native fixtures."""
    import numpy as np
    _, campaign = runtime_modules(Path(config['root']))
    numerical = runtime_admission['socket_parity']; path = Path(numerical['path'])
    checked = campaign.verify_parity(path, numerical['sha256'], config['models'][0])
    h.require(checked == numerical, 'Saved runtime parity admission differs from fresh verification')
    socket = evidence(path)
    direct_path = Path(socket['reference']); direct = evidence(direct_path)
    fixtures_path = Path(direct['fixtures']); fixtures = evidence(fixtures_path)
    capture = evidence(fixtures_path.parent / 'capture-receipt.json')
    guard_path = fixtures_path.parent / 'readback-guard/summary.json'; guard = evidence(guard_path)
    h.require(capture.get('schema') == 'originx_gr00t_native_fixture_capture_v1'
              and capture.get('passed') is True and capture.get('synthetic') is False
              and capture.get('environment_steps') == capture.get('policy_queries') == 0
              and capture.get('development_ids_disclosed') is True and capture.get('no_scored_outcomes_read') is True
              and capture.get('readback_guard_sha256') == h.sha(guard_path), 'Native fixture capture provenance differs')
    h.require(guard.get('status') == 'closed' and guard.get('failure') is None
              and guard.get('raw_frames_checked', 0) > 0 and guard.get('sentinel_checks', 0) > 0
              and set(guard.get('gl_error_counts', {})) <= {'after_draw:1281'}, 'Native fixture rendering guard failed')
    proof = capture['native_reset']
    h.require(proof.get('native_reset') is True and proof.get('reset_patch_applied') is False
              and proof.get('pythonhashseed') == '0' and proof.get('source_sha256') ==
              '77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a', 'Fixture natural reset proof differs')
    h.require(capture.get('reference_manifest_sha256') == config['reference_manifest_sha256']
              and capture.get('reference_config_sha256') == config['reference_config_sha256'], 'Fixture reference pins differ')
    h.require(direct.get('no_environment_rollouts') is True and direct.get('no_confirmation_outcomes_read') is True,
              'Direct parity accessed scored outcomes')
    for stream in direct['streams']:
        for query in stream['trace']:
            action = np.asarray(query['full_actions'], dtype=np.float32)
            h.require(action.shape == (16,12) and np.isfinite(action).all()
                      and campaign.r.sha(path) == numerical['sha256'], 'Direct action tensor or parity record changed')
            from_module = importlib.import_module(NAMESPACE + '.parity_probe')
            h.require(from_module.action_hash(action) == query['action_sha256'], 'Direct full action hash differs')
    # Every file in the small parity and capture directories is bound, including
    # client JSON and public-input NPZ. These directories contain no model weights.
    directories = (path.parent, direct_path.parent, fixtures_path.parent)
    return dict(socket_clients=6, forwards=18, hold_seconds=socket['hold_seconds'],
                actual_native_fixtures=2, direct_full_actions_verified=True,
                directories=[str(p) for p in directories],
                owner_paths=[str(p/'owner.json') for p in directories])


def check(root, broker_state, terminal_path, analysis, h, *, raw_verifier=verify_raw, parity_verifier=verify_parity,
          owner_stopped=None):
    root = Path(root).resolve(); output = root/'results'/DEV
    broker_state, terminal_path, analysis = map(lambda p:Path(p).resolve(), (broker_state, terminal_path, analysis))
    h.require(analysis.parent == output and analysis.name.startswith('analysis'), 'Analysis outside own development output')
    h.require(broker_state.is_relative_to(output) and terminal_path.is_relative_to(broker_state), 'Uploaded local inventory must stay under dev output')
    full = root/'results'/FULL
    h.require(not (full/'launch.json').exists() and not (full/'campaign.owner.json').exists()
              and not any((full/'claims').glob('*.json')) and not any((full/'episodes').glob('*/result.json')),
              'Full confirmation already started; admission is not prospective')
    h.require(not (root/NAMESPACE/'campaigns'/FULL/'owner.json').exists(), 'Full campaign owner already exists')
    owner_stopped = h.linux_owner_stopped if owner_stopped is None else owner_stopped
    bindings = {}
    def evidence(path):
        path = Path(path).resolve()
        h.require(path.is_relative_to(root), 'Evidence path outside project')
        value=h.read(path);bindings[str(path)]=h.sha(path);return value
    def bind_tree(directory):
        directory=Path(directory).resolve();h.require(directory.is_relative_to(root), 'Artifact tree outside project')
        for name,digest in h.inventory_files(directory).items():bindings[str(directory/name)]=digest
    config=evidence(output/'config.json');config_sha=h.sha(output/'config.json')
    h.require(config.get('schema')=='originx_gr00t_confirmatory_config_v1' and config.get('namespace')==NAMESPACE
              and config.get('service_namespace')==SERVICE_NAMESPACE
              and config.get('policy_id')==POLICY_ID and config.get('development') is True
              and config.get('root')==str(root) and config.get('output')==str(output), 'Wrong development configuration')
    prior=root/'results'/PRIOR_DEV
    prior_completion=evidence(prior/'completion.json');prior_finished=evidence(prior/'campaign-finished.json')
    prior_audit=evidence(prior/'failure-terminal-audit.json');prior_cli=evidence(prior/'broker_inventory/terminal-audit.json')
    evidence(prior/'supervisor-error.json');evidence(prior/'failed-export-receipt.json')
    h.require(prior_completion.get('all_children_drained') is False and prior_finished.get('failed') is True,
              'Original v1 failed records were not preserved')
    h.require(prior_audit.get('schema')=='originx_gr00t_failed_development_terminal_audit_v1'
              and prior_audit.get('all_recorded_worker_and_model_identities_inactive') is True
              and prior_audit.get('request_count')==0 and prior_audit.get('original_completion_preserved') is True
              and prior_cli.get('cli_starts')==0 and prior_cli.get('broker_owner_inactive') is True,
              'Prior development failure/zero-call terminal evidence incomplete')
    h.require(config.get('manifest')==str(output/'manifest.json'), 'Manifest path differs')
    for key in ('reference_config', 'reference_manifest'):
        evidence(config[key]);h.require(h.sha(config[key])==config[key+'_sha256'], 'Pinned '+key+' changed')
    manifest=evidence(output/'manifest.json')
    h.require(h.sha(output/'manifest.json')==config['manifest_sha256'] and manifest.get('development') is True
              and manifest.get('case_count')==2 and manifest.get('arm_outcome_count')==4
              and manifest.get('seed_selection_uses_outcomes') is False, 'Development denominator/authority differs')
    h.require(manifest.get('source_manifest_path')==config['reference_manifest']
              and manifest.get('source_manifest_sha256')==config['reference_manifest_sha256'], 'Projected roster binding differs')
    jobs=manifest['jobs'];h.require(len(jobs)==4 and len({j['id'] for j in jobs})==len({j['request_id'] for j in jobs})==4,
                                   'Exactly four unique arm assignments required')
    cases={j['case_id'] for j in jobs}
    h.require(len(cases)==2 and all({(j['policy_id'],j['arm']) for j in jobs if j['case_id']==case}=={(POLICY_ID,'C'),(POLICY_ID,'V')} for case in cases),
              'Require both C/V arms for each disclosed development case')
    freeze=evidence(config['source_freeze'])
    h.require(h.sha(config['source_freeze'])==config['source_freeze_sha256']
              and freeze.get('schema')=='originx_gr00t_source_freeze_v1' and freeze.get('namespace')==NAMESPACE
              and freeze.get('broker_source_sha256')==config['broker_source_sha256'], 'Independent source-freeze authority differs')
    h.require(bool(config.get('source_sha256')), 'Empty source inventory')
    for name,digest in config['source_sha256'].items():
        path=Path(name);h.require(path.resolve().is_relative_to(root) and not path.is_symlink() and h.sha(path)==digest,
                                'Frozen executable source differs: '+name);bindings[name]=digest
    for name,digest in freeze['source_sha256'].items():
        path=Path(config['source_freeze']).parent/name
        h.require(Path(name).name==name and config['source_sha256'].get(str(path))==digest, 'Freeze/config module pin mismatch')
    preflight=evidence(output/'preflight.json');ready=evidence(output/'broker-ready.json')
    for receipt in (preflight,ready):
        h.require(receipt.get('config_sha256')==config_sha and receipt.get('manifest_sha256')==config['manifest_sha256'], 'Runtime readiness bound to another configuration')
    h.require(preflight.get('passed') is True and ready.get('broker_source_sha256')==config['broker_source_sha256']
              and ready.get('model')=='gpt-6-astra' and ready.get('reasoning_effort')=='high', 'Frozen broker/source readiness differs')
    runtime=evidence(output/'runtime-admission.json')
    h.require(runtime.get('schema')=='originx_gr00t_runtime_admission_v1' and runtime.get('passed') is True
              and runtime.get('development') is True and runtime.get('config_sha256')==config_sha
              and runtime.get('manifest_sha256')==config['manifest_sha256'], 'Runtime parity admission differs')
    parity=parity_verifier(config,runtime,evidence,h)
    for directory in parity['directories']:bind_tree(directory)
    completion=evidence(output/'completion.json');finished=evidence(output/'campaign-finished.json')
    h.require(completion.get('schema')=='originx_gr00t_confirmatory_completion_v1' and completion.get('complete') is True
              and completion.get('planned')==completion.get('finished')==4 and completion.get('all_children_drained') is True
              and completion.get('remaining_owned_workers')==[] and completion.get('unstarted_arm_outcomes')==0,
              'Development workers/outcomes not fully completed and drained')
    h.require(finished.get('schema')=='originx_gr00t_campaign_finished_v1' and finished.get('development') is True
              and finished.get('failed') is False and finished.get('phase')=='completed'
              and finished.get('rollout_returncode')==0, 'Development campaign did not complete')
    cleanup=evidence(output/'cleanup.json')
    h.require(cleanup.get('schema')=='originx_gr00t_confirmatory_cleanup_v1'
              and cleanup.get('all_owned_models_inactive') is True and cleanup.get('foreign_processes_untouched') is True,
              'Development models have not been owner-scoped cleaned up')
    h.require(len(cleanup.get('models',[]))==len(config['models'])==1, 'Development model cleanup inventory differs')
    model=config['models'][0];clean=cleanup['models'][0]
    h.require(clean.get('owner')==model['owner'] and clean.get('server_manifest')==model['server_manifest']
              and clean.get('owned_process_still_active') is False, 'Cleanup receipt owner/profile differs')
    model_owner=evidence(Path(model['server_manifest']).parent/'owner.json')
    h.require(model_owner==model['owner'] and owner_stopped(model_owner), 'Development model owner remains alive')
    model_terminal=evidence(Path(model['server_manifest']).parent/'terminated.json')
    h.require(model_terminal.get('owner')==model_owner and model_terminal.get('model_thread_drained') is True,
              'Development model thread did not drain normally')
    control=root/NAMESPACE/'campaigns'/DEV
    owner_paths=[output/'campaign.owner.json',output/'launch.json',control/'owner.json',
                 control/'preflight.owner.json',control/'rollout.owner.json',control/'aggregate.owner.json']
    owner_paths += [Path(p) for p in parity['owner_paths']]
    for job in jobs:owner_paths += [output/'episodes'/job['id']/'owner.json',output/'processes'/(job['id']+'.json')]
    owners=[]
    for path in owner_paths:
        owner=evidence(path);h.require(owner_stopped(owner), 'Owned development worker is still live: '+str(path))
        owners.append(dict(path=str(path),pid=owner['pid'],process_start_ticks=owner['process_start_ticks'],stopped=True))
    terminal=evidence(terminal_path)
    h.require(terminal.get('schema')=='originx_confirmatory_broker_terminal_audit_v1'
              and terminal.get('broker_owner_inactive') is True and terminal.get('all_cli_wait_returns_terminal') is True
              and terminal.get('all_owned_cli_processes_absent') is True and terminal.get('scoped_cli_process_matches')==[]
              and terminal.get('no_cli_invoked_by_audit') is True
              and terminal.get('broker_source_sha256')==config['broker_source_sha256']
              and terminal.get('excluded_non_evidence_files')==['broker.lock'], 'Missing strict complete local broker terminal audit')
    inventory=h.inventory_files(broker_state,excluded=[terminal_path,broker_state/'broker.lock'])
    h.require(inventory==terminal['files_sha256'], 'Uploaded broker inventory changed or is incomplete')
    for name,digest in inventory.items():bindings[str(broker_state/name)]=digest
    identity=evidence(broker_state/'identity.json');broker_owner=evidence(broker_state/'owner.json')
    h.require(identity==terminal.get('broker_identity') and broker_owner==terminal.get('broker_owner')
              and identity.get('config_sha256')==config_sha and identity.get('manifest_sha256')==config['manifest_sha256']
              and identity.get('remote_output')==str(output), 'Local broker inventory belongs to another study')
    report=evidence(analysis/'report.json');authority=evidence(analysis/'authority.json')
    normalized=evidence(analysis/'normalized_records.json');usage=evidence(analysis/'cli_usage.json')
    h.require(report.get('schema')=='originx_gr00t_confirmation_analysis_v1' and report.get('development') is True
              and report.get('scored_confirmation') is False and report.get('authority_valid') is True
              and report.get('raw_evidence_valid')==report.get('planned_arm_outcomes')==report.get('present_arm_records')==4
              and report.get('N_per_policy')==2 and report.get('config_sha256')==config_sha
              and report.get('manifest_sha256')==config['manifest_sha256'], 'Final aggregate is stale, incomplete or invalid')
    h.require(authority.get('config_sha256')==config_sha and authority.get('manifest_sha256')==config['manifest_sha256'], 'Aggregate authority differs')
    for entry in authority['source_checks']:
        path=Path(entry['path']);h.require(entry.get('valid') is True and path.resolve().is_relative_to(root)
              and h.sha(path)==entry['expected_sha256']==entry['actual_sha256'], 'Aggregate executable/checkpoint source authority changed')
        bindings[str(path)]=entry['expected_sha256']
    rows,audits,recomputed_usage,pairs=raw_verifier(config,config_sha,manifest,output,broker_state)
    h.require(rows==normalized and audits==authority['episode_checks'] and all(a['valid'] for a in audits), 'Fresh raw verification differs from final aggregate')
    h.require(usage==recomputed_usage and report['cli_usage']=={k:v for k,v in usage.items() if k!='calls'}, 'Complete actual CLI inventory not reflected in final aggregate')
    h.require(usage.get('inventory_complete') is True and usage.get('broker_inventory_provided') is True
              and usage.get('complete_input_output_token_accounting') is True and usage.get('invalid_receipts')==[]
              and usage.get('interrupted_cli_starts_without_receipt')==[], 'CLI inventory/accounting incomplete')
    h.require(terminal['cli_starts']==len(terminal['verified_cli_wait_returns'])==usage['unique_observed_cli_invocations'], 'Terminal CLI count differs from deduplicated accounting')
    applied=0
    for row in rows:
        h.require(row.get('valid') is True, 'Raw-invalid development record')
        if row['arm']=='C':
            h.require(not row['assistance'].get('delivered') and len({q['instruction_hash'] for q in row['trace']})==1,
                      'Control instruction changed')
        elif row['assistance'].get('delivered') is True:
            h.require(row['assistance'].get('evidence_valid') is True and row['assistance'].get('triggered') is True
                      and row['assistance'].get('cli_evidence',{}).get('actual_cli_and_application_verified') is True,
                      'V application lacks actual linked Astra receipt')
            applied+=1
    h.require(applied>=1, 'At least one actual verified V application required; E=0 cannot pass')
    h.require(pairs.get('unresolved_comparisons')==0 and pairs.get('categories',{}).get('matched')==2,
              'Initial/prefix paired evidence is incomplete or deviates')
    saved=report['policies'][POLICY_ID]['comparisons']['V_vs_C']
    h.require(all(saved.get(k)==v for k,v in pairs.items()), 'Fresh paired comparison differs')
    for directory in ('episodes','requests','claims','processes'):bind_tree(output/directory)
    for path,digest in bindings.items():h.require(h.sha(path)==digest, 'Evidence changed during admission')
    return dict(schema='originx_gr00t_development_admission_v1',passed=True,no_confirmatory_outcomes_seen=True,
                outcome_blindness_scope='This independent GR00T confirmation only; earlier disclosed development evidence is retained.',
                config_sha256=config_sha,manifest_sha256=config['manifest_sha256'],source_freeze_sha256=config['source_freeze_sha256'],
                broker_source_sha256=config['broker_source_sha256'],development_cases=2,raw_verified_arm_outcomes=4,
                study_namespace=NAMESPACE,service_namespace=SERVICE_NAMESPACE,prior_failed_development_preserved=PRIOR_DEV,
                matched_case_pairs=2,verified_V_applications=applied,actual_cli_calls=usage['unique_observed_cli_invocations'],
                local_broker_usage_totals=usage['known_usage_totals'],all_owned_models_stopped=True,owned_worker_checks=owners,
                parity=parity,negative_policy_outcomes_allowed=True,development_is_not_effectiveness_evidence=True,
                bindings_sha256=bindings,admission_tool_sha256=h.sha(__file__),common_helper_sha256=HELPER_SHA,
                checked_at=h.utc(),rollouts_or_CLI_started_by_tool=False,
                cli_termination_evidence=terminal['process_check'],per_cli_pid_start_cwd_available=False)


def approve(candidate, root, h):
    h.require(candidate.get('passed') is True and candidate.get('no_confirmatory_outcomes_seen') is True, 'Failed admission cannot be approved')
    for path,digest in candidate['bindings_sha256'].items():h.require(h.sha(path)==digest, 'Evidence changed before approval')
    path=Path(root)/'results'/DEV/'admission-review.json';h.exclusive_write(path,candidate);return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check',action='store_true');actions.add_argument('--approve',action='store_true')
    actions.add_argument('--snapshot-broker-terminal',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--helper-source',type=Path,default=Path(__file__).with_name('admit_full.py'))
    parser.add_argument('--broker-state',type=Path,required=True);parser.add_argument('--broker-terminal',type=Path,required=True)
    parser.add_argument('--broker-source',type=Path);parser.add_argument('--analysis-directory',type=Path)
    args=parser.parse_args()
    try:
        h=helper(args.helper_source)
        if args.snapshot_broker_terminal:
            h.require(args.broker_source is not None,'Snapshot needs actual local --broker-source')
            receipt=snapshot_broker_terminal(h,args.broker_state,args.broker_source,args.broker_terminal)
            print(json.dumps(dict(passed=True,cli_starts=receipt['cli_starts'],audit=str(args.broker_terminal))))
        else:
            result=check(args.root,args.broker_state,args.broker_terminal,
                         args.analysis_directory or args.root/'results'/DEV/'analysis-final',h)
            if args.approve:result['approval_path']=str(approve(result,args.root,h))
            print(json.dumps(result,indent=2))
        return 0
    except Exception as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,error=str(error))));return 2


if __name__=='__main__':raise SystemExit(main())
