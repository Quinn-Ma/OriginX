"""Local-only finalization after verified synchronization; never invokes a model or SSH."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

import aggregate_campaigns as cases
import aggregate_cli_usage as usage
import sync_continuation as sync

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / 'outputs'
REMOTE = sync.PROJECT + '/results/astra-native-reset-continuation-20261009-v2'
POLL_SECONDS = 30


def write(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def wait_sync(directory, run_seconds, clock=time.monotonic, sleeper=time.sleep):
    deadline = clock() + run_seconds
    error = None
    while True:
        try:
            finished = cases.read(directory / 'sync_finished.json')
            if finished.get('status') in ('evidence_collected', 'synchronization_stopped_incomplete'):
                return finished
            error = 'Unrecognized sync_finished status'
        except (OSError, ValueError) as exc:
            error = type(exc).__name__
        write(directory / 'finalizer-status.json', dict(stage='waiting_for_local_sync',
              observed_utc=sync.utc(), last_read_error=error, remote_actions=False))
        if clock() >= deadline:
            return dict(status='finalizer_wait_expired', error=error)
        sleeper(min(POLL_SECONDS, max(0, deadline - clock())))


def verified_extract(directory, remote):
    receipt_path, archive_path = directory / 'evidence_receipt.json', directory / 'study-evidence.tar.gz'
    receipt = cases.read(receipt_path)
    cases.require(receipt.get('schema') == 'astra_campaign_local_evidence_receipt_v1'
                  and receipt.get('remote_output') == remote
                  and receipt.get('individual_file_hashes_verified') is True
                  and receipt.get('remote_writes') is False,
                  'Evidence receipt contract/root mismatch')
    cases.require(archive_path.stat().st_size == receipt.get('archive_bytes')
                  and cases.sha(archive_path) == receipt.get('archive_sha256'), 'Archive size/SHA differs')
    index, snapshot = sync.verify_archive(archive_path, remote)
    cases.require(len(index['files']) == receipt.get('evidence_files')
                  and snapshot.get('remote_output') == remote, 'Inventory/snapshot root mismatch')
    expected = {row['path']: row for row in index['files']}
    destination = directory / 'evidence'
    cases.require(not destination.is_symlink(), 'Evidence destination must not be a symlink')
    destination.mkdir(exist_ok=True)
    root = destination.resolve()
    for present in destination.rglob('*'):
        cases.require(not present.is_symlink() and present.resolve().is_relative_to(root),
                      'Existing evidence path escapes extraction root')
        if present.is_file():
            cases.require(present.relative_to(destination).as_posix() in expected
                          or present.name == sync.EVIDENCE_INDEX, 'Unexpected pre-existing evidence file')
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            rel = PurePosixPath(member.name)
            cases.require(member.isfile() and not rel.is_absolute() and '..' not in rel.parts,
                          'Unsafe archive member')
            target = destination.joinpath(*rel.parts)
            cases.require(target.resolve().is_relative_to(root) and not target.is_symlink(),
                          'Archive destination escapes root')
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.extractfile(member).read()
            digest = hashlib.sha256(raw).hexdigest()
            if member.name in expected:
                cases.require(digest == expected[member.name]['sha256']
                              and len(raw) == expected[member.name]['bytes'], 'Archive changed during extraction')
            if target.exists():
                cases.require(cases.sha(target) == digest, 'Existing evidence file differs; preserved for review')
            else:
                with target.open('xb') as stream:
                    stream.write(raw)
    cases.require(cases.sha(archive_path) == receipt['archive_sha256'], 'Archive changed during extraction')
    return destination, dict(receipt_sha256=cases.sha(receipt_path), archive_sha256=receipt['archive_sha256'],
                             verified_files=len(index['files']), snapshot=snapshot)


def broker_active(owner_path):
    """Read only the recorded local Windows PID/start identity; no signals."""
    owner = cases.read(owner_path)
    pid = owner.get('pid')
    cases.require(type(pid) is int and pid > 0, 'Broker owner PID invalid')
    start = owner.get('start_time', owner.get('start_utc'))
    cases.require(isinstance(start, str), 'Broker start identity missing')
    command = ("$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "';"
               "if($null -eq $p){'null'}else{[pscustomobject]@{"
               "start=$p.CreationDate.ToUniversalTime().ToString('o');command=$p.CommandLine}|ConvertTo-Json -Compress}")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, timeout=20,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    cases.require(result.returncode == 0, 'Local broker identity inspection failed')
    actual = json.loads(result.stdout)
    if actual is None:
        return False
    parse = lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parse(start) != parse(actual['start']):
        return False
    if owner.get('wrapper'):
        cases.require(owner['wrapper'] in (actual.get('command') or ''), 'Broker command identity mismatch')
    return True


def wait_broker(owner_path, seconds=180, clock=time.monotonic, sleeper=time.sleep, check=broker_active):
    deadline = clock() + seconds
    while True:
        try:
            if check(owner_path) is False:
                return dict(inactive=True, usage_provisional=False, checked_utc=sync.utc())
            error = None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        if clock() >= deadline:
            return dict(inactive=False, usage_provisional=True, checked_utc=sync.utc(),
                        reason=error or 'Broker still active after bounded 180 second wait')
        sleeper(min(POLL_SECONDS, max(0, deadline - clock())))


def build_result(v1_directory, continuation_directory, pin, finalization):
    cases.require(v1_directory is not None, 'v1 archive was not verified; do not admit existing extracted files')
    result = cases.aggregate(HERE / 'failure_manifest.json', HERE / 'continuation_selection.json',
                             HERE / 'continuation_selection_receipt.json', v1_directory,
                             continuation_directory, pin)
    result['finalization'] = finalization
    result['original_benchmark'] = dict(episodes=2500, successes=1496, failures=1004, unchanged=True)
    result['experiment_complete'] = bool(finalization['execution_coverage_complete']
        and result['summary']['classification']['unknown'] == 0
        and result['summary']['classification']['initial_state_deviation'] == 0)
    result['official_benchmark_score_emitted'] = False
    return result


def report_text(result, usage_result):
    summary = result.get('summary') or {}
    final = result['finalization']
    labels = summary.get('classification')
    return ('# OriginX Astra 救援复测报告\n\n'
        + '本报告由本地收尾脚本根据已验证证据生成；未改论文、未公开发布。原冻结基准保持 1496/2500（59.84%），失败子集固定 1004。\n\n'
        + f"交付状态：{final['status']}。全量尝试覆盖：{final['execution_coverage_complete']}；完整实验结果：{result['experiment_complete']}。\n\n"
        + (f"固定 1004 分类：救回 {labels['rescued']}、未救回 {labels['not_rescued']}、未知 {labels['unknown']}、初始状态偏差 {labels['initial_state_deviation']}。\n\n" if labels else '分类证据校验失败，计数未知；未将缺失证据转换成失败或成功。\n\n')
        + (f"严格 R={summary['strict_rescued_R']}；匹配完整回放 M={summary['matched_complete_pairs_M']}；确实应用 Astra 且证据有效的配对 E={summary['verified_astra_matched_pairs_E']}。R/1004={summary['strict_R_over_1004']}，R/M={summary['strict_R_over_M']}，R/E={summary['strict_R_over_E']}。M 可包含未成功应用 Astra 的技术未知案例。raw paired={summary['raw_paired']}。\n\n" if labels else '')
        + 'v1 首次 claim 的 36 个案例与 continuation 的 968 个案例按冻结分区唯一聚合；未执行、未知与初始状态偏差均保留。pilot 独立列示，不混入主研究结果。条件比值和阶段下界不构成新官方基准成绩。\n\n'
        + f"用量状态：{'provisional' if usage_result.get('provisional') else 'settled_local_evidence'}；主研究唯一 CLI 调用={usage_result.get('primary', {}).get('summary', {}).get('unique_cli_calls', 'unknown')}，pilot={usage_result.get('pilot', {}).get('summary', {}).get('unique_cli_calls', 'unknown')}。失败及未应用的调用仍计入；缺失 token 保留未知，美元 cost=null。详细 token/延迟见独立 CLI 用量 JSON。\n\n"
        + f"收尾错误或限制：{final.get('errors', [])}。cleanup 仅继承归档记录，未在此脚本中独立远端核验。\n")


def finalize(directory, output, run_seconds, broker_owner):
    finished = wait_sync(directory, run_seconds)
    errors, admitted, pin, provenance = [], None, None, {}
    v1 = None
    coverage = False
    try:
        v1, provenance['v1'] = verified_extract(HERE / 'full_sync', sync.PROJECT + '/results/astra-native-reset-full-20261009-v1')
        if finished.get('status') == 'evidence_collected':
            evidence, provenance['continuation'] = verified_extract(directory, REMOTE)
            cases.require(sync.terminal_reason(provenance['continuation']['snapshot'])
                          == finished.get('terminal_observation'), 'Sync terminal reason differs from archived snapshot')
            binding = cases.read(HERE / 'broker_state_continuation_v2_launch2/continuation-binding.json')
            pin = binding['config_sha256']
            cases.require(binding.get('selected_cases') == 968 and binding.get('episodes') == 1936,
                          'Frozen broker binding denominators differ')
            provenance['binding_sha256'] = cases.sha(HERE / 'broker_state_continuation_v2_launch2/continuation-binding.json')
            admitted = evidence
            completion = cases.read(evidence / 'completion.json') if (evidence / 'completion.json').exists() else {}
            report = cases.read(evidence / 'analysis/report.json') if (evidence / 'analysis/report.json').exists() else {}
            v1completion = cases.read(v1 / 'completion.json')
            coverage = (completion.get('all_children_drained') is True and completion.get('episodes') == 1936
                        and completion.get('attempt_coverage_complete') is True
                        and report.get('attempt_coverage_complete') is True and report.get('planned_episodes') == 1936
                        and v1completion.get('all_children_drained') is True and v1completion.get('episodes') == 72)
        else:
            errors.append('Continuation sync did not collect verified terminal evidence: ' + str(finished.get('status')))
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
        admitted, pin, coverage = None, None, False
    final = dict(created_utc=sync.utc(), status='verified_terminal_evidence' if admitted else 'failed_or_unknown_continuation',
                 execution_coverage_complete=coverage, sync_finished=finished, errors=errors,
                 provenance=provenance, remote_actions=False, llm_calls=False, publicly_uploaded=False)
    try:
        result = build_result(v1, admitted, pin, final)
    except (OSError, ValueError, KeyError) as exc:
        final['errors'].append('Classification aggregation rejected: ' + str(exc))
        final.update(status='failed_or_unknown_continuation', execution_coverage_complete=False)
        admitted = None
        try:
            result = build_result(v1, None, None, final)
        except (OSError, ValueError, KeyError) as fallback:
            final['errors'].append('v1 fallback rejected: ' + str(fallback))
            result = dict(schema='astra_failed_or_unknown_delivery_v1', fixed_denominator=1004,
                          summary=None, experiment_complete=False, finalization=final,
                          original_benchmark=dict(episodes=2500, successes=1496, failures=1004, unchanged=True))
    broker = wait_broker(broker_owner)
    final['broker_observation'] = broker
    primary = [HERE / name for name in ('broker_state_full_v1', 'broker_state_full_v1_after_reset_1',
                                        'broker_state_continuation_v2_launch2')]
    try:
        usage_result = usage.aggregate(primary, [HERE / 'broker_state_pilot_v1'])
        usage_result['provisional'] = broker['usage_provisional'] or not admitted
    except (OSError, ValueError, KeyError) as exc:
        usage_result = dict(schema='astra_usage_failed_or_unknown_v1', provisional=True, cost_usd=None,
                            error=type(exc).__name__ + ': ' + str(exc), remote_actions=False)
    usage_result['broker_observation'] = broker
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'OriginX_Astra救援复测结果.json'
    usage_path = output / 'OriginX_Astra救援CLI用量最终.json'
    report_path = output / 'OriginX_Astra救援复测报告.md'
    cases.require(not any(p.exists() for p in (result_path, usage_path, report_path)),
                  'Final output already exists; preserved without overwrite')
    write(result_path, result, exclusive=True)
    write(usage_path, usage_result, exclusive=True)
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write(report_text(result, usage_result))
    receipt = dict(stage='local_delivery_written', created_utc=sync.utc(), status=final['status'],
                   experiment_complete=result['experiment_complete'], usage_provisional=usage_result['provisional'],
                   outputs={str(p): cases.sha(p) for p in (result_path, usage_path, report_path)}, errors=final['errors'])
    write(directory / 'finalizer-status.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync-directory', type=Path, default=HERE / 'continuation_sync')
    parser.add_argument('--output-directory', type=Path, default=OUTPUT)
    parser.add_argument('--broker-owner', type=Path, default=HERE / 'broker_continuation_v2.launch2.owner.json')
    parser.add_argument('--run-seconds', type=int, default=36000)
    args = parser.parse_args()
    cases.require(1 <= args.run_seconds <= 36000, 'Bounded wait must be 1..36000 seconds')
    for path in (args.sync_directory.resolve(), args.output_directory.resolve()):
        cases.require(path.drive.upper() == 'D:', 'Local artifacts must be on D:')
    cases.require(not args.output_directory.resolve().is_relative_to(HERE),
                  'Keep final outputs outside script, broker and synchronized source directories')
    print(json.dumps(finalize(args.sync_directory.resolve(), args.output_directory.resolve(),
                              args.run_seconds, args.broker_owner.resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
