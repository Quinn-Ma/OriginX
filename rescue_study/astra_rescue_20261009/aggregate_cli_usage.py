"""Read only local broker batches; count each CLI invocation once, including errors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

TOKENS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(calls):
    usage = {}
    for token in TOKENS:
        known = [row['usage'].get(token) for row in calls if row['usage'].get(token) is not None]
        require(all(type(n) is int and n >= 0 for n in known), 'Invalid token counter')
        usage[token] = dict(known_sum=sum(known), calls_with_counter=len(known),
                            calls_missing_counter=len(calls) - len(known),
                            complete_total=sum(known) if len(known) == len(calls) else None)
    latency = [row['latency_seconds'] for row in calls if row['latency_seconds'] is not None]
    require(all(type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in latency),
            'Invalid latency')
    return dict(unique_cli_calls=len(calls), ok_calls=sum(x['status'] == 'ok' for x in calls),
                error_calls=sum(x['status'] == 'error' for x in calls),
                unknown_status_calls=sum(x['status'] not in ('ok', 'error') for x in calls),
                calls_with_verified_applied_case=sum(x['verified_applied_case_count'] > 0 for x in calls)
                if all(x['verified_applied_case_count'] is not None for x in calls) else None,
                usage=usage, latency_seconds=dict(known_calls=len(latency), missing_calls=len(calls)-len(latency),
                sum=sum(latency), mean=statistics.mean(latency) if latency else None,
                median=statistics.median(latency) if latency else None,
                minimum=min(latency) if latency else None, maximum=max(latency) if latency else None),
                cost_usd=None)


def collect(directories, scope, verified_ids=None):
    calls, by_batch, by_sha, skipped, references = [], {}, {}, [], []
    for directory in directories:
        directory = Path(directory).resolve()
        require(directory.is_dir(), 'Missing broker directory: ' + str(directory))
        for batchdir in sorted((directory / 'batches').glob('*')):
            if not batchdir.is_dir():
                continue
            batch_path, receipt_path = batchdir / 'batch.json', batchdir / 'receipt.json'
            require(batch_path.is_file(), 'Batch manifest missing: ' + str(batchdir))
            batch = read(batch_path)
            batch_id = batch.get('batch_id')
            require(isinstance(batch_id, str) and batch_id == batchdir.name, 'Batch identity mismatch')
            receipt = read(receipt_path) if receipt_path.is_file() else {}
            if receipt:
                require(receipt.get('schema') == 'astra_cli_receipt_v1', 'Unknown CLI receipt schema')
                require(receipt.get('batch_manifest_sha256') == sha(batch_path), 'CLI/batch hash mismatch')
            if receipt.get('cli_executed') is not True and not (batchdir / 'cli_started.json').is_file():
                skipped.append(dict(batch_id=batch_id, reason='no_cli_execution_evidence'))
                continue
            receipt_sha = sha(receipt_path) if receipt else None
            existing = by_batch.get(batch_id) or (by_sha.get(receipt_sha) if receipt_sha else None)
            if existing is not None:
                require(existing['batch_manifest_sha256'] == sha(batch_path)
                        and existing['cli_receipt_sha256'] == receipt_sha,
                        'Conflicting copies of one batch/CLI identity')
                existing['source_directories'].append(str(directory))
                continue
            request_ids = receipt.get('request_ids', [r['request_id'] for r in batch.get('requests', [])])
            require(len(request_ids) == len(set(request_ids)), 'Duplicate request in one CLI call')
            row = dict(scope=scope, batch_id=batch_id, cli_receipt_sha256=receipt_sha,
                       batch_manifest_sha256=sha(batch_path), source_directories=[str(directory)],
                       status=receipt.get('status', 'unknown_receipt_missing'),
                       error_kind=receipt.get('error_kind'), started_at=receipt.get('started_at'),
                       finished_at=receipt.get('finished_at'), request_ids=request_ids,
                       usage={key: receipt.get('usage_totals', {}).get(key) for key in TOKENS},
                       latency_seconds=receipt.get('latency_seconds'), cost_usd=None,
                       verified_applied_case_count=sum(x in verified_ids for x in request_ids)
                       if verified_ids is not None else None)
            calls.append(row)
            by_batch[batch_id] = row
            if receipt_sha:
                by_sha[receipt_sha] = row
        for path in sorted((directory / 'receipts').glob('*.json')):
            reference = read(path)
            if reference.get('cli_receipt_sha256'):
                references.append((path, reference))
    for path, reference in references:
        row = by_batch.get(reference.get('batch_id'))
        require(row is not None and row['cli_receipt_sha256'] == reference['cli_receipt_sha256']
                and reference.get('request_id') in row['request_ids'],
                'Request receipt does not resolve to an admitted unique CLI call: ' + str(path))
    return dict(calls=calls, summary=summarize(calls), request_receipt_references_checked=len(references),
                nonexecuted_batches=skipped)


def aggregate(primary_directories, pilot_directories, classification=None):
    verified = None
    if classification is not None:
        classified = read(classification)
        require(classified.get('schema') == 'astra_partitioned_joint_classification_v1'
                and len({row['case_id'] for row in classified['cases']}) == 1004,
                'Expected unique partitioned 1004-case classification')
        verified = {row['case_id'] + '--astra' for row in classified['cases']
                    if row.get('verified_astra_matched_pair') is True}
    primary = collect(primary_directories, 'primary', verified)
    pilot = collect(pilot_directories, 'pilot_development')
    primary_shas = {row['cli_receipt_sha256'] for row in primary['calls']} - {None}
    pilot_shas = {row['cli_receipt_sha256'] for row in pilot['calls']} - {None}
    require(not (primary_shas & pilot_shas), 'One CLI receipt was assigned to both pilot and primary')
    require(not ({row['batch_id'] for row in primary['calls']} & {row['batch_id'] for row in pilot['calls']}),
            'One batch was assigned to both pilot and primary')
    return dict(schema='astra_unique_cli_usage_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                primary=primary, pilot=pilot, classification_sha256=sha(classification) if classification else None,
                cost_usd=None, read_only_sources=True, publicly_uploaded=False,
                notes=['CLI invocations deduplicated by receipt SHA or batch identity; per-request references never multiply usage.',
                       'All evidenced CLI invocations count, including quota errors and responses not applied to a valid matched case.',
                       'Missing token counters stay unknown; known_sum is a lower bound, not a complete total.',
                       'Latency is measured once per CLI invocation, not per request, queue delay or full rollout.',
                       'Input/cache/output/reasoning counters are kept separate; potentially overlapping counters are not added together.',
                       'No API dollar cost is inferred for authenticated CLI quota.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-broker-dir', type=Path, action='append', default=[])
    parser.add_argument('--pilot-broker-dir', type=Path, action='append', default=[])
    parser.add_argument('--classification', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.primary_broker_dir, 'At least one explicit primary broker directory is required')
    output = args.output.resolve()
    require(output.drive.upper() == 'D:', 'Output must be on D:')
    for directory in args.primary_broker_dir + args.pilot_broker_dir:
        require(not output.is_relative_to(directory.resolve()), 'Do not write inside input broker evidence')
    result = aggregate(args.primary_broker_dir, args.pilot_broker_dir, args.classification)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output),
                         primary=result['primary']['summary'], pilot=result['pilot']['summary'])))


if __name__ == '__main__':
    main()
