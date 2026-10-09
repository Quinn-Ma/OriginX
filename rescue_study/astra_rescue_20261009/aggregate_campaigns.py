"""Local-only, claim-partitioned aggregation; never runs models or edits sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
LABELS = ('rescued', 'not_rescued', 'unknown', 'initial_state_deviation')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def load_selection(failure_path, selection_path, receipt_path):
    fixed, selection, receipt = read(failure_path), read(selection_path), read(receipt_path)
    require(receipt.get('passed') is True and sha(selection_path) == receipt.get('selection_sha256'),
            'Selection receipt/hash mismatch')
    require(selection.get('schema') == 'astra_claim_partitioned_continuation_selection_v1', 'Wrong selection schema')
    require(sha(failure_path) == selection.get('failure_manifest_sha256') == receipt.get('failure_manifest_sha256'),
            'Frozen failure manifest changed')
    jobs = fixed['jobs']
    ordered = [job['id'] for job in jobs]
    require(len(ordered) == len(set(ordered)) == selection['fixed_case_denominator'] == 1004,
            'Fixed denominator is not exactly 1004 unique cases')
    partition_rows = selection['case_partition']
    require([row['case_id'] for row in partition_rows] == ordered, 'Partition changed original case order')
    partition = {row['case_id']: row['source_partition'] for row in partition_rows}
    require(set(partition.values()) <= {'v1', 'continuation'}, 'Unknown source partition')
    for name in ('v1', 'continuation'):
        expected = [case_id for case_id in ordered if partition[case_id] == name]
        declared = selection[name + '_partition']
        require(declared['case_ids'] == expected and declared['cases'] == len(expected), 'Partition membership/count differs')
    remaining = [job for job in jobs if partition[job['id']] == 'continuation']
    require(selection['remaining_jobs'] == remaining and selection['remaining_assignment_indices'] ==
            [job['assignment_index'] for job in remaining], 'Remaining original jobs or indices changed')
    require(selection['pilot_development']['excluded_from_primary_study'] is False,
            'Pilot development IDs must remain in the primary partition')
    pilot = selection['pilot_development']['cases']
    require(len(pilot) == 4 and len({row['case_id'] for row in pilot}) == 4
            and all(partition[row['case_id']] == row['source_partition'] for row in pilot),
            'Pilot development membership changed')
    return fixed, selection, partition


def load_source(directory, name, fixed, selection, config_pin=None):
    if directory is None or not Path(directory).exists():
        require(name == 'continuation', 'v1 local evidence directory is required')
        return {'name': name, 'available': False, 'rows': {}, 'classification_available': False}
    directory = Path(directory).resolve()
    config_path, manifest_path = directory / 'config.json', directory / 'manifest.json'
    require(config_path.is_file() and manifest_path.is_file(), name + ' config/manifest evidence missing')
    config, manifest = read(config_path), read(manifest_path)
    config_sha, manifest_sha = sha(config_path), sha(manifest_path)
    if name == 'v1':
        require(config_sha == selection['source_evidence']['config_sha256']
                and manifest_sha == selection['source_evidence']['manifest_sha256'],
                'v1 configuration/manifest differ from the claim snapshot')
        expected_selected = fixed['jobs']
    else:
        require(isinstance(config_pin, str) and re.fullmatch(r'[0-9a-f]{64}', config_pin),
                'A pinned --continuation-config-sha256 is required for existing continuation evidence')
        require(config_sha == config_pin, 'Continuation configuration differs from its supplied pin')
        expected_selected = selection['remaining_jobs']
    require(config.get('manifest_sha256') == manifest_sha, name + ' manifest hash differs from configuration')
    require(config.get('failure_manifest_sha256') == selection['failure_manifest_sha256'],
            name + ' references a different frozen 1004-case manifest')
    require(manifest.get('selected_original_jobs') == expected_selected,
            name + ' source manifest selects different original cases/order')
    arms = manifest.get('arms')
    require(isinstance(arms, list) and len(arms) == 2 and set(arms) == {'control', 'astra'},
            name + ' must plan one control and one Astra arm per original case')
    expected_jobs = [dict(job, id=job['id'] + '--' + arm, original_id=job['id'], rescue_arm=arm)
                     for job in expected_selected for arm in arms]
    require(manifest.get('jobs') == expected_jobs, name + ' paired jobs/seed/horizon/order differ')
    if name == 'v1':
        expected_claims = selection['v1_claim_records']
        actual_paths = sorted((directory / 'claims').glob('*.json'))
        require({path.name for path in actual_paths} == {row['episode_id'] + '.json' for row in expected_claims},
                'v1 claims were added/removed after the continuation partition was frozen')
        index = []
        expected_by_episode = {row['episode_id']: row for row in expected_claims}
        for path in actual_paths:
            episode = path.name[:-5]
            expected = expected_by_episode[episode]
            claim = read(path)
            require(sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
                    and claim.get('config_sha256') == config_sha,
                    'v1 claim hash/config differs: ' + episode)
            index.append({'path': expected['path'], 'sha256': expected['sha256'], 'bytes': expected['bytes']})
        index.sort(key=lambda row: row['path'])
        require(canonical_sha(index) == selection['source_evidence']['claims_index_canonical_sha256'],
                'v1 claim index hash differs')
    classification_path = directory / 'analysis/case-classification.json'
    rows = {}
    classification_sha = None
    if classification_path.is_file():
        classification = read(classification_path)
        require(classification.get('schema') == 'astra_strict_case_classification_v1'
                and classification.get('fixed_denominator') == 1004
                and classification.get('selected_denominator') == len(expected_selected),
                name + ' classification schema/denominators differ')
        fixed_ids = {job['id'] for job in fixed['jobs']}
        selected_ids = {job['id'] for job in expected_selected}
        for row in classification['cases']:
            case_id = row.get('case_id')
            require(case_id in fixed_ids and case_id not in rows, name + ' has unknown/duplicate classification ID')
            require(type(row.get('selected')) is bool and row.get('classification') in LABELS,
                    name + ' has invalid selected flag or classification label')
            require(not row['selected'] or case_id in selected_ids, name + ' selects an unplanned case')
            rows[case_id] = row
        classification_sha = sha(classification_path)
    report_path = directory / 'analysis/report.json'
    if report_path.is_file():
        report = read(report_path)
        require(report.get('config_sha256') == config_sha and report.get('manifest_sha256') == manifest_sha,
                name + ' report hashes do not match admitted configuration/manifest')
    return {'name': name, 'available': True, 'directory': str(directory), 'config_sha256': config_sha,
            'manifest_sha256': manifest_sha, 'classification_available': classification_path.is_file(),
            'classification_sha256': classification_sha, 'raw_selected_count': sum(row['selected'] for row in rows.values()),
            'rows': rows}


def pair_flags(row):
    statuses = row.get('arm_status', {})
    valid = all(statuses.get(arm, {}).get('valid_completed') is True for arm in ('control', 'astra'))
    if valid:
        require(type(row.get('control_success')) is bool and type(row.get('astra_success')) is bool,
                'Valid-completed pair lacks boolean outcomes: ' + row['case_id'])
    matched = valid and not row.get('initial_deviation_arms') and row.get('prefix', {}).get('exact') is True
    return valid, matched


def summarize(rows):
    counts = {label: sum(row['classification'] == label for row in rows) for label in LABELS}
    require(sum(counts.values()) == len(rows), 'Four-category identity failed')
    raw_pairs = [row for row in rows if row['raw_complete_pair']]
    matched = sum(row['matched_complete_pair'] for row in rows)
    verified_astra = sum(row['verified_astra_matched_pair'] for row in rows)
    rescued = counts['rescued']
    require(rescued <= verified_astra <= matched <= len(raw_pairs) <= len(rows),
            'Rescue/verified-Astra/matched/raw pair cardinalities differ')
    raw_counts = dict(cases=len(raw_pairs),
                      both_success=sum(row['control_success'] and row['astra_success'] for row in raw_pairs),
                      control_only=sum(row['control_success'] and not row['astra_success'] for row in raw_pairs),
                      astra_only=sum(not row['control_success'] and row['astra_success'] for row in raw_pairs),
                      both_failure=sum(not row['control_success'] and not row['astra_success'] for row in raw_pairs))
    require(sum(raw_counts[key] for key in ('both_success', 'control_only', 'astra_only', 'both_failure')) == len(raw_pairs),
            'Raw-paired outcome identity failed')
    return dict(cases=len(rows), classification=counts, strict_rescued_R=rescued,
                matched_complete_pairs_M=matched, strict_R_over_M=rescued / matched if matched else None,
                verified_astra_matched_pairs_E=verified_astra,
                strict_R_over_E=rescued / verified_astra if verified_astra else None,
                strict_R_over_cases=rescued / len(rows) if rows else None, raw_paired=raw_counts)


def aggregate(failure_path, selection_path, receipt_path, v1_directory, continuation_directory=None,
              continuation_config_sha256=None):
    fixed, selection, partition = load_selection(failure_path, selection_path, receipt_path)
    sources = {
        'v1': load_source(v1_directory, 'v1', fixed, selection),
        'continuation': load_source(continuation_directory, 'continuation', fixed, selection,
                                    continuation_config_sha256),
    }
    metadata = {row['id']: row for row in fixed['failure_records']}
    output_rows = []
    for job in fixed['jobs']:
        case_id = job['id']
        source_name = partition[case_id]
        source = sources[source_name]
        original = source['rows'].get(case_id)
        row = dict(original or {})
        raw_selected = row.get('selected')
        if original is None or raw_selected is not True:
            reason = ('source_classification_missing' if not source['classification_available'] else
                      'expected_source_row_missing' if original is None else 'expected_source_row_not_selected')
            row = dict(case_id=case_id, classification='unknown', reason=reason,
                       control_success=None, astra_success=None, arm_status={}, initial_deviation_arms=[],
                       prefix={'exact': None}, astra_evidence={'valid': False})
        valid_pair, matched_pair = pair_flags(row)
        verified_astra_pair = (matched_pair and row.get('assistance_status') == 'applied'
                               and row.get('astra_evidence', {}).get('valid') is True)
        if row['classification'] == 'rescued':
            require(matched_pair and row.get('control_success') is False and row.get('astra_success') is True
                    and row.get('astra_evidence', {}).get('valid') is True
                    and row.get('assistance_status') == 'applied',
                    'Rescued label lacks strict source evidence flags: ' + case_id)
        row.update(selected=True, effective_selected=True, raw_selected=raw_selected,
                   source=source_name, source_output=source.get('directory'),
                   source_classification_sha256=source.get('classification_sha256'),
                   assignment_index=job['assignment_index'], task=job['task'], stratum=metadata[case_id]['stratum'],
                   raw_complete_pair=valid_pair, matched_complete_pair=matched_pair,
                   verified_astra_matched_pair=verified_astra_pair)
        output_rows.append(row)
    require(len(output_rows) == len({row['case_id'] for row in output_rows}) == 1004,
            'Union is not exactly 1004 unique primary cases')
    summary = summarize(output_rows)
    summary['strict_R_over_1004'] = summary['strict_rescued_R'] / 1004
    by_stratum = {name: summarize([row for row in output_rows if row['stratum'] == name])
                  for name in sorted({row['stratum'] for row in output_rows})}
    by_task = {name: summarize([row for row in output_rows if row['task'] == name])
               for name in sorted({row['task'] for row in output_rows})}
    require(sum(group['cases'] for group in by_stratum.values()) == 1004
            and sum(group['cases'] for group in by_task.values()) == 1004, 'Stratum/task denominators differ')
    source_metadata = {name: {key: value for key, value in source.items() if key != 'rows'}
                       for name, source in sources.items()}
    for name, source in sources.items():
        source_metadata[name]['effective_primary_cases'] = sum(row['source'] == name for row in output_rows)
        source_metadata[name]['raw_selected_rows_ignored_outside_partition'] = sum(
            row['selected'] and partition[case_id] != name for case_id, row in source['rows'].items())
    return dict(schema='astra_partitioned_joint_classification_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                fixed_denominator=1004, selection_sha256=sha(selection_path), failure_manifest_sha256=sha(failure_path),
                source_evidence=source_metadata, summary=summary, by_stratum=by_stratum, by_task=by_task,
                cases=output_rows, pilot_development=selection['pilot_development'],
                checks=dict(unique_primary_partition=True, exactly_1004_cases=True,
                            four_class_counts_sum_to_1004=True,
                            available_source_config_and_manifest_hashes_verified=True,
                            verified_sources=[name for name, source in sources.items() if source['available']]),
                limits=['Local classification aggregation; no new rollout or model call.',
                        'A source row is eligible only in its claim-frozen partition; raw selected=true is not sufficient.',
                        'Missing source classifications or expected rows remain unknown; there is no cross-run fallback.',
                        'Strict labels and recorded evidence flags are checked here; replay/image/CLI artifact audits remain in each source analysis.',
                        'M requires both valid-completed source arms, no initial deviation and exact preintervention prefix.',
                        'M can include technical-unknown pairs where Astra assistance was not successfully applied; it is not the count of verified Astra comparisons.',
                        'E additionally requires assistance_status=applied and valid Astra evidence; R/E is conditional on these verified Astra matched pairs.',
                        'Raw paired outcomes are separate; raw astra_only is never substituted for strict rescue R.',
                        'R/1004 retains unknown and deviation cases; R/M is conditional and neither is a new official benchmark score.',
                        'Tokens, cost and latency are not aggregated by this program.'],
                modifies_source_evidence=False, publicly_uploaded=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failure-manifest', type=Path, default=HERE / 'failure_manifest.json')
    parser.add_argument('--selection', type=Path, default=HERE / 'continuation_selection.json')
    parser.add_argument('--selection-receipt', type=Path, default=HERE / 'continuation_selection_receipt.json')
    parser.add_argument('--v1-dir', type=Path, required=True)
    parser.add_argument('--continuation-dir', type=Path)
    parser.add_argument('--continuation-config-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.drive.upper() == 'D:', 'Aggregate output must be stored on D:')
    for source in (args.v1_dir, args.continuation_dir):
        if source is not None:
            require(not output.is_relative_to(source.resolve()), 'Keep aggregate output outside source evidence directories')
    result = aggregate(args.failure_manifest, args.selection, args.selection_receipt,
                       args.v1_dir, args.continuation_dir, args.continuation_config_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), output_sha256=sha(output), summary=result['summary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
