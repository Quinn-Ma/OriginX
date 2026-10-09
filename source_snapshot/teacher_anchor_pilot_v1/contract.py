"""Frozen recipe/sample evidence checks; no model load or GPU operations."""
import hashlib
import json
import random
from pathlib import Path

PLAN_SHA256 = '80521193d0b52ed198d28feaff6d6fd4a78c1835545705b63a23b81004d043ef'
CONTROL_IDENTITY_SHA256 = '49051830631f96a40fd94e60374d726d1325ddddaf18553a00ee5bedec7fd5ff'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def regenerate_trace(manifest, lengths, count=1600):
    rng = random.Random(manifest['seed'])
    result = []
    for index in range(count):
        dataset = manifest['datasets'][rng.randrange(len(manifest['datasets']))]
        episode = dataset['episode_ids'][rng.randrange(len(dataset['episode_ids']))]
        length = lengths[dataset['task']][episode]
        if type(length) is not int or length <= 0:
            raise ValueError('Invalid episode length')
        result.append(dict(task=dataset['task'], episode=episode,
                           timestep=rng.randrange(length), draw=index + 1))
    return result


def control_trace(metrics):
    rows = [json.loads(line) for line in Path(metrics).read_text().splitlines()]
    steps = [row for row in rows if 'samples' in row]
    if [row['step'] for row in steps] != list(range(1, 201)) or any(len(row['samples']) != 8 for row in steps):
        raise ValueError('Expected all200 control updates with8 samples each')
    return [sample for row in steps for sample in row['samples']], steps


def verify_inputs(plan_path, manifest_path, control_run):
    if sha(plan_path) != PLAN_SHA256:
        raise ValueError('Frozen method plan differs')
    plan = read(plan_path)
    control_run = Path(control_run)
    if sha(control_run / 'run_identity.json') != CONTROL_IDENTITY_SHA256:
        raise ValueError('Original control identity differs')
    control = read(control_run / 'run_identity.json')
    if sha(manifest_path) != plan['data']['raw_manifest_sha256']:
        raise ValueError('Original Human300 manifest differs')
    manifest = read(manifest_path)
    if canonical_sha(manifest) != plan['data']['canonical_manifest_sha256'] or manifest != control['manifest']:
        raise ValueError('Control manifest is not exactly reproduced')
    if sha(control_run / 'metrics.jsonl') != plan['data']['control_metrics_sha256']:
        raise ValueError('Original control sample metrics differ')
    trace, steps = control_trace(control_run / 'metrics.jsonl')
    if canonical_sha(trace) != plan['data']['control_sample_trace_sha256']:
        raise ValueError('Original1600 sample trace differs')
    lengths = {}
    inventories = {item['task']: item for item in control['identity']['data']}
    for dataset in manifest['datasets']:
        task = dataset['task']
        path = Path(dataset['root']) / 'meta/episodes.jsonl'
        if sha(path) != inventories[task]['provenance']['metadata_sha256']['meta/episodes.jsonl']:
            raise ValueError('Episode metadata differs from original control: ' + task)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        lengths[task] = {row['episode_index']: row['length'] for row in rows}
    if regenerate_trace(manifest, lengths) != trace:
        raise ValueError('Regenerated1600 sample IDs differ; do not reuse old control as matched arm')
    return plan, manifest, control, trace, steps


def verify_bridge_sources(control, directory):
    for name, digest in control['identity']['training_code'].items():
        if sha(Path(directory) / name) != digest:
            raise ValueError('Control training source differs: ' + name)


def anchored_manifest(manifest, contract):
    # Preserve the existing export/checkpoint format while binding every new
    # objective/source to the run identity through its canonical manifest hash.
    result = dict(manifest)
    result['teacher_anchor_pilot'] = contract
    return result
