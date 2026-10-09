"""Verify the released original episode records without simulation or a GPU."""
import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / 'evidence/final-audit.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == receipt['archive_sha256'], 'Archive hash mismatch'
    prefix = 'results/native-reset-b2500-20261009-v1/'
    remote_root = '/ephemeral/qinzhen/robocasa-xr1-20261003/'
    with tarfile.open(args.archive, 'r:gz') as tar:
        def read(name):
            return tar.extractfile(prefix + name).read()
        report = json.loads(read('analysis/report.json'))
        manifest = json.loads(read('manifest.json'))
        assert report['complete'] and not report['smoke'] and len(manifest['jobs']) == 2500
        assert hashlib.sha256(read('config.json')).hexdigest() == report['config_sha256']
        assert hashlib.sha256(read('manifest.json')).hexdigest() == report['manifest_sha256']
        task_counts, task_wins = collections.Counter(), collections.Counter()
        for job in manifest['jobs']:
            name = 'episodes/' + job['id'] + '/result.json'
            content = read(name)
            key = remote_root + prefix + name
            assert hashlib.sha256(content).hexdigest() == report['result_sha256'][key]
            result = json.loads(content)
            assert result['job'] == job and result['status'] == 'completed'
            assert type(result['success']) is bool
            assert result['native_reset_before'] == result['native_reset_after']
            assert result['native_reset_before']['native_reset']
            assert not result['native_reset_before']['reset_patch_applied']
            ep = result['stats']['episodes'][0]
            assert result['success'] == ep['success'] and ep['seed'] == job['env_seed']
            assert result['success'] or ep['steps'] == job['horizon']
            task_counts[job['task']] += 1
            task_wins[job['task']] += result['success']
        assert len(task_counts) == 50 and set(task_counts.values()) == {50}
        for task in task_counts:
            assert task_wins[task] == report['tasks'][task]['successes']
        assert sum(task_wins.values()) == report['successes'] == 1496
        assert abs(report['overall_task_mean'] - 1496 / 2500) < 1e-12
    print(json.dumps({'verified':True, 'episodes':2500, 'successes':1496, 'failures':1004, 'success_rate':0.5984}))

if __name__ == '__main__':
    main()
