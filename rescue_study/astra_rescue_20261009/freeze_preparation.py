"""Archive local preparation without turning synthetic QA into rollout results."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
OUT = WORKSPACE / 'outputs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = [
        '__init__.py', 'assistance.py', 'runner.py', 'broker.py',
        'launch_models.py', 'build_failure_manifest.py', 'freeze_preparation.py',
        'README.md', 'protocol.md', 'failure_manifest.json', 'manifest_verification.json',
    ]
    sources = [HERE / name for name in names]
    for path in sources:
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    failure = json.loads((HERE / 'failure_manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((HERE / 'manifest_verification.json').read_text(encoding='utf-8'))
    assert len(failure['jobs']) == len({j['id'] for j in failure['jobs']}) == 1004
    assert verification['passed'] is True
    assert sha(HERE / 'failure_manifest.json') == verification['failure_manifest_sha256']
    receipts = [HERE / 'probe/launcher_mock_receipt.json']
    for pattern in ('probe/strict-analysis-qa-*/qa_receipt.json', 'broker_tests/self-test-*/self_test_result.json'):
        matches = sorted(HERE.glob(pattern), key=lambda p: p.stat().st_mtime_ns)
        assert matches, pattern
        receipts.append(matches[-1])
    for path in receipts:
        json.loads(path.read_text(encoding='utf-8'))
    package_files = sources + receipts
    inventory = {
        'schema': 'originx_astra_preparation_archive_v1',
        'real_rescue_episodes': 0,
        'experiment_complete': False,
        'scope': 'Local source, fixed failure manifest, and synthetic QA receipts; not rollout results.',
        'files': {str(p.relative_to(HERE)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                  for p in package_files},
    }
    inventory_path = HERE / 'preparation_inventory.json'
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    archive = OUT / 'OriginX_Astra救援复测执行包.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in package_files + [inventory_path]:
            z.write(path, 'astra_rescue_20261009/' + path.relative_to(HERE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, meta in inventory['files'].items():
            assert hashlib.sha256(z.read('astra_rescue_20261009/' + name)).hexdigest() == meta['sha256']
    receipt = dict(inventory, archive=str(archive), archive_bytes=archive.stat().st_size,
                   archive_sha256=sha(archive), archive_verified=True,
                   source_syntax_checked=True, original_failure_cases=1004)
    (OUT / 'OriginX_Astra救援准备校验.json').write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('archive', 'archive_bytes', 'archive_sha256',
                                           'archive_verified', 'real_rescue_episodes')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
