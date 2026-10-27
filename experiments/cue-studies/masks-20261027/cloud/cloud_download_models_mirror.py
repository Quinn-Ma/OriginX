"""Exact frozen public weights; direct server download with per-file SHA admission."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib, json, os, shutil, time, urllib.request

ROOT = Path('/root/originx-models')
MANIFEST = ROOT / 'models_manifest.json'
START = time.monotonic()
DEADLINE = START + 3600

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def one(role, config, item):
    name = Path(item['destination_relative']).name
    dest = ROOT / role / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size == item['bytes'] and digest(dest) == item['sha256']:
            return {'role': role, 'name': name, 'status': 'verified_reuse', 'bytes': item['bytes'], 'sha256': item['sha256']}
        raise ValueError('existing final asset differs: ' + str(dest))
    part = dest.with_name(dest.name + '.partial')
    url = 'https://hf-mirror.com/' + config['repo'] + '/resolve/' + config['revision'] + '/' + name
    began = time.monotonic()
    for attempt in range(3):
        if time.monotonic() > DEADLINE:
            raise TimeoutError('bounded download deadline')
        offset = part.stat().st_size if part.exists() else 0
        headers = {'User-Agent': 'originx-frozen-baseline/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url + '?download=true', headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if offset and response.status != 206:
                    offset = 0
                with part.open('ab' if offset else 'wb') as out:
                    last = time.monotonic()
                    while True:
                        if time.monotonic() > DEADLINE:
                            raise TimeoutError('bounded download deadline')
                        chunk = response.read(8 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        if time.monotonic() - last >= 20:
                            last = time.monotonic()
                            print(json.dumps({'event': 'progress', 'role': role, 'name': name,
                                              'bytes': out.tell(), 'target': item['bytes'], 'utc': now()}), flush=True)
            if part.stat().st_size != item['bytes']:
                raise ValueError('download length differs for ' + name)
            actual = digest(part)
            if actual != item['sha256']:
                raise ValueError('frozen SHA mismatch for ' + name)
            os.replace(part, dest)
            row = {'role': role, 'name': name, 'status': 'verified_download', 'bytes': item['bytes'],
                   'sha256': actual, 'seconds': time.monotonic() - began, 'source': url}
            print(json.dumps(row), flush=True)
            return row
        except Exception as exc:
            print(json.dumps({'event': 'retry_or_fail', 'role': role, 'name': name, 'attempt': attempt + 1,
                              'error_type': type(exc).__name__, 'error': str(exc)}), flush=True)
            if 'SHA mismatch' in str(exc) or attempt == 2:
                raise
            time.sleep(2)

def main():
    manifest = json.loads(MANIFEST.read_text())
    need = sum(i['bytes'] for c in manifest.values() for i in c['files'])
    if shutil.disk_usage(ROOT).free < need + (4 << 30):
        raise RuntimeError('system disk lacks model bytes plus 4 GiB reserve')
    rows, errors = [], []
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(one, r, c, i): (r, i['destination_relative'])
                   for r, c in manifest.items() for i in sorted(c['files'], key=lambda x: x['bytes'])}
        for f in as_completed(futures):
            try:
                rows.append(f.result())
            except Exception as exc:
                errors.append({'asset': futures[f], 'error': str(exc)})
            value = {'schema': 'originx-cloud-frozen-model-download-v1', 'updated_utc': now(),
                     'status': 'complete' if len(rows) == len(futures) else 'running',
                     'verified_files': len(rows), 'expected_files': len(futures), 'files': rows, 'errors': errors,
                     'model_identity': {r: {k: c[k] for k in ('repo', 'revision', 'checkpoint_sha256')} for r, c in manifest.items()},
                     'elapsed_seconds': time.monotonic() - START, 'free_bytes': shutil.disk_usage(ROOT).free}
            tmp = ROOT / 'download_receipt.json.tmp'
            tmp.write_text(json.dumps(value, indent=2))
            os.replace(tmp, ROOT / 'download_receipt.json')
    if errors:
        raise RuntimeError(str(errors))

if __name__ == '__main__':
    main()
