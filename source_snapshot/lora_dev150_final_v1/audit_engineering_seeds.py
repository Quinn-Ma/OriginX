"""CPU-only engineering gate seed audit; reuse fixed450 metadata-only semantics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

MANIFEST_SHA = None
MIN_SEED, MAX_SEED = 2040200900, 2040200902
# Excludes data/runtime code and training-only RNG, never an evaluation namespace.
SKIP = {'envs', 'data', 'code', 'runtime', 'cache', 'models', 'training', 'exports',
        '.git', '__pycache__'}
NAME = re.compile(r'(manifest|design|plan|schedule|assignment|seed|initial_scene|first_request|fixture|bindings|job|claim|dispatch|launch)', re.I)
SCALAR = re.compile(r'"[^"\n]*seed[^"\n]*"\s*:\s*(\d+)', re.I)
ARRAY = re.compile(r'"[^"\n]*seed[^"\n]*"\s*:\s*\[([\d,\s]+)\]', re.I)
USAGE_DIRS = {'claims', 'processes', 'episodes', '.sealed'}
MARKERS = ('lora-retention-dev450', 'lora-retention-dev150', 'lora-dev150-final')


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    global MANIFEST_SHA
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    a = p.parse_args();root = a.root.resolve(strict=True);output = a.output.resolve()
    MANIFEST_SHA = a.manifest_sha256
    manifest = a.manifest.resolve(strict=True)
    if sha(manifest.read_bytes()) != MANIFEST_SHA: raise ValueError('Engineering manifest SHA differs')
    if output.exists(): raise FileExistsError('Do not overwrite prior audit: ' + str(output))
    sources=[];collisions=[];errors=[];exclusions=[];seed_count=0;usage=[];future=[]
    for parent, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP]
        directory = Path(parent);relative = directory.relative_to(root)
        is_cohort = any(marker in str(relative) for marker in MARKERS)
        if is_cohort:
            for file in files:
                # Inventory only: result/record bytes are never opened here.
                if (any(d in USAGE_DIRS for d in relative.parts)
                        or file in {'records.json','launch.json','dispatch.jsonl','completion.json'}
                        or file.endswith('.lease.json')):
                    usage.append(str(directory / file))
        for file in files:
            path = directory / file
            in_claim = any(d in {'claims','processes'} for d in relative.parts)
            if not file.endswith(('.json','.jsonl')) or not (NAME.search(file) or in_claim): continue
            if path == output: continue
            try:
                content = path.read_bytes();digest=sha(content);text=content.decode('utf-8')
                # Only a byte-identical prospective source manifest is exempt;
                # claims, launch wrappers and any actual usage remain examined.
                if path.resolve() == manifest and digest == MANIFEST_SHA and not in_claim:
                    exclusions.append({'path':str(path),'sha256':digest,
                        'reason':'Exact known frozen future schedule source, not an execution receipt; associated claims/usage are separately inventoried.'})
                    future.append(str(path));continue
                # Byte-verified preexisting aggregate audit repeats the two
                # prospective range endpoints, and references actual old audits.
                if digest == 'e7cb0772c563be5c0c8903aa2ccf6931a0e40fba4252a7a8935b64bbd0b20469':
                    obj=json.loads(text)
                    for receipt in obj['scope']['reports']:
                        if sha(Path(receipt['path']).read_bytes()) != receipt['sha256']:
                            raise ValueError('Aggregate audit source receipt changed')
                    exclusions.append({'path':str(path),'sha256':digest,
                        'reason':'Verified historical aggregate audit, not usage: two numeric hits are prospective range endpoints; cited audit hashes rechecked.'})
                    continue
                # Old audit reports repeat candidate seed ranges; they are audit
                # provenance, not evidence that any episode consumed that seed.
                if 'audit' in file.lower() and '"scope"' in text and '"collisions"' in text:
                    obj=json.loads(text)
                    if (obj.get('manifest_sha256') == MANIFEST_SHA and
                            obj.get('environment_inference_seed_range') == [MIN_SEED,MAX_SEED] and
                            isinstance(obj.get('sources'),list)):
                        exclusions.append({'path':str(path),'sha256':digest,
                            'reason':'Prior metadata audit repeats prospective range; underlying execution receipts are still independently scanned.'})
                        continue
                values=[int(m.group(1)) for m in SCALAR.finditer(text)]
                for match in ARRAY.finditer(text):
                    values.extend(map(int,re.findall(r'\d+',match.group(1))))
                seed_count+=len(values)
                sources.append({'path':str(path),'sha256':digest,'seed_values_found':len(values)})
                for seed in sorted(set(values)):
                    if MIN_SEED<=seed<=MAX_SEED:collisions.append({'path':str(path),'sha256':digest,'seed':seed})
            except Exception as error:errors.append({'path':str(path),'error':repr(error)})
    presence={name:(root/name).exists() for name in [
        'lora_retention_eval_v1','results/dev/lora-retention-dev450-v1',
        'results/dev/lora-dev150-final-v1','results/lora-dev150-final-v1']}
    report={'kind':'engineering_gate_current_metadata_seed_audit_v1','observed_unix':time.time(),
        'passed':not collisions and not errors and not usage,
        'manifest_sha256':MANIFEST_SHA,'environment_inference_seed_range':[MIN_SEED,MAX_SEED],
        'collisions':collisions,'errors':errors,'actual_cohort_usage_paths':sorted(set(usage)),
        'no_named_cohort_claims_or_results_found':not usage,
        'namespace_presence':presence,'prospective_manifest_source_exclusions':exclusions,
        'scope':{'root':str(root),'filename_pattern':NAME.pattern,
            'excluded_directory_names':sorted(SKIP),'only_seed_fields_extracted':True,
            'claims_and_process_metadata_included':True,'outcome_files_not_opened':True,
            'limitations':'Metadata scalar and numeric-array seed fields plus named cohort usage inventories; no data archives or outcome files interpreted. No claim about arbitrary undocumented or externally stored trials.'},
        'sources':sources,'seed_values_found':seed_count,
        'auditor_sha256':sha(Path(__file__).read_bytes()),'GPU_or_simulator_started':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'output':str(output),'sha256':sha(output.read_bytes()),'passed':report['passed'],
        'metadata_files':len(sources),'seed_values':seed_count,'collisions':len(collisions),
        'errors':len(errors),'actual_usage_paths':len(usage),'exclusions':exclusions,'namespace_presence':presence}),flush=True)


if __name__=='__main__':main()
