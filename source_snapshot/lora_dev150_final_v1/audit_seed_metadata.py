"""CPU-only fixed450 seed audit: inspect metadata, never parse outcome fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

MANIFEST_SHA = '8643051dab874f68ca8763242265ad3016788c4da62af3f57060004fe5144fe1'
MIN_SEED, MAX_SEED = 2040200000, 2040200449
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();root = a.root.resolve(strict=True);output = a.output.resolve()
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
                if file == 'dev450-manifest.json' and digest == MANIFEST_SHA and not in_claim:
                    exclusions.append({'path':str(path),'sha256':digest,
                        'reason':'Exact known frozen future schedule source, not an execution receipt; associated claims/usage are separately inventoried.'})
                    future.append(str(path));continue
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
    report={'kind':'fixed450_current_metadata_seed_audit_v2','observed_unix':time.time(),
        'passed':not collisions and not errors and not usage,
        'manifest_sha256':MANIFEST_SHA,'environment_inference_seed_range':[MIN_SEED,MAX_SEED],
        'collisions':collisions,'errors':errors,'actual_cohort_usage_paths':sorted(set(usage)),
        'old_dev450_no_claims_or_results_found':not usage,
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
