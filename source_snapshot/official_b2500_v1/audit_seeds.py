"""CPU-only B2500 seed audit; never inspect outcome fields or launch work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

MANIFEST_SHA = None
MIN_SEED, MAX_SEED = 2040500000, 2040502499
# Excludes data/runtime code and training-only RNG, never an evaluation namespace.
SKIP = {'envs', 'data', 'code', 'runtime', 'cache', 'models', 'training', 'exports',
        '.git', '__pycache__'}
NAME = re.compile(r'(manifest|design|plan|schedule|assignment|seed|initial_scene|first_request|fixture|bindings|job|claim|dispatch|launch|protocol)', re.I)
SCALAR = re.compile(r'"[^"\n]*seed[^"\n]*"\s*:\s*(\d+)', re.I)
ARRAY = re.compile(r'"[^"\n]*seed[^"\n]*"\s*:\s*\[([\d,\s]+)\]', re.I)
USAGE_DIRS = {'claims', 'processes', 'episodes', '.sealed'}
MARKERS = ('official-b2500-v1', 'official_b2500_v1', 'lora-formal2500', 'lora_formal2500')


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    global MANIFEST_SHA
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    a = p.parse_args();root = a.root.resolve(strict=True);output = a.output.resolve()
    MANIFEST_SHA = a.manifest_sha256
    manifest = a.manifest.resolve(strict=True)
    if sha(manifest.read_bytes()) != MANIFEST_SHA: raise ValueError('Engineering manifest SHA differs')
    protocol=a.protocol.resolve(strict=True)
    if sha(protocol.read_bytes()) != a.protocol_sha256: raise ValueError('Protocol SHA differs')
    future_files={manifest:MANIFEST_SHA,protocol:a.protocol_sha256}
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
                if path.resolve() in future_files and digest == future_files[path.resolve()] and not in_claim:
                    exclusions.append({'path':str(path),'sha256':digest,
                        'reason':'Exact known frozen future schedule source, not an execution receipt; associated claims/usage are separately inventoried.'})
                    future.append(str(path));continue
                if digest == '2476f1011fd97d6379ab60e3d2c6940cd3690c2982e988eb0b57a47e39f2e3de' and file=='manifest.json' and any(k in str(relative) for k in ('lora-formal2500','lora_formal2500')):
                    exclusions.append({'path':str(path),'sha256':digest,'reason':'Exact prior prospective2500 manifest. Any formal claims/processes/episode usage anywhere under old namespaces blocks admission independently.'})
                    continue
                if digest in ['dd9bdf3bcf31f34c0c69038fa7470cafbbad4edf5747a2bf0c273951136ea540'] and file=='protocol.json' and any(k in str(relative) for k in ('lora-formal2500','lora_formal2500')):
                    exclusions.append({'path':str(path),'sha256':digest,'reason':'Exact prior prospective protocol; all execution usage remains a hard block.'})
                    continue
                # Old audit reports repeat candidate seed ranges; they are audit
                # provenance, not evidence that any episode consumed that seed.
                if 'audit' in file.lower() and '"scope"' in text and '"collisions"' in text:
                    obj=json.loads(text)
                    if (obj.get('manifest_sha256') in (MANIFEST_SHA, '2476f1011fd97d6379ab60e3d2c6940cd3690c2982e988eb0b57a47e39f2e3de') and obj.get('passed') is True and obj.get('collisions')==[] and not obj.get('actual_cohort_usage_paths') and
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
        'official_b2500_v1','results/official-b2500-v1/cohort','results/official-b2500-v1']}
    report={'kind':'official_B2500_current_metadata_seed_audit_v1','observed_unix':time.time(),
        'passed':not collisions and not errors and not usage,'remote_execution_metadata_checked':True,'within_manifest_shared_seed_blocks':2500,
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
