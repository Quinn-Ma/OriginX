"""Prospective seed exclusion from project metadata/source; CPU only.

Never imports experiment/model/simulator modules or reads arrays/journals.
The new validation namespace is excluded to avoid circular contamination.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('unused-seeds-v1.json')
EXCLUDE = {'envs','env','.venv','venv','vendor','weights','checkpoints','node_modules',
    'site-packages','__pycache__','.git','.pytest_cache','.cache','hf-cache','deployment',
    'package-check','minimal-assets','assets','cache','uv-cache','tf-source','tf-source-v2',
    'tf-target-v2','tmp-awomo','vlabench-env','VLABench','shared-deps','data','restore',
    'asset-pack','id60_regions_validation','id60-regions-validation'}
KEYS={'seed','seeds','random_seed','random_seeds','rng_seed','episode_seed','ordered_seeds','selected_seeds'}
TOKEN=re.compile(r'(?<!\d)(20\d{6})(?!\d)')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def fields(value, where='$'):
    result=[]
    if isinstance(value,dict):
        for k,v in value.items():
            if k.lower() in KEYS:
                for n in v if isinstance(v,list) else [v]:
                    if type(n) is int and 0<=n<2**32: result.append({'seed':n,'json_path':where+'.'+k})
            result.extend(fields(v,where+'.'+k))
    elif isinstance(value,list):
        for i,v in enumerate(value): result.extend(fields(v,f'{where}[{i}]'))
    return result

def candidates():
    command=['rg','--files','runs','work','outputs','configs','scripts','originx_harness','tests']
    for name in sorted(EXCLUDE): command+=['-g',f'!**/{name}/**']
    command+=['-g','!**/snapshot-*/**','-g','!**/backup-rehearsal-verified/**',
        '-g','!**/test-tmp/**','-g','!**/pytest_tmp/**','-g','!**/pytest-of-*/**',
        '-g','!**/test-*/**','-g','!**/local-check*/**','-g','!**/audit-script-selftest*/**',
        '-g','!**/models--*/**','-g','*.json','-g','*.py','-g','*.yaml','-g','*.yml','-g','*.toml']
    result=subprocess.run(command,cwd=ROOT,check=True,text=True,capture_output=True)
    paths=[]
    for name in result.stdout.splitlines():
        p=Path(name); text=p.as_posix()
        if 'id60_regions_validation' in text or 'id60-regions-validation' in text:continue
        if p.parts[0]=='work' and not (len(p.parts)>1 and p.parts[1].startswith('id60')):continue
        if p.parts[0]=='runs':
            # All run registration and per-case identity metadata, no large trace journal.
            if p.suffix!='.json' or not (any(t in p.name.lower() for t in ('manifest','registration','amendment'))
                    or 'results' in p.parts or p.name=='trace.json'):continue
        paths.append(ROOT/p)
    return sorted(set(paths))

def audit():
    scanned=[]; skipped=[]; evidence=defaultdict(list); history=[]
    for p in candidates():
        rel=p.relative_to(ROOT).as_posix(); size=p.stat().st_size
        if p.is_symlink() or size>8*1024**2:
            skipped.append({'path':rel,'bytes':size,'reason':'symlink_or_over_8MiB'});continue
        try: text=p.read_text(encoding='utf-8-sig')
        except UnicodeError:
            skipped.append({'path':rel,'reason':'not_utf8'});continue
        tokens=sorted({int(t) for t in TOKEN.findall(text)})
        found=[]
        if p.suffix=='.json':
            try:found=fields(json.loads(text))
            except ValueError:pass
        seeds=sorted(set(tokens)|{f['seed'] for f in found})
        row={'path':rel,'bytes':size,'sha256':sha(p),'seed_fields':found,'conservative_tokens':tokens}
        scanned.append(row)
        if any(t in p.name.lower() for t in ('manifest','registration','amendment')):
            history.append({'path':rel,'sha256':row['sha256'],'seed_fields':found})
        for seed in seeds:evidence[seed].append(rel)
    selected=[]; n=20261028
    while len(selected)<12:
        if n not in evidence:selected.append(n)
        n+=1
    return {'status':'PASS','purpose':'prospective_regions_validation_seed_exclusion',
        'ordered_seeds':selected,'selected_seeds':selected,'all_selected_absent':True,
        'selected_seed_prior_matches':{str(s):evidence[s] for s in selected},
        'selection_rule':'First 12 integer seeds >=20261028 absent from audited prior seed fields and conservative date-like source tokens; integers are RNG seeds, not calendar dates.',
        'previous_seed_fields':sorted({f['seed'] for row in scanned for f in row['seed_fields']}),
        'excluded_prior_seed_or_token_values':sorted(evidence),
        'history_manifests':history,'scanned_files':scanned,'scanned_file_count':len(scanned),
        'skipped_files':skipped,'excluded_directory_names':sorted(EXCLUDE),
        'scope':['All runs small manifest/registration/amendment/results/trace.json files after documented cache/asset exclusion.',
            'Unique original60 final-verified metadata included; redundant snapshots/backup rehearsal excluded.',
            'Prior ID60 work/source and repository-owned configs/scripts/tests/output JSON; conservative integer tokens also exclude merely mentioned seeds.'],
        'limits':['Absence is project-evidence scoped, not a claim about external or deleted experiments.',
            'No records.jsonl, NPZ, images, model packages or third-party assets inspected.',
            'Current validation source/work excluded; no outcome-dependent seed replacement allowed.',
            'Historical development samples are excluded from prospective denominators.'],
        'audit_script_sha256':sha(Path(__file__)),'new_model_generations':0,'new_env_resets':0,'new_env_steps':0,
        'outcome_adaptive_seed_replacement_allowed':False}

if __name__=='__main__':
    if OUT.exists():raise FileExistsError(OUT)
    result=audit();OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':result['status'],'ordered_seeds':result['ordered_seeds'],
        'scanned_files':result['scanned_file_count'],'history_manifests':len(result['history_manifests']),
        'skipped_files':len(result['skipped_files']),'receipt_sha256':sha(OUT)}))
