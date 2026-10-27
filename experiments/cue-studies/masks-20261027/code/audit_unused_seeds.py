"""Prospective seed exclusion for the point/masks validation; CPU only.

Copied from work/id60-regions-validation/audit_unused_seeds.py
(sha256 b3253152201ed92c4dfcd850628e1b776048b6aaf36a631464253394dd6b7657) and
adapted per work/id60-masks-design/harness-map-v1.md section 11.1:

1. The regions validation namespace (id60_regions_validation /
   id60-regions-validation) is no longer excluded: its seeds are now used and
   must count as prior evidence. The masks validation namespace
   (id60_masks_validation / id60-masks-validation) is excluded instead, by
   directory glob and by path substring, to avoid circular contamination.
2. Receipt bug fix: selected_seed_prior_matches reads evidence.get(seed, []).
   The original evidence[seed] on a defaultdict inserted the selected seeds,
   so excluded_prior_seed_or_token_values also listed them.
3. Explicit fail-closed disjointness from the development seeds (20261025-27)
   and from the regions validation seeds, which are read from the regions
   receipt and its manifest and cross-checked against 20261031 + range(12).
4. Command-line interface with --help and --count (12, 24 or 36 seeds; the
   default 12 equals the original rule). The receipt records the count.

Unchanged: the scanned roots, cache/asset exclusions, JSON seed-like keys, the
conservative date-like TOKEN rule, the 8 MiB size cap, and the rule "first
COUNT integers >= 20261028 absent from all evidence".

Needs ripgrep: 'rg' on PATH or --rg PATH. Neither the Windows shell PATH nor
WSL Ubuntu-22.04 currently provides rg; a Windows rg.exe can be passed with
--rg (also from WSL via interop; output separators are normalised). The
receipt records the rg executable/version and platform. It refuses to
overwrite an existing receipt. Never imports experiment,
model or simulator modules and never reads arrays or journals.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('unused-seeds-v1.json')
NAMESPACE = ('id60_masks_validation', 'id60-masks-validation')
EXCLUDE = {'envs','env','.venv','venv','vendor','weights','checkpoints','node_modules',
    'site-packages','__pycache__','.git','.pytest_cache','.cache','hf-cache','deployment',
    'package-check','minimal-assets','assets','cache','uv-cache','tf-source','tf-source-v2',
    'tf-target-v2','tmp-awomo','vlabench-env','VLABench','shared-deps','data','restore',
    'asset-pack',*NAMESPACE}
KEYS={'seed','seeds','random_seed','random_seeds','rng_seed','episode_seed','ordered_seeds','selected_seeds'}
TOKEN=re.compile(r'(?<!\d)(20\d{6})(?!\d)')
BACKSLASH=chr(92)
START_SEED = 20261028
ALLOWED_COUNTS = (12, 24, 36)
DEVELOPMENT_SEEDS = (20261025, 20261026, 20261027)
REGIONS_FIRST_SEED, REGIONS_COUNT = 20261031, 12
REGIONS_RECEIPT = 'work/id60-regions-validation/unused-seeds-v1.json'
REGIONS_MANIFEST = 'work/id60-regions-validation/manifest.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

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

def in_namespace(text):
    return any(name in text for name in NAMESPACE)

def keep(name):
    """Path filter applied to rg output (relative path); mirrors the original L47-53.

    Backslashes are normalised so a Windows rg.exe (native or via WSL interop) and a
    Linux rg produce the same candidate set."""
    p=Path(name.replace(BACKSLASH,'/')); text=p.as_posix()
    if in_namespace(text):return False
    if p.parts[0]=='work' and not (len(p.parts)>1 and p.parts[1].startswith('id60')):return False
    if p.parts[0]=='runs':
        # All run registration and per-case identity metadata, no large trace journal.
        if p.suffix!='.json' or not (any(t in p.name.lower() for t in ('manifest','registration','amendment'))
                or 'results' in p.parts or p.name=='trace.json'):return False
    return True

def rg_command(rg='rg'):
    command=[rg,'--files','runs','work','outputs','configs','scripts','originx_harness','tests']
    for name in sorted(EXCLUDE): command+=['-g',f'!**/{name}/**']
    command+=['-g','!**/snapshot-*/**','-g','!**/backup-rehearsal-verified/**',
        '-g','!**/test-tmp/**','-g','!**/pytest_tmp/**','-g','!**/pytest-of-*/**',
        '-g','!**/test-*/**','-g','!**/local-check*/**','-g','!**/audit-script-selftest*/**',
        '-g','!**/models--*/**','-g','*.json','-g','*.py','-g','*.yaml','-g','*.yml','-g','*.toml']
    return command

def ripgrep(rg='rg'):
    exe=shutil.which(rg) or (str(Path(rg)) if Path(rg).is_file() else None)
    if exe is None:
        raise RuntimeError(f'ripgrep executable {rg!r} not found; install rg or pass --rg PATH')
    version=subprocess.run([exe,'--version'],check=True,text=True,capture_output=True).stdout.splitlines()
    return exe,(version[0] if version else '')

def candidates(root=ROOT, rg='rg'):
    result=subprocess.run(rg_command(rg),cwd=root,check=True,text=True,capture_output=True)
    names=[name.replace(BACKSLASH,'/') for name in result.stdout.splitlines() if name.strip()]
    return sorted({root/Path(name) for name in names if keep(name)})

def known_used_seeds(root=ROOT):
    """Seeds that must never be selected, independent of the scan (fail closed)."""
    receipt=json.loads((root/REGIONS_RECEIPT).read_text(encoding='utf-8-sig'))
    manifest=json.loads((root/REGIONS_MANIFEST).read_text(encoding='utf-8-sig'))
    regions=list(range(REGIONS_FIRST_SEED,REGIONS_FIRST_SEED+REGIONS_COUNT))
    if receipt.get('ordered_seeds')!=regions or manifest.get('seeds')!=regions:
        raise ValueError('Regions validation seeds differ from the recorded receipt/manifest')
    return {'development':list(DEVELOPMENT_SEEDS),'regions_validation':regions}

def select(evidence, count=12, start=START_SEED, forbidden=()):
    """First `count` integers >= start with no evidence; never mutates `evidence`."""
    if count not in ALLOWED_COUNTS:raise ValueError(f'count must be one of {ALLOWED_COUNTS}')
    selected=[]; n=start
    while len(selected)<count:
        if n not in evidence:selected.append(n)
        n+=1
    clash=sorted(set(selected)&set(forbidden))
    if clash:raise ValueError(f'Selected seeds overlap used development/regions seeds: {clash}')
    return selected

def receipt_matches(evidence, selected):
    # Bug fix vs. the regions audit L83: .get() never inserts keys into the defaultdict.
    return {str(s):list(evidence.get(s,[])) for s in selected}

def audit(count=12, root=ROOT, rg='rg'):
    exe,version=ripgrep(rg)
    used=known_used_seeds(root)
    forbidden=set(used['development'])|set(used['regions_validation'])
    scanned=[]; skipped=[]; evidence=defaultdict(list); history=[]
    for p in candidates(root,exe):
        rel=p.relative_to(root).as_posix(); size=p.stat().st_size
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
    selected=select(evidence,count,START_SEED,forbidden)
    matches=receipt_matches(evidence,selected)
    absent=all(v==[] for v in matches.values())
    if not absent:raise AssertionError('Selected seed has prior evidence')
    unseen_regions=[s for s in used['regions_validation'] if s not in evidence]
    if unseen_regions:
        # The regions namespace is no longer excluded, so its seeds must be visible to the scan.
        raise AssertionError(f'Scan scope misses used regions seeds {unseen_regions}')
    return {'status':'PASS','purpose':'prospective_masks_validation_seed_exclusion',
        'selected_count':count,'start_seed':START_SEED,
        'ripgrep':{'executable':exe,'version':version},'platform':sys.platform,
        'ordered_seeds':selected,'selected_seeds':selected,'all_selected_absent':absent,
        'selected_seed_prior_matches':matches,
        'selection_rule':f'First {count} integer seeds >={START_SEED} absent from audited prior seed fields and conservative date-like source tokens; integers are RNG seeds, not calendar dates.',
        'forbidden_used_seeds':used,'selected_disjoint_from_forbidden_used_seeds':not (set(selected)&forbidden),
        'regions_seeds_absent_from_scan_evidence':unseen_regions,
        'previous_seed_fields':sorted({f['seed'] for row in scanned for f in row['seed_fields']}),
        'excluded_prior_seed_or_token_values':sorted(evidence),
        'history_manifests':history,'scanned_files':scanned,'scanned_file_count':len(scanned),
        'skipped_files':skipped,'excluded_directory_names':sorted(EXCLUDE),
        'excluded_path_substrings':list(NAMESPACE),
        'scope':['All runs small manifest/registration/amendment/results/trace.json files after documented cache/asset exclusion.',
            'Unique original60 final-verified metadata included; redundant snapshots/backup rehearsal excluded.',
            'Prior ID60 work/source, including the completed regions validation, and repository-owned configs/scripts/tests/output JSON; conservative integer tokens also exclude merely mentioned seeds.',
            'Only the masks validation namespace itself is excluded; masks pilot runs/drivers and other masks files outside that namespace are scanned (conservative).'],
        'limits':['Absence is project-evidence scoped, not a claim about external or deleted experiments.',
            'No records.jsonl, NPZ, images, model packages or third-party assets inspected.',
            'Current masks validation source/work excluded; no outcome-dependent seed replacement allowed.',
            'Historical development samples are excluded from prospective denominators.'],
        'derived_from_audit_sha256':'b3253152201ed92c4dfcd850628e1b776048b6aaf36a631464253394dd6b7657',
        'audit_script_sha256':sha(Path(__file__)),'new_model_generations':0,'new_env_resets':0,'new_env_steps':0,
        'outcome_adaptive_seed_replacement_allowed':False}

def main(argv=None):
    parser=argparse.ArgumentParser(description='Select prospectively unused RNG seeds for the point/masks '
        'validation and write work/id60-masks-validation/unused-seeds-v1.json (never overwrites). '
        'Needs ripgrep: rg on PATH, or --rg PATH (a Windows rg.exe also works from WSL).')
    parser.add_argument('--count',type=int,choices=ALLOWED_COUNTS,default=12,
        help='number of seeds (= registered pairs) to select; default 12')
    parser.add_argument('--rg',default='rg',help='ripgrep executable (default: rg on PATH)')
    args=parser.parse_args(argv)
    if OUT.exists():raise FileExistsError(OUT)
    result=audit(args.count,rg=args.rg);OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':result['status'],'ordered_seeds':result['ordered_seeds'],
        'scanned_files':result['scanned_file_count'],'history_manifests':len(result['history_manifests']),
        'skipped_files':len(result['skipped_files']),'receipt_sha256':sha(OUT)}))

if __name__=='__main__':
    main()
