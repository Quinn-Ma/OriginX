"""Read historical execution metadata only, not scored episodes or sealed outcomes."""
import argparse, hashlib, json, os
from pathlib import Path
MANIFEST_SHA='6817fdd613e1fb5cb5a365bb09ac2d4661a07cecdf74633a9e7c8527255e605f'
LOW,HIGH=2060700000,2060700149
SKIP_DIRS={'.git','.sealed','envs','cache','code','data','models','exports','__pycache__','logs','returns','offline_diagnostics','deps','videos','episodes'}
TOKENS=('manifest','plan','design','launch','binding','config','claim','dispatch','protocol','freeze','schedule')

def seed_values(value,path=''):
    if isinstance(value,dict):
        for k,v in value.items():
            child=path+'/'+str(k)
            if 'seed' in str(k).lower():
                yield from numbers(v,child)
            elif isinstance(v,(dict,list)):yield from seed_values(v,child)
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from seed_values(v,path+'/'+str(i))
def numbers(value,path):
    if type(value)is int:yield path,value
    elif isinstance(value,dict):
        for k,v in value.items():yield from numbers(v,path+'/'+str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from numbers(v,path+'/'+str(i))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve(strict=True);files=[];collisions=[];errors=[];excluded=[];total=0
    for directory,dirs,names in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
        for name in sorted(names):
            path=Path(directory)/name
            if path.suffix not in ('.json','.jsonl') or not any(x in name.lower() for x in TOKENS) and path.parent.name!='claims':continue
            if path.resolve()==a.output.resolve():continue
            if path.resolve() in {root/'continuous_eval2000_v3/protocol.json'}:
                excluded.append(dict(path=str(path),reason='current prospective protocol',sha256=hashlib.sha256(path.read_bytes()).hexdigest()));continue
            try:
                raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                if digest==MANIFEST_SHA:
                    excluded.append(dict(path=str(path),reason='exact prospective manifest',sha256=digest));continue
                values=[json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix=='.jsonl' else [json.loads(raw)]
                pairs=[pair for v in values for pair in seed_values(v)];total+=len(pairs)
                files.append(dict(path=str(path),sha256=digest,bytes=len(raw),seed_fields=len(pairs)))
                collisions.extend(dict(path=str(path),field=k,value=v) for k,v in pairs if LOW<=v<=HIGH)
            except Exception as e:errors.append(dict(path=str(path),type=type(e).__name__,error=str(e)))
    report=dict(schema='stage_dev150_v2_remote_seed_audit',passed=not errors and not collisions and bool(files),
        manifest_sha256=MANIFEST_SHA,environment_inference_seed_range=[LOW,HIGH],within_manifest_shared_seed_blocks=150,
        remote_execution_metadata_checked=True,collisions=collisions,errors=errors,files=files,seed_fields_checked=total,
        scope=dict(root=str(root),filename_tokens=TOKENS,also='all claims/*.json',excluded_directory_names=sorted(SKIP_DIRS),
            excluded_exact_prospective_manifests=excluded,execution_host=os.uname().nodename,
            results_and_sealed_episode_outcomes_read=False,all_matching_metadata_files_scanned=True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),passed=report['passed'],files=len(files),seed_fields=total,collisions=len(collisions),errors=len(errors))))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
