"""No-LLM status/evidence synchronizer for the new confirmation namespace."""
import argparse,hashlib,json,os,subprocess,sys,tarfile,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from broker import ProcessLock,local_owner,write_json
HERE=Path(__file__).resolve().parent
ROOT='/ephemeral/qinzhen/robocasa-xr1-20261003'
OUTPUT=HERE.parents[1]/'outputs/OriginX_补充实验状态.json'
REMOTE_CODE=r'''
from pathlib import Path
import json
r=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
names=['originx-confirmatory-development-20261009-v2','originx-confirmatory-20261009-v1']
answer={}
for name in names:
 p=r/'results'/name; v={}
 for f in ['campaign-status.json','progress.json','completion.json','campaign-finished.json','campaign-error.json','cleanup.json','cleanup-error.json']:
  q=p/f
  if q.exists():v[f]=json.loads(q.read_text())
 answer[name]=v
p=r/'results/originx-confirmatory-services-20261009-v2/probes/development-v2/result.json'
if p.exists():answer['development_socket_probe']=json.loads(p.read_text())
print(json.dumps(answer))
'''
def remote(code,timeout=90):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','hkust-cluster','python3','-'],input=code.encode(),capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
    return json.loads(p.stdout)
def archive(name):
    dest=HERE/'sync'/name;dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/'archive-receipt.json'
    if receipt.exists():return
    code='''from pathlib import Path
import hashlib,json,tarfile
r=Path(%r);name=%r;p=r/'results'/name
if not (p/'campaign-finished.json').exists():raise RuntimeError('Not terminal')
artifact=r/(name+'-evidence.tar.gz')
if not artifact.exists():
 with tarfile.open(artifact,'x:gz') as t:
  t.add(p,arcname='results/'+name)
  t.add(r/'originx_confirmatory_20261009',arcname='originx_confirmatory_20261009')
  t.add(r/'results/originx-confirmatory-services-20261009-v2',arcname='results/originx-confirmatory-services-20261009-v2')
h=hashlib.sha256()
with artifact.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
print(json.dumps(dict(path=str(artifact),size=artifact.stat().st_size,sha256=h.hexdigest())))
'''%(ROOT,name)
    meta=remote(code,timeout=3600)
    local=dest/'evidence.tar.gz'
    subprocess.run(['scp','-q','hkust-cluster:'+meta['path'],str(local)],check=True,timeout=7200)
    h=hashlib.sha256()
    with local.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    if h.hexdigest()!=meta['sha256'] or local.stat().st_size!=meta['size']:raise RuntimeError('Archive hash mismatch')
    evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
    with tarfile.open(local) as t:
        for m in t.getmembers():
            if m.issym() or m.islnk() or not (evidence/m.name).resolve().is_relative_to(evidence.resolve()):raise RuntimeError('Unsafe archive member')
        t.extractall(evidence,filter='data')
    write_json(receipt,dict(**meta,local=str(local),verified_at=datetime.now(timezone.utc).isoformat(),extracted=str(evidence)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    (HERE/'sync').mkdir(exist_ok=True)
    with ProcessLock(HERE/'sync/sync.lock'):
        write_json(HERE/'sync/owner.json',local_owner())
        deadline=time.monotonic()+6*86400
        while time.monotonic()<deadline:
            try:
                snap=remote(REMOTE_CODE)
                value=dict(schema='originx_confirmatory_local_status_v1',checked_at=datetime.now(timezone.utc).isoformat(),
                    remote=snap,historical_benchmark='1496/2500 unchanged',historical_conditional_rescue='56/1004 unchanged',
                    scope='2500 fresh cases; B C/R/G/L/V and original Xiaomi C/V; GR00T independent setup separately pending',
                    official_rank_verified=False)
                for name in ('broker_development_v2','broker_full_v1'):
                    path=HERE/name/'broker_status.json'
                    if path.exists():value[name]=json.loads(path.read_text())
                write_json(HERE/'sync/snapshot.json',value);write_json(OUTPUT,value)
                for name,data in snap.items():
                    if name.startswith('originx-confirmatory-') and 'campaign-finished.json' in data:archive(name)
                if 'campaign-finished.json' in snap['originx-confirmatory-20261009-v1']:return
                if snap['originx-confirmatory-development-20261009-v2'].get('campaign-finished.json',{}).get('failed'):return
            except Exception:
                write_json(HERE/'sync/error.json',dict(at=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc()))
                if a.once:raise
            if a.once:return
            time.sleep(120)
if __name__=='__main__':main()
