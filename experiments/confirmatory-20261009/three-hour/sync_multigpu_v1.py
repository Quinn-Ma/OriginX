"""Read-only polling; after terminal workers, request owned group retirement and archive.

No rollout, training or CLI calls. New release requests are only published after
the matching campaign and every recorded worker birth have ended.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, tarfile, time, traceback
from broker import ProcessLock, local_owner, write_json

HERE = Path(__file__).resolve().parent
STATE = HERE / 'sync_multigpu'
ROOT = '/ephemeral/qinzhen/robocasa-xr1-20261003'
NAME = 'originx-confirmatory-multigpu-20261009-v1'
SERVICES = 'originx-confirmatory-multigpu-services-20261009-v1'

REMOTE = r'''
from pathlib import Path
import json,hashlib,time
r=Path('/ephemeral/qinzhen/robocasa-xr1-20261003');name='originx-confirmatory-multigpu-20261009-v1';p=r/'results'/name
def read(f):return json.loads(f.read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def inactive(o):
 try:
  a=(Path('/proc')/str(o['pid'])/'stat').read_text().rsplit(')',1)[1].split()
  return a[0] in ('Z','X') or int(a[19])!=o['process_start_ticks']
 except (FileNotFoundError,ProcessLookupError):return True
def write_new(f,v):
 with f.open('x') as s:json.dump(v,s,indent=2)
d={n:read(p/n) for n in ['campaign-status.json','progress.json','completion.json','campaign-finished.json','campaign-error.json','cleanup.json'] if (p/n).exists()}
if 'campaign-finished.json' in d:
 owners=[p/'campaign.owner.json']+list((p/'processes').glob('*.json'))+list((p/'episodes').glob('*/owner.json'))
 gone=all(inactive(read(f)) for f in owners)
 d['retirement_workers_inactive']=gone
 if gone:
  groups=[]
  for gpu in (3,6):
   g=r/'results/originx-confirmatory-multigpu-services-20261009-v1'/('gpu'+str(gpu))
   release=g/'release-request.json'
   if not release.exists():
    write_new(release,dict(schema='originx_multigpu_service_release_v1',services_ready_sha256=sha(g/'services-ready.json'),all_assigned_workers_stopped=True,campaign_finished_sha256=sha(p/'campaign-finished.json')))
   fin=g/'finished.json';clean=g/'cleanup.json'
   group=dict(gpu=gpu,finished=read(fin) if fin.exists() else None,cleanup=read(clean) if clean.exists() else None)
   group['guardian_inactive']=inactive(read(g/'owner.json'))
   group['models_inactive']=all(inactive(m['owner']) for m in read(g/'services-ready.json')['models'])
   group['verified']=bool(group['finished'] and group['finished']['all_children_reaped'] and group['cleanup'] and group['cleanup']['all_children_reaped'] and group['guardian_inactive'] and group['models_inactive'])
   groups.append(group)
  d['service_retirement']=groups
  if all(g['verified'] for g in groups) and not (p/'cleanup.json').exists():
   write_new(p/'cleanup.json',dict(all_owned_services_stopped=True,still_active_owned_pids=[],worker_owners_all_inactive=True,groups=groups,foreign_processes_untouched=True))
  if (p/'cleanup.json').exists():d['cleanup.json']=read(p/'cleanup.json')
print(json.dumps({name:d}))
'''


def ssh(source, timeout=120):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','hkust-cluster','python3','-'],input=source.encode(),capture_output=True,timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-4000:])
    return json.loads(p.stdout)


def archive():
    dest=STATE/NAME;dest.mkdir(exist_ok=True);receipt=dest/'archive-receipt.json'
    if receipt.exists():return
    code=f'''from pathlib import Path
import json,tarfile,hashlib
r=Path({ROOT!r});p=r/'results'/{NAME!r};a=r/({NAME!r}+'-evidence.tar.gz')
assert json.loads((p/'cleanup.json').read_text())['all_owned_services_stopped'] is True
assert not a.exists(),'Preserve existing archive; inspect before retry'
with tarfile.open(a,'x:gz') as t:
 for q in [p,r/'results'/{SERVICES!r},r/'originx_confirmatory_20261009',r/'continue_multigpu_v1.py',r/'multigpu_services_v1.py',r/'broker_multigpu_v1.py',r/'results/originx-three-hour-amendment-20261009-v1']:
  t.add(q,arcname=q.relative_to(r).as_posix())
h=hashlib.sha256()
with a.open('rb') as f:
 for x in iter(lambda:f.read(8388608),b''):h.update(x)
print(json.dumps(dict(path=str(a),size=a.stat().st_size,sha256=h.hexdigest())))
'''
    meta=ssh(code,1800);local=dest/'evidence.tar.gz'
    subprocess.run(['scp','-q','hkust-cluster:'+meta['path'],str(local)],check=True,timeout=1800)
    h=hashlib.sha256()
    with local.open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    if h.hexdigest()!=meta['sha256'] or local.stat().st_size!=meta['size']:raise RuntimeError('Raw evidence hash mismatch')
    evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
    with tarfile.open(local) as t:
        for m in t.getmembers():
            if m.issym() or m.islnk() or not (evidence/m.name).resolve().is_relative_to(evidence.resolve()):raise RuntimeError('Unsafe archive')
        t.extractall(evidence,filter='data')
    write_json(receipt,dict(**meta,local=str(local),extracted=str(evidence),verified_at=datetime.now(timezone.utc).isoformat()))


def main():
    STATE.mkdir(exist_ok=True)
    with ProcessLock(STATE/'sync.lock'):
        if (STATE/'owner.json').exists():raise RuntimeError('Existing sync owner; inspect instead of duplicate')
        write_json(STATE/'owner.json',local_owner());deadline=time.monotonic()+10800
        while time.monotonic()<deadline:
            try:
                data=ssh(REMOTE);snap=dict(checked_at=datetime.now(timezone.utc).isoformat(),remote=data)
                bs=HERE/'broker_multigpu_v1/broker_status.json'
                if bs.exists():snap['broker_multigpu_v1']=json.loads(bs.read_text(encoding='utf-8'))
                write_json(STATE/'snapshot.json',snap)
                if data[NAME].get('cleanup.json',{}).get('all_owned_services_stopped'):
                    archive();return
            except BaseException:
                write_json(STATE/'error.json',dict(at=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc()))
                if (STATE/NAME/'evidence.tar.gz').exists():raise
            time.sleep(30)
        write_json(STATE/'deadline.json',dict(finished=False,reason='sync three hour limit',no_new_work=True))


if __name__=='__main__':main()
