"""Local terminal broker host attestation. Never invokes a teacher or rollout."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
SPRINT=HERE.parent.parent
STATE=SPRINT/'gr_broker_v2/state'
SEAL=SPRINT/'gr_broker_seal'
REMOTE='/ephemeral/qinzhen/robocasa-xr1-20261003/originx_sprint_gr_final_usage_20261011_v1'
sys.path.insert(0,str(HERE.parent/'v2'))
from broker import ProcessLock,local_owner,local_owner_active
STOP=datetime.datetime.fromisoformat('2026-10-11T18:25:18+00:00').timestamp()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v,new=True):
    with Path(p).open('x' if new else 'w',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
def attest():
    owner=read(STATE/'owner.json')
    if local_owner_active(owner) or not (SEAL/'complete.json').exists():return None
    complete=read(SEAL/'complete.json');local=read(SEAL/'local-seal.json')
    if local.get('broker_inactive') is not True or local['broker_owner']!=owner:raise RuntimeError('Broker inactive seal identity differs')
    if complete.get('all_file_hashes_verified') is not True or complete['remote']['archive_sha256']!=local['archive_sha256']:raise RuntimeError('Sealer transfer incomplete')
    files={p.relative_to(STATE).as_posix():sha(p) for p in STATE.rglob('*') if p.is_file()}
    if files!=local['files'] or files!=complete['remote']['files']:raise RuntimeError('Broker terminal inventory changed after seal')
    if read(STATE/'broker_status.json').get('state') not in ('finished','stopped'):raise RuntimeError('Nonterminal broker status')
    if local_owner_active(owner):raise RuntimeError('Broker owner became active during attestation')
    return dict(broker_inactive=True,broker_owner=owner,all_file_hashes_verified=True,remote_seal=complete['remote'],
        source_sha256=sha(__file__),checked_unix=time.time(),local_seal_sha256=sha(SEAL/'local-seal.json'),sealer_complete_sha256=sha(SEAL/'complete.json'))
def main():
    with ProcessLock(HERE/'waiter.lock'):
        if (HERE/'waiter.owner.json').exists():raise RuntimeError('Waiter owner exists; never duplicate')
        write(HERE/'waiter.owner.json',dict(local_owner(),source_sha256=sha(__file__)))
        while time.time()<STOP:
            try:
                value=attest()
                if value is None:
                    write(HERE/'waiter.status.json',dict(phase='waiting_actual_broker_exit_and_seal',unix=time.time()),False);time.sleep(30);continue
                path=HERE/'broker-host-attestation.json';write(path,value)
                target=REMOTE+'/broker-host-attestation.staged.json'
                subprocess.run(['scp','-q',str(path),'hkust-cluster:'+target],check=True,timeout=180)
                code=f'''from pathlib import Path
import hashlib,json
p=Path({target!r});final=p.with_name('broker-host-attestation.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha(path)!r}
assert not final.exists(),'Attestation already exists; no overwrite'
p.rename(final)
print(json.dumps(dict(path=str(final),sha256=hashlib.sha256(final.read_bytes()).hexdigest())))
'''
                result=subprocess.run(['ssh','-o','BatchMode=yes','hkust-cluster','python3 -'],input=code.encode(),capture_output=True,timeout=60)
                if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
                write(HERE/'waiter.completion.json',dict(passed=True,remote=json.loads(result.stdout),unix=time.time()))
                write(HERE/'waiter.status.json',dict(phase='terminal_host_proof_uploaded',unix=time.time()),False);return 0
            except Exception:
                write(HERE/'waiter.error.json',dict(error=traceback.format_exc(),unix=time.time()))
                write(HERE/'waiter.status.json',dict(phase='blocked_or_failed',unix=time.time()),False);return 1
        write(HERE/'waiter.status.json',dict(phase='deadline_without_terminal_seal',unix=time.time()),False);return 1
if __name__=='__main__':sys.exit(main())
