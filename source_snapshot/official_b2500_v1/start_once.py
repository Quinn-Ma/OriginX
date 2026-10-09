"""Explicit idempotent entry point. Existing completed/failed owners are preserved."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from official_b2500_v1 import runner
R=Path(__file__).resolve().parent.parent;P=R/'official_b2500_v1';D=R/'results/official-b2500-v1'
def main():
 a=argparse.ArgumentParser();a.add_argument('--authorize',action='store_true',required=True);a.add_argument('--max-model-instances',type=int,default=21);args=a.parse_args();assert 1<=args.max_model_instances<=48
 assert R==Path('/ephemeral/qinzhen/robocasa-xr1-20261003') and D.is_dir()
 with (D/'entry.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name in ['owner.json','start-receipt.json']:
   p=D/name
   if p.exists():
    old=runner.read(p);active=runner.process_active(old)
    if active:
     proc=Path('/proc')/str(old['pid']);actual=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0');assert actual==old['command'] and str((proc/'cwd').resolve())==old['cwd'],'Existing owner changed identity; preserve'
    print(json.dumps(dict(action='already_active' if active else 'prior_owner_terminal_requires_inspection',owner=old,status=runner.read(D/'status.json') if (D/'status.json').exists() else None,new_process_started=False)))
    return 0 if active else 2
  decision=dict(authorized=args.authorize,candidate='B',manifest_sha256=runner.sha(P/'manifest.json'),protocol_sha256=runner.sha(P/'protocol.json'),pins_sha256=runner.sha(P/'runner-source-pins.json'),max_model_instances=args.max_model_instances,unix=time.time(),instruction='User authorized five-hour campaign and official Overall59-60 goal; root chose frozen B2500 after reviewing available budget. No automatic submission.')
  q=D/'launch-decision.json'
  if q.exists():
   old=runner.read(q);assert all(old[k]==decision[k] for k in ['authorized','candidate','manifest_sha256','protocol_sha256','pins_sha256','max_model_instances'])
  else:
   with q.open('x') as f:json.dump(decision,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  cmd=[str(R/'envs/training/bin/python'),'-u','-m','official_b2500_v1.launch'];env=dict(os.environ,PYTHONPATH=str(R)+':'+str(R/'remote_processor_candidate_v1/deps'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  with (D/'driver.log').open('x') as f:process=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+10
  while time.monotonic()<deadline:
   try:
    p=Path('/proc')/str(process.pid);st=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if argv==cmd and st[0] not in ('Z','X'):
     owner=dict(pid=process.pid,process_start_ticks=int(st[19]),command=cmd,cwd=str(R),started_unix=time.time())
     with (D/'start-receipt.json').open('x') as f:json.dump(owner,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
     print(json.dumps(dict(action='started_once',owner=owner)));return 0
   except FileNotFoundError:pass
   if process.poll() is not None:raise RuntimeError('Driver exited before identity capture; preserve driver.log and inspect')
   time.sleep(.05)
  raise RuntimeError('Driver exec identity timed out. Preserve child/log; do not repeat or kill blindly.')
if __name__=='__main__':raise SystemExit(main())
