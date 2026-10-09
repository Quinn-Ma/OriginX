"""Start isolated v2 development once, with parent-owned launch handshake."""
from pathlib import Path
import argparse,json,subprocess
HERE=Path(__file__).resolve().parent
ROOT='/ephemeral/qinzhen/robocasa-xr1-20261003'
NS='originx_gr00t_confirmation_20261009_v2'
SERVICE=ROOT+'/originx_gr00t_confirmation_20261009'
p=argparse.ArgumentParser();p.add_argument('--profile-sha256',required=True);p.add_argument('--parity-sha256',required=True);a=p.parse_args()
for digest in (a.profile_sha256,a.parity_sha256):
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise RuntimeError('Malformed pinned SHA')
freeze=json.loads((HERE.parent/NS/'source-freeze.remote.json').read_text())
code='''from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(%r);base=root/%r;service=Path(%r);sys.path.insert(0,str(root))
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=base/'source-freeze.json'
if digest(f)!=%r:raise RuntimeError('New source freeze changed')
for name,expected in json.loads(f.read_text())['source_sha256'].items():
 if digest(base/name)!=expected:raise RuntimeError('Frozen module changed: '+name)
from originx_gr00t_confirmation_20261009_v2.runner import environment
from originx_gr00t_confirmation_20261009_v2.processes import TrackedChild
profile=service/'services/port-27902/server.json';parity=service/'parity-socket-v3/result.json'
if digest(profile)!=%r or digest(parity)!=%r:raise RuntimeError('Service/parity binding differs')
if json.loads(parity.read_text()).get('passed') is not True:raise RuntimeError('Actual parity must pass')
ref=root/'results/native-reset-b2500-20261009-v1/config.json';cfg=json.loads(ref.read_text())
manifest=root/'originx_confirmatory_20261009/shared-reference/fresh-manifest.json'
if digest(manifest)!='ef3b1fb0492e0d6136ab4aa0e24193bcfbbd00865e0339a26ee0e2851b04648a':raise RuntimeError('Reference changed')
receipt=root/'originx_confirmatory_20261009/gr00t-development-launch-v2.owner.json'
if receipt.exists() or receipt.with_suffix('.intent.json').exists() or (root/'results/originx-gr00t-confirmatory-development-20261009-v2').exists():raise RuntimeError('Prior v2 attempt exists; no duplicate')
env=environment(cfg)
cmd=[cfg['python'],'-B','-u','-m',base.name+'.campaign','--development',
 '--reference-config',str(ref),'--reference-config-sha256',digest(ref),
 '--reference-manifest',str(manifest),'--reference-manifest-sha256',digest(manifest),
 '--service-manifest',str(profile),'--service-manifest-sha256',digest(profile),
 '--source-freeze',str(f),'--source-freeze-sha256',digest(f),
 '--checkpoint',str(service/'checkpoint-120000'),'--parity-result',str(parity),'--parity-result-sha256',digest(parity)]
with receipt.with_suffix('.intent.json').open('x') as stream:json.dump(dict(command=cmd,source_freeze_sha256=digest(f),unix=time.time()),stream,indent=2)
with receipt.with_suffix('.log').open('xb') as log:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 tracked=TrackedChild(child,cmd,root)
try:
 owner=tracked.await_identity()
 campaign_receipt=base/'campaigns/originx-gr00t-confirmatory-development-20261009-v2/owner.json'
 deadline=time.monotonic()+10
 while True:
  if child.poll() is not None or time.monotonic()>=deadline:raise RuntimeError('Campaign did not publish its own owner')
  try:
   declared=json.loads(campaign_receipt.read_text())
   break
  except (OSError,json.JSONDecodeError):time.sleep(.05)
 if child.poll() is not None or any(declared.get(k)!=owner[k] for k in ('pid','process_start_ticks','command','cwd')):raise RuntimeError('Campaign owner handoff differs')
 with receipt.open('x') as stream:json.dump(dict(owner,development=True,unix=time.time()),stream,indent=2)
 with receipt.with_suffix('.spawn.json').open('x') as stream:json.dump(tracked.receipt(),stream,indent=2)
 # Ownership deliberately passes to the separately recorded campaign, whose
 # signal handling and fixed deadline supervise all descendants.
 if tracked.pidfd is not None:os.close(tracked.pidfd);tracked.pidfd=None
 print(json.dumps(dict(owner,receipt=str(receipt),planned_cases=2,planned_arm_outcomes=4,prior_v1_preserved=True)))
except BaseException:
 tracked.drain(grace=45)
 with receipt.with_suffix('.failed-spawn.json').open('x') as stream:json.dump(tracked.receipt(),stream,indent=2)
 raise
'''%(ROOT,NS,SERVICE,freeze['source_freeze_sha256'],a.profile_sha256,a.parity_sha256)
p=subprocess.run(['ssh','-o','BatchMode=yes','hkust-cluster',ROOT+'/envs/sim/bin/python','-B','-'],input=code.encode(),capture_output=True,timeout=120)
print(p.stdout.decode(errors='replace'));print(p.stderr.decode(errors='replace'))
if p.returncode:raise SystemExit(p.returncode)
with (HERE/'gr00t-development-v2-launch.json').open('xb') as f:f.write(p.stdout)
