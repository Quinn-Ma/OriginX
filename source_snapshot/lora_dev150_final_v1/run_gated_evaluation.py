"""Execute preflight, engineering gate, then fixed dev450; stop on any failure."""
import json,os,subprocess,time
from pathlib import Path
from . import runner as r
R=Path(__file__).resolve().parent.parent
D=R/'results/lora-dev150-final-v1'
def main():
 bpath=D/'bindings.json';b=r.read(bpath);env=r.environment(b)
 owner=dict(r.identity(),started_unix=time.time(),bindings_sha256=r.sha(bpath),source_sha256=r.sha(__file__))
 r.new_json(D/'evaluation-chain-owner.json',owner)
 manifest=R/'lora_dev150_final_v1/dev450-manifest.json'
 common=[b['python'],'-u',str(R/'lora_dev150_final_v1/runner.py')]
 options=['--bindings',str(bpath),'--manifest',str(manifest),'--arms','base','L','A','--parallel','48']
 gateout=R/'results/engineering/lora-dev150-final-gate-v1'
 stages=[('preflight',common+['preflight']+options),('engineering_gate',[b['python'],'-u','-m','lora_dev150_final_v1.engineering_gate','--bindings',str(bpath),'--bindings-sha256',r.sha(bpath),'--manifest',str(R/'lora_dev150_final_v1/gate_manifest.json'),'--seed-audit',str(D/'engineering-seed-audit.json'),'--seed-audit-sha256','55c91c02b4472d33be506efd64542c04fe73a1058deb9c79669e31e47511b174','--output',str(gateout)]),('dev450',common+['execute']+options+['--output',str(D/'cohort/dev450')])]
 for name,cmd in stages:
  if name=='dev450':
   gate=r.read(gateout/'report.json');r.require(gate['passed'] is True and gate['bindings_sha256']==r.sha(bpath),'Engineering gate did not pass')
   r.require(r.read(D/'services-ready.json')['ready'] is True,'Services not ready')
  print(json.dumps(dict(stage=name,event='start',unix=time.time())),flush=True)
  with (D/(name+'-supervisor.log')).open('x') as log:
   child=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   ticks=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
   r.new_json(D/(name+'-supervisor-owner.json'),dict(pid=child.pid,process_start_ticks=ticks,command=cmd,cwd=str(R),started_unix=time.time()))
   code=child.wait()
  r.new_json(D/(name+'-exit.json'),dict(returncode=code,ended_unix=time.time()))
  print(json.dumps(dict(stage=name,event='exit',code=code,unix=time.time())),flush=True)
  if code:return code
 r.new_json(D/'evaluation-chain-complete.json',dict(completed_unix=time.time(),all450_complete=True,score_emitted=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
