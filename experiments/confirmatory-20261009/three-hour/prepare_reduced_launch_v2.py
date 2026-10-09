"""One-use v2 preparation after both prior batches end. No model or CLI starts."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
OLD=ROOT/'results/originx-confirmatory-20261009-v1'
V1=ROOT/'results/originx-confirmatory-multigpu-20261009-v1'
SERVICES=ROOT/'results/originx-confirmatory-multigpu-services-20261009-v2'
OUT=ROOT/'results/originx-three-hour-amendment-20261009-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def run(cmd,label):
 with (OUT/(label+'-command.json')).open('x') as f:json.dump(dict(command=cmd),f)
 result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=300)
 (OUT/(label+'.stdout.log')).write_text(result.stdout);(OUT/(label+'.stderr.log')).write_text(result.stderr)
 assert result.returncode==0,result.stderr[-3000:]
 return json.loads(result.stdout)
def main():
 p=argparse.ArgumentParser();p.add_argument('--pins-sha256',required=True);p.add_argument('--duration-seconds',type=int,required=True);a=p.parse_args()
 pins_path=OUT/'v2-source-pins.json';assert sha(pins_path)==a.pins_sha256;pins=read(pins_path)
 for name,digest in pins.items():assert sha(ROOT/name)==digest,(name,'source changed')
 for previous in [OLD,V1]:
  assert read(previous/'cleanup.json')['all_owned_services_stopped'] is True
  assert read(previous/'completion.json')['all_children_drained'] is True
 assert 600<=a.duration_seconds<=5400
 python=str(ROOT/'envs/training/bin/python');script=str(ROOT/'continue_multigpu_v2.py');debit=OUT/'continuation-debit-v2.json'
 terminal=OLD/'broker_inventory_final/terminal-inventory.json';v1terminal=V1/'broker_inventory_final/terminal-inventory.json'
 cmd=[python,'-B',script,'debit','--output',str(debit)]
 for flag,x in [('broker-terminal',terminal),('v1-broker-terminal',v1terminal)]:cmd+=['--'+flag,str(x),'--'+flag+'-sha256',sha(x)]
 cmd+=['--broker-state',str(terminal.parent),'--v1-broker-state',str(v1terminal.parent)]
 usage=run(cmd,'combined-debit-v2')
 profiles=[];groups=[]
 from continue_multigpu_v2 import owner_inactive
 for gpu in (3,6):
  g=SERVICES/('gpu'+str(gpu));gate=read(g/'ready.json')
  assert gate['passed'] is True and gate['same_socket_action_rng_parity_passed'] is True and gate['hold_seconds']>=360
  assert gate['helper_sha256']==pins['multigpu_services_v2.py']
  assert sha(g/'services-ready.json')==gate['services_ready_sha256'] and sha(Path(gate['parity_path']))==gate['parity_sha256']
  assert not (g/'draining.json').exists() and not owner_inactive(read(g/'owner.json'))
  ms=read(g/'services-ready.json')['models'];assert len(ms)==6
  for m in ms:assert not owner_inactive(m['owner']) and m['gpu_index']==gpu and sha(Path(m['server_manifest']))==m['server_manifest_sha256']
  profiles.extend(ms);groups.append(ref(g/'ready.json'))
 assert len({m['port'] for m in profiles})==12
 ready=SERVICES/'combined-services-ready.json';admission=SERVICES/'combined-service-admission.json'
 write(ready,dict(ready=True,models=profiles,model_count=12,clients=72,groups=groups,namespace='originx_confirmatory_20261009'))
 write(admission,dict(schema='originx_multigpu_service_admission_v1',passed=True,services_ready_sha256=sha(ready),same_socket_action_rng_parity_passed=True,hold_seconds=360,groups=groups,helper_sources=pins))
 cmd=[python,'-B',script,'prepare','--duration-seconds',str(a.duration_seconds)]
 for flag,x in [('services-ready',ready),('service-admission',admission),('broker-terminal',terminal),('v1-broker-terminal',v1terminal),('broker-source',ROOT/'broker_multigpu_v2.py'),('debit',debit)]:cmd+=['--'+flag,str(x),'--'+flag+'-sha256',sha(x)]
 cmd+=['--broker-state',str(terminal.parent),'--v1-broker-state',str(v1terminal.parent)]
 result=run(cmd,'prepare-continuation-v2');print(json.dumps(dict(result=result,debit=usage)))
if __name__=='__main__':main()
