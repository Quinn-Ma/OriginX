"""Complete readiness for the eight already-owned services; never spawns models."""
import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from .launch_services import R,D,read,save,identity
from .loader import sha,require

def main():
 require(not (D/'bindings.json').exists(),'Bindings already created')
 sys.path.insert(0,str(R/'remote_processor_candidate_v1/deps'))
 rootpins=read(R/'lora_dev150_final_v1/runner-source-pins.json')
 for n,h in rootpins.items():require(sha(R/'lora_dev150_final_v1'/n)==h,'Execution source changed')
 layout=[tuple(x) for x in read(D/'services-launch.json')['layout']]
 parities={a:read(D/(a+'-parity.json')) for a in ['base','L','A']}
 launched=[];model='/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365'
 for i,(arm,g,slots) in enumerate(layout):
  d=D/'services'/f'{arm}-gpu{g}';owner=read(d/'owner.json');o={k:owner[k] for k in ['pid','process_start_ticks','command','cwd']}
  require(identity(o['pid'])==o,'Owned service identity changed')
  def poll(o=o):
   return None if identity(o['pid'])==o else 1
  launched.append((SimpleNamespace(pid=o['pid'],poll=poll),d,arm,g,owner['port'],slots,o['command'],o))
 from multiplex_inference.client import WireClient
 from local_eval.remote_client import validate_endpoint
 from lora_dev150_final_v1.client import validate_adapter_endpoint
 fragments={};mapping={'base':[],'L':[],'A':[]}
 deadline=time.monotonic()+300
 pending=list(launched)
 while pending and time.monotonic()<deadline:
  for item in list(pending):
   p,d,arm,g,port,slots,cmd,o=item;require(p.poll() is None,'Owned service exited: '+str(d))
   if arm!='base' and not (d/'servers.json').exists():continue
   try:c=WireClient(port,timeout=2)
   except (ConnectionRefusedError,TimeoutError):continue
   try:h=c.hello
   finally:c.close()
   parity=parities[arm]
   for key,expected in [('model_assets_sha256',parity['model']['model_assets_sha256']),('official_server_sha256',parity['model']['official_server_sha256']),('multiplex_sources_sha256',parity['source_sha256']),('model_config_runtime_sha256',parity['trace'][0]['actual_rng']['model_config_runtime_sha256'])]:require(h.get(key)==expected,'New service/parity differs '+key)
   require(identity(p.pid)==o,'Service ownership changed')
   if arm=='base':save(d/'servers.json',dict(repo=str(R),python=cmd[0],model=model,ready=True,source_sha256=parity['source_sha256'],parity_report=str(D/'base-parity.json'),parity_sha256=sha(D/'base-parity.json'),hello=h,servers=[dict(pid=p.pid,port=port,gpu=str(g),command=cmd,process_start_ticks=o['process_start_ticks'])],max_clients=8))
   manifest=d/'servers.json';key=f'{arm}-gpu{g}';mapping[arm].append(key)
   fragment=dict(arm=arm,checkpoint_path=model,checkpoint_sha256=__import__('hashlib').sha256(json.dumps(h['model_assets_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),server_manifest=str(manifest),server_manifest_sha256=sha(manifest),parity_report=str(D/f'{arm}-parity.json'),parity_sha256=sha(D/f'{arm}-parity.json'),available_client_slots=slots)
   if arm=='base':validate_endpoint(manifest,D/'base-parity.json',model)
   else:
    b=D/f'{arm}-endpoint.json';fragment.update(endpoint_binding=str(b),endpoint_binding_sha256=sha(b));validate_adapter_endpoint(manifest,D/f'{arm}-parity.json',model,binding_path=b,binding_sha256=sha(b))
   fragments[key]=fragment;pending.remove(item);print(json.dumps({'ready':key,'pid':p.pid}),flush=True)
  if pending:time.sleep(1)
 require(not pending,'Startup timeout; preserve process records for investigation')
 # Device-order routing is fixed independently of readiness order.
 mapping={a:[f'{arm}-gpu{g}' for arm,g,_ in layout if arm==a] for a in ['base','L','A']}
 old=read(R/'human_mg_dev_deploy_v1/base.ready.json')
 b={k:old[k] for k in ['root','repo','package_root','python','osmesa_library','packages','source_sha256']}
 b.update(schema='lora_dev150_final_bindings_v1',approved_for_evaluation=True,manifest_sha256='8643051dab874f68ca8763242265ad3016788c4da62af3f57060004fe5144fe1',runner_pins_sha256=sha(R/'lora_dev150_final_v1/runner-source-pins.json'),cohort_root=str(D/'cohort'),seed_audit={'path':str(D/'seed-audit.json'),'sha256':sha(D/'seed-audit.json')},arm_endpoints=mapping,models=fragments,outcomes_sealed=True,decision_protocol={'path':str(R/'lora_dev150_final_v1/decision-protocol.json'),'sha256':sha(R/'lora_dev150_final_v1/decision-protocol.json')})
 save(D/'bindings.json',b);save(D/'services-ready.json',{'ready':True,'bindings_sha256':sha(D/'bindings.json'),'servers':[{k:o[k] for k in ['pid','process_start_ticks','command','cwd']} for _,_,_,_,_,_,_,o in launched],'reserved_client_slots':52,'planned_simulator_parallel':48,'no_score_episodes_started':True})
 print(json.dumps({'all_services_ready':True,'bindings':str(D/'bindings.json')}),flush=True)
if __name__=='__main__':main()
