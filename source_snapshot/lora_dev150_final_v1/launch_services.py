"""Launch only this evaluation's eight GPU services after three parity gates."""
import json,os,resource,socket,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent.parent
D=R/'results/lora-dev150-final-v1'
from lora_dev150_final_v1.loader import sha,require

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def identity(pid):
 p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 require(fields[0] not in ('Z','X'),'Owned service exited')
 return dict(pid=pid,process_start_ticks=int(fields[19]),command=p.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0'),cwd=str(p.joinpath('cwd').resolve()))
def main():
 require(not (D/'services-launch.json').exists(),'Do not duplicate services')
 rootpins=read(R/'lora_dev150_final_v1/runner-source-pins.json')
 for n,h in rootpins.items():require(sha(R/'lora_dev150_final_v1'/n)==h,'Execution source changed')
 parities={a:read(D/(a+'-parity.json')) for a in ['base','L','A']}
 for a,p in parities.items():
  require(p['passed'] is True and len(p['trace'])==6 and p['ambient_rng_unchanged'] and all(t['full_action_exact'] and t['all_rng_exact'] for t in p['trace']),'Actual same-model parity missing '+a)
  o=read(D/(a+'-parity-owner.json'));proc=Path('/proc')/str(o['pid'])
  require(not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z','Parity process still active')
 health=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
 gpu={int(f[0]):[v.strip() for v in f] for line in health.splitlines() if (f:=line.split(','))}
 layout=[('base',0,8),('base',1,8),('L',2,6),('L',3,6),('L',4,6),('A',5,6),('A',6,6),('A',7,6)]
 for _,g,_ in layout:require(gpu[g][3]=='0' and int(gpu[g][2])>=32768,'Need healthy GPU with32GiB free')
 for i in range(8):
  with socket.socket() as s:s.bind(('127.0.0.1',12100+i))
 resource.setrlimit(resource.RLIMIT_NOFILE,(65536,resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
 model='/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365'
 common=dict(os.environ,PYTHONPATH=str(R),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTORCH_NVML_BASED_CUDA_CHECK='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 launched=[]
 save(D/'services-launch.json',{'started_unix':time.time(),'layout':layout,'ports':list(range(12100,12108)),'sources':rootpins,'foreign_processes_untouched':True})
 for i,(arm,g,slots) in enumerate(layout):
  d=D/'services'/f'{arm}-gpu{g}';d.mkdir(parents=True,exist_ok=False);port=12100+i
  if arm=='base':cmd=[str(R/'envs/training/bin/python'),'-u','-m','multiplex_inference.server','--xr1-repo',str(R/'code/Xiaomi-Robotics-1'),'--model',model,'--port',str(port),'--max-clients','8']
  else:
   b=D/f'{arm}-endpoint.json';parity=D/f'{arm}-parity.json'
   cmd=[str(R/'envs/training/bin/python'),'-u','-m','lora_dev150_final_v1.server','--mode','serve','--binding',str(b),'--binding-sha256',sha(b),'--parity',str(parity),'--parity-sha256',sha(parity),'--port',str(port),'--max-clients','8','--server-manifest',str(d/'servers.json')]
  with (d/'server.log').open('x') as f:p=subprocess.Popen(cmd,cwd=R,env=dict(common,CUDA_VISIBLE_DEVICES=gpu[g][1]),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(.05);o=identity(p.pid);save(d/'owner.json',dict(o,gpu=g,gpu_uuid=gpu[g][1],port=port,arm=arm))
  launched.append((p,d,arm,g,port,slots,cmd,o))
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
