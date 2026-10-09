"""One frozen B2500 local benchmark. Existing shared GPU owners preserved. No automatic submission."""
from pathlib import Path
import os,json,time,socket,subprocess,signal,traceback
from official_b2500_v1 import runner
from official_b2500_v1.layout import choose_layout
from stage_dev150_v2.launch import gpu_health,common_env,warmup,alive
R=Path(__file__).resolve().parent.parent;D=R/'results/official-b2500-v1';OLD=R/'results/continued-ab2000-v1';P=R/'official_b2500_v1'
read=runner.read;sha=runner.sha

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def status(stage,**extra):runner.save(D/'status.json',dict(stage=stage,unix=time.time(),**extra))
def start(cmd,path,env):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as log:process=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  q=Path('/proc')/str(process.pid)
  try:
   st=(q/'stat').read_text().rsplit(')',1)[1].split();actual=(q/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
   if actual==cmd and st[0] not in ('Z','X'):return process,dict(pid=process.pid,process_start_ticks=int(st[19]),command=cmd,cwd=str(R))
  except FileNotFoundError:pass
  if process.poll() is not None:raise RuntimeError('Child exited before exec: '+str(path))
  time.sleep(.05)
 raise RuntimeError('Child exec identity timed out; preserve process/log and do not duplicate')
def cleanup(arm,items):
 assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Need pidfd ownership-safe termination'
 sent=[];handles=[]
 try:
  for i in items:
   owner=i['owner']
   if not alive(owner):continue
   # An unaccounted live client means the service stays alive for investigation.
   out=subprocess.run(['ss','-Htn','state','established',f"( sport = :{i['port']} )"],capture_output=True,text=True,check=True,timeout=10)
   assert not out.stdout.strip(),'Service still has client connections; preserve model'
   fd=os.pidfd_open(owner['pid']);handles.append((owner,fd));assert alive(owner),'Owner changed after pidfd open'
  for owner,fd in handles:
   assert alive(owner),'Owner changed before signal';signal.pidfd_send_signal(fd,signal.SIGTERM);sent.append(owner['pid'])
 finally:
  for _,fd in handles:os.close(fd)
 deadline=time.monotonic()+30
 while time.monotonic()<deadline and any(alive(i['owner']) for i in items):time.sleep(.2)
 remaining=[i['owner']['pid'] for i in items if alive(i['owner'])];write(D/arm/'cleanup.json',dict(owned_term_sent=sent,remaining_owned=remaining,foreign_untouched=True,pidfd_used=True));assert not remaining,'Service cleanup incomplete'
def run_arm(arm,c,old):
 items=[];out=D/arm;out.mkdir(exist_ok=False)
 try:
  ep=OLD/'B-endpoint.json';parity_path=R/'results/lora-dev150-final-v1/base-parity.json' if arm=='base' else OLD/'parity/B.json';parity=read(parity_path)
  assert parity['passed'] and parity['ambient_rng_unchanged'] and len(parity['trace'])==6
  assert all(t['full_action_exact'] and t['all_rng_exact'] for t in parity['trace'])
  health=gpu_health();layout,counts=choose_layout(health,read(D/'launch-decision.json').get('max_model_instances',21))
  write(out/'layout.json',dict(health=health,counts=counts,layout=layout,clients_per_model=6,foreign_untouched=True));status('starting_models',arm=arm,models=len(layout))
  children={};model='/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365'
  from multiplex_inference.client import WireClient
  from local_eval.remote_client import validate_endpoint
  from continuous_eval2000_v3.client import validate_stage_endpoint
  # Keep CUDA context creation bounded while using every admitted GPU.
  for offset in range(0,len(layout),16):
   wave=layout[offset:offset+16]
   for i in wave:
    path=out/'services'/i['key'];path.mkdir(parents=True,exist_ok=False);i['directory']=str(path)
    with socket.socket() as s:s.bind(('127.0.0.1',i['port']))
    if arm=='base':cmd=[c['server_python'],'-u','-m','multiplex_inference.server','--xr1-repo',old['repo'],'--model',model,'--port',str(i['port']),'--max-clients','8']
    else:cmd=[c['server_python'],'-u','-m','continuous_eval2000_v3.server','--mode','serve','--binding',str(ep),'--binding-sha256',sha(ep),'--parity',str(parity_path),'--parity-sha256',sha(parity_path),'--port',str(i['port']),'--max-clients','8','--server-manifest',str(path/'servers.json')]
    process,owner=start(cmd,path/'server.log',common_env(c,i['gpu_uuid']));i['owner']=owner;items.append(i);children[i['key']]=process;write(path/'owner.json',owner)
   deadline=time.monotonic()+900;pending=list(wave)
   while pending:
    for i in list(pending):
     assert children[i['key']].poll() is None,'Service exited '+i['key']
     path=Path(i['directory'])/'servers.json'
     if arm=='B' and not path.exists():continue
     try:client=WireClient(i['port'],timeout=2)
     except (ConnectionRefusedError,TimeoutError):continue
     try:h=client.hello
     finally:client.close()
     assert h['model_assets_sha256']==parity['model']['model_assets_sha256'] and alive(i['owner'])
     if arm=='base':
      assert not h.get('lora_serving_identity') and not h.get('stage_serving_identity')
      write(path,dict(repo=str(R),python=c['server_python'],model=model,ready=True,source_sha256=parity['source_sha256'],parity_report=str(parity_path),parity_sha256=sha(parity_path),hello=h,servers=[dict(i['owner'],port=i['port'],gpu=str(i['gpu_index']))],max_clients=8));validate_endpoint(path,parity_path,model)
     else:validate_stage_endpoint(path,parity_path,model,binding_path=str(ep),binding_sha256=sha(ep))
     pending.remove(i)
    assert time.monotonic()<deadline,'Model startup timeout'
    if pending:time.sleep(1)
   status('starting_models',arm=arm,models=len(layout),ready=len(items))
  write(out/'all-owners.json',[i['owner'] for i in items]);status('warmup',arm=arm,models=len(items));write(out/'warmup.json',warmup(c,items,{arm:parity}))
  b={k:old[k] for k in ('root','repo','package_root','python','osmesa_library','packages','source_sha256')};models={}
  for i in items:
   p=Path(i['directory'])/'servers.json';h=read(p)['hello'];models[i['key']]=dict(arm=arm,gpu_index=i['gpu_index'],gpu_uuid=i['gpu_uuid'],checkpoint_path=model,checkpoint_sha256=runner.canonical(h['model_assets_sha256']),server_manifest=str(p),server_manifest_sha256=sha(p),parity_report=str(parity_path),parity_sha256=sha(parity_path),available_client_slots=6)
  b.update(schema='official_b2500_v1_bindings',approved_for_evaluation=True,runner_pins_sha256=sha(P/'runner-source-pins.json'),cohort_root=str(D/'cohort'),baseline_cohort=str(D/'cohort/base'),seed_audit=dict(path=str(D/'seed-audit.json'),sha256=sha(D/'seed-audit.json')),decision_protocol=dict(path=str(P/'protocol.json'),sha256=sha(P/'protocol.json')),models=models,arm_endpoints={arm:[i['key'] for i in items]},reference_B=dict(report_path=str(OLD/'analysis/report.json'),report_sha256=sha(OLD/'analysis/report.json'),bindings_path=str(OLD/'B-bindings.json'),bindings_sha256=sha(OLD/'B-bindings.json'),endpoint_path=str(ep),endpoint_sha256=sha(ep)))
  write(D/f'{arm}-bindings.json',b);cmd=[c['sim_python'],'-u','-m','official_b2500_v1.runner','preflight','--bindings',str(D/f'{arm}-bindings.json'),'--manifest',str(P/'manifest.json'),'--arms',arm,'--parallel',str(len(items)*6)]
  subprocess.run(cmd,cwd=R,env=runner.environment(b),check=True);status('evaluating',arm=arm,models=len(items),parallel=len(items)*6)
  cmd[4]='execute';cmd+=['--output',str(D/'cohort'/arm)];evaluation,o=start(cmd,out/'evaluation.log',runner.environment(b));write(out/'evaluation-owner.json',o);assert evaluation.wait()==0,'Incomplete cohort; preserve claims and unknowns'
 finally:
  active=[]
  for p in (D/'cohort'/arm/'processes').glob('*.json'):
   if runner.process_active(read(p)):active.append(read(p)['pid'])
  if items and not active:cleanup(arm,items)
def main():
 import fcntl
 D.mkdir(parents=True,exist_ok=True);lock=(D/'launch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (D/'owner.json').exists(),'Prior owner exists; no duplicate launch';write(D/'owner.json',runner.identity())
 try:
  decision=read(D/'launch-decision.json');assert decision.get('authorized') is True and decision.get('candidate')=='B' and decision.get('manifest_sha256')==sha(P/'manifest.json') and decision.get('protocol_sha256')==sha(P/'protocol.json') and decision.get('pins_sha256')==sha(P/'runner-source-pins.json'),'Missing exact launch decision'
  for name,digest in read(P/'runner-source-pins.json').items():assert sha(P/name)==digest,'Source inventory changed: '+name
  assert sha(P/'protocol.json')==runner.PROTOCOL_SHA
  status('seed_audit');runner.design(P/'manifest.json');protocol=read(P/'protocol.json');assert sha(OLD/'B-endpoint.json')==protocol['frozen_B_endpoint_sha256']
  ep=read(OLD/'B-endpoint.json');assert sha(ep['branch']['path'])==protocol['frozen_B_branch_sha256']
  old=read(OLD/'B-bindings.json')
  for p,h in old['source_sha256'].items():assert sha(p)==h,p
  for p,h in read(P/'runtime-dependencies.json').items():assert sha(p)==h,'Launch dependency changed: '+p
  c=dict(read(R/'stage_dev150_v2/launch-ready.json'),root=str(R))
  assert Path(old['root']).resolve()==R and old['python']==c['sim_python']
  assert Path(c['server_python'])==R/'envs/training/bin/python' and Path(c['sim_python'])==R/'envs/sim/bin/python'
  assert all(Path(c[k]).is_file() for k in ['server_python','sim_python'])
  for f in c['fixtures']:assert sha(f['path'])==f['sha256'] and sha(Path(f['path']).with_suffix('.npz'))==f['npz_sha256']
  subprocess.run([c['server_python'],'-m','official_b2500_v1.audit_seeds','--root',str(R),'--output',str(D/'seed-audit.json'),'--manifest',str(P/'manifest.json'),'--manifest-sha256',sha(P/'manifest.json'),'--protocol',str(P/'protocol.json'),'--protocol-sha256',sha(P/'protocol.json')],cwd=R,env=common_env(c),check=True)
  runner.audit_seed_binding(dict(path=str(D/'seed-audit.json'),sha256=sha(D/'seed-audit.json')))
  status('cpu_preflight');subprocess.run([c['sim_python'],'-m','official_b2500_v1.cpu_probe'],cwd=R,env=runner.environment(old),check=True)
  for arm in ['B']:run_arm(arm,c,old)
  status('analyzing');subprocess.run([c['server_python'],'-m','official_b2500_v1.analyze'],cwd=R,env=common_env(c),check=True)
  write(D/'completion.json',dict(completed=True,unix=time.time(),episodes=2500,formal2500_started=True,leaderboard_submission_started=False));status('complete')
 except BaseException:status('failed',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
