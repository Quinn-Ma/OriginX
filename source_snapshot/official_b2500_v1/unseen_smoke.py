"""CPU-only four-worker50-task asset/reset/readback smoke; no policy, no score."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,re,subprocess,sys,time,traceback
from official_b2500_v1 import runner
R=Path(__file__).resolve().parent.parent;P=R/'official_b2500_v1';D=R/'results/b2500-assets-smoke-v1'
SEED_START=2090700000
read=runner.read;sha=runner.sha

def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def seed_audit(manifest_path):
 from official_b2500_v1 import audit_seeds as a
 old=sys.argv[:];oldmarkers=a.MARKERS;oldmin=a.MIN_SEED;oldmax=a.MAX_SEED
 try:
  a.MIN_SEED=SEED_START;a.MAX_SEED=SEED_START+49;a.MARKERS=('b2500-assets-smoke-v1',)
  sys.argv=[str(P/'audit_seeds.py'),'--root',str(R),'--output',str(D/'seed-audit.json'),'--manifest',str(manifest_path),'--manifest-sha256',sha(manifest_path),'--protocol',str(P/'protocol.json'),'--protocol-sha256',sha(P/'protocol.json')];a.main()
 finally:sys.argv=old;a.MARKERS=oldmarkers;a.MIN_SEED=oldmin;a.MAX_SEED=oldmax
 audit=read(D/'seed-audit.json');assert audit['passed'] and audit['collisions']==[] and audit['environment_inference_seed_range']==[SEED_START,SEED_START+49]
 # The reused source emits a formal2500 cardinality label; explicitly override interpretation here.
 write(D/'seed-audit-scope.json',dict(episodes=50,diagnostic_only=True,reused_auditor_cardinality_label_not_applicable=True,outcome_use=False,audit_sha256=sha(D/'seed-audit.json')))
def worker(index):
 manifest=read(D/'manifest.json');job=manifest['jobs'][index];out=D/'trials'/f'{index:02}-{job["task"]}';out.mkdir(parents=True,exist_ok=False);write(out/'owner.json',runner.identity());env=guard=None;started=time.time();result={}
 try:
  b=read(R/'results/continued-ab2000-v1/B-bindings.json');expected=runner.environment(b)
  for k in ['MUJOCO_GL','PYOPENGL_PLATFORM','CUDA_VISIBLE_DEVICES','LP_NUM_THREADS','LD_LIBRARY_PATH','PYTHONHASHSEED']:assert os.environ.get(k)==expected[k],k
  for n,h in read(P/'runner-source-pins.json').items():assert sha(P/n)==h,n
  for p,h in b['source_sha256'].items():assert sha(p)==h,p
  sys.path[:0]=[str(P),str(R/'remote_processor_candidate_v1/deps'),str(R)]
  import random,numpy as np,torch,gymnasium as gym,robocasa
  from continuous_branch_v3.reset_fix import install as reset_install
  from continuous_branch_v3.scene_identity import extra_identity
  from context_lifetime import install
  from readback_guard import install_guard
  import dev_repeat_audit as audit
  assert not torch.cuda.is_initialized() and torch.version.cuda is None
  patch=reset_install();context=install();guard=install_guard(out/'readback-guard',library_path=b['osmesa_library'])
  seed=job['env_seed'];random.seed(seed);np.random.seed(seed)
  env=gym.make('robocasa/'+job['task'],split='pretrain',seed=seed);obs,_=env.reset(seed=seed);guard.check()
  entry=runner.load(Path(b['repo'])/'eval_robocasa365/entry.py','_smoke_official_entry')
  core=env.unwrapped.env;initial=audit.capture_initial(out,obs,entry.observation_to_state(obs),core.sim);initial.update(extra_identity(core));initial.update(xml_sha256=sha(out/'initial.xml'),layout_id=int(core.layout_id),style_id=int(core.style_id));write(out/'initial_scene.json',initial)
  checks={}
  for name in audit.reference_tools.CAMERAS:
   arr=np.asarray(obs[name]);assert arr.ndim==3 and arr.shape[-1]==3 and arr.size>0 and np.isfinite(arr).all(),name
   checks[name]=dict(shape=list(arr.shape),dtype=str(arr.dtype),sha256=hashlib.sha256(arr.tobytes()).hexdigest())
  env.close();env=None;guard.check();guard.close();guard=None
  proof=read(out/'readback-guard/summary.json');assert runner.guard_complete(proof)
  assert not torch.cuda.is_initialized()
  result=dict(passed=True,task=job['task'],seed=seed,cameras=checks,reset_patch=patch,context=context,policy_actions=0,success_score_emitted=False)
 except BaseException:result=dict(passed=False,task=job['task'],seed=job['env_seed'],traceback=traceback.format_exc(),policy_actions=0,success_score_emitted=False)
 finally:
  for name,obj in [('env',env),('guard',guard)]:
   if obj is not None:
    try:obj.close()
    except BaseException:result.update(passed=False,cleanup=name,cleanup_error=traceback.format_exc())
  result.update(wall_seconds=time.time()-started);write(out/'result.json',result)
 return 0 if result['passed'] else 1

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);a=parser.parse_args()
 if a.worker is not None:return worker(a.worker)
 assert R==Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
 if D.exists():print(json.dumps(dict(action='existing_smoke_preserved_no_rerun',directory=str(D),report=read(D/'report.json') if (D/'report.json').exists() else None)));return 2
 D.mkdir();write(D/'owner.json',runner.identity());tasks=sorted(runner.design(P/'manifest.json')['tasks'],key=lambda x:(x['stratum']!='composite_unseen',-x['horizon'],x['task']));assert len(tasks)==50
 manifest=dict(kind='official50_assets_reset_CPU_smoke_no_policy',jobs=[dict(task=t['task'],env_seed=SEED_START+i) for i,t in enumerate(tasks)],workers=4,policy_actions=0,not_formal_episodes=True,no_score=True);write(D/'manifest.json',manifest);seed_audit(D/'manifest.json')
 b=read(R/'results/continued-ab2000-v1/B-bindings.json');environment=runner.environment(b)
 def one(i):
  path=D/f'worker-{i:02}.log';cmd=[b['python'],'-u','-m','official_b2500_v1.unseen_smoke','--worker',str(i)]
  with path.open('x') as log:child=subprocess.Popen(cmd,cwd=R,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  # CPU construction may take minutes under concurrent evaluation; preserve any incomplete child on timeout.
  try:returncode=child.wait(timeout=600)
  except subprocess.TimeoutExpired:return dict(index=i,pid=child.pid,timed_out=True,returncode=None,child_preserved=True)
  return dict(index=i,pid=child.pid,timed_out=False,returncode=returncode)
 processes=[];next_index=0;stop=False
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  active=set()
  while active or (next_index<50 and not stop):
   while len(active)<4 and next_index<50 and not stop:active.add(pool.submit(one,next_index));next_index+=1
   done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:
    result=future.result();processes.append(result);active.remove(future)
    if result['timed_out']:stop=True

 results=[]
 for i,t in enumerate(tasks):
  p=D/'trials'/f'{i:02}-{t["task"]}'/'result.json';results.append(read(p) if p.exists() else dict(passed=False,task=t['task'],missing_terminal_evidence=True))
 report=dict(passed=all(x.get('passed') for x in results) and all(x['returncode']==0 for x in processes),tasks=50,workers=4,policy_actions=0,score_emitted=False,formal_seeds_used=False,results=results,processes=processes)
 write(D/'report.json',report);print(json.dumps(report));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
