"""One-use no-LLM launch supervisor for never-claimed v2 work within the user deadline."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time,traceback
from broker import ProcessLock,local_owner,write_json
from sync_multigpu_v1 import ssh
HERE=Path(__file__).resolve().parent
STATE=HERE/'launch_multigpu_v2'
ROOT='/ephemeral/qinzhen/robocasa-xr1-20261003'
NAME='originx-confirmatory-multigpu-20261009-v2'
GROUP='originx-confirmatory-multigpu-services-20261009-v2'
AMEND='results/originx-three-hour-amendment-20261009-v1'
EVAL_END=datetime.fromisoformat('2026-10-09T22:40:12+00:00')
READY_END=datetime.fromisoformat('2026-10-09T22:05:00+00:00')
def utc():return datetime.now(timezone.utc)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(phase,**kw):write_json(STATE/'status.json',dict(phase=phase,at=utc().isoformat(),**kw))
def once(name,value):
 p=STATE/name
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def remote_spawn(label,command,published_owner):
 # Explicit one-use attempt + Popen ownership + bounded identity, then safe handoff.
 code='''from pathlib import Path
import json,sys,subprocess,hashlib
r=Path(%r);sys.path.insert(0,str(r));a=r/%r
assert hashlib.sha256((r/'start_full_admitted_v2.py').read_bytes()).hexdigest()=='157dcdde03e64455a6efd7d6595741259afb34350754e41d1d7eaa81b5a7c58a'
from start_full_admitted_v2 import TrackedChild
label=%r;cmd=%r;owner_path=Path(%r)
assert not owner_path.exists(),'Existing owner: no duplicate launch'
with (a/(label+'-attempt.json')).open('x') as f:json.dump(dict(command=cmd),f,indent=2)
with (a/(label+'.log')).open('x') as log:
 p=subprocess.Popen(cmd,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 t=TrackedChild(p,cmd,r)
 try:
  owner=t.await_identity(10)
  import time
  deadline=time.monotonic()+15;published=None
  while p.poll() is None and time.monotonic()<deadline:
   try:
    candidate=json.loads(owner_path.read_text())
   except (FileNotFoundError,json.JSONDecodeError):
    time.sleep(.1);continue
   assert all(candidate.get(k)==owner[k] for k in ('pid','process_start_ticks','command','cwd')),'Published child owner differs'
   published=candidate;break
  assert published is not None,'Complete child owner not published within bounded handshake'
  t.handoff(published)
 except BaseException:
  t.drain();raise
 finally:
  with (a/(label+'-spawn.json')).open('x') as f:json.dump(t.receipt(),f,indent=2)
 print(json.dumps(dict(owner=published,handed_off=t.handed_off)))
'''%(ROOT,AMEND,label,command,published_owner)
 result=ssh(code,60);once(label+'-spawn.json',result);return result

def local_spawn(script,args=()):
 script_path=HERE/script;label=script.removesuffix('.py')
 once(label+'-attempt.json',dict(script_sha256=sha(script_path),arguments=list(args)))
 with (HERE/(label+'.stdout.log')).open('x',encoding='utf-8') as out,(HERE/(label+'.stderr.log')).open('x',encoding='utf-8') as err:
  p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(script_path),*args],cwd=HERE,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
  once(label+'-spawn.json',dict(pid=p.pid,script_sha256=sha(script_path),actual_owner_must_be_read=True));return p

def failure_release():
 # Read exact owners plus exact campaign/config argv to catch an unrecorded startup child.
 # A release request asks the existing guardian to retire its own children only.
 code=r'''from pathlib import Path
import json,os,hashlib,time
r=Path('/ephemeral/qinzhen/robocasa-xr1-20261003');out=r/'results/originx-confirmatory-multigpu-20261009-v2'
config=str(out/'config.json');script=str(r/'continue_multigpu_v2.py');runner=str(r/'originx_confirmatory_20261009/runner.py')
def read(p):return json.loads(p.read_text())
def inactive(o):
 assert type(o.get('pid')) is int and type(o.get('process_start_ticks')) is int and isinstance(o.get('command'),list) and o.get('cwd'),'Malformed owner'
 try:
  fields=(Path('/proc')/str(o['pid'])/'stat').read_text().rsplit(')',1)[1].split()
 except (FileNotFoundError,ProcessLookupError):return True
 return fields[0] in ('Z','X') or int(fields[19])!=o['process_start_ticks']
blockers=[];checked=[]
paths=set(out.glob('*.owner.json'))|set((out/'processes').glob('*.json'))|set((out/'episodes').glob('*/owner.json'))
if (out/'launch.json').exists():paths.add(out/'launch.json')
for p in paths:
 try:
  ended=inactive(read(p));checked.append(dict(path=str(p),inactive=ended))
  if not ended:blockers.append(dict(path=str(p),reason='recorded_owner_active'))
 except Exception as e:blockers.append(dict(path=str(p),reason='owner_identity_unknown',error=repr(e)))
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  # UID only narrows a read-only scan; exact argv, not UID, determines scope.
  if proc.stat().st_uid!=os.getuid():continue
  command=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
  scoped=(script in command and 'campaign' in command) or (runner in command and '--config' in command and config in command)
  if scoped:
   fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
   if fields[0] not in ('Z','X'):blockers.append(dict(pid=int(proc.name),reason='scoped_campaign_or_worker_active',command=command))
 except (FileNotFoundError,ProcessLookupError):pass
 except Exception as e:blockers.append(dict(pid=int(proc.name),reason='owned_candidate_scan_unknown',error=repr(e)))
groups=[]
for gpu in (3,6):
 g=r/'results/originx-confirmatory-multigpu-services-20261009-v2'/('gpu'+str(gpu));row=dict(gpu=gpu,path=str(g),action='preserve')
 if not g.exists():row['reason']='guardian_not_started';groups.append(row);continue
 if blockers:row['reason']='campaign_or_worker_active_or_unknown';groups.append(row);continue
 try:
  if not (g/'ready.json').exists() or not (g/'services-ready.json').exists():
   row['reason']='guardian_loading_or_not_ready_preserve_existing_owned_lease';row['manual_handoff']='Inspect guardian owner and bounded lease; no kill or release before readiness';groups.append(row);continue
  gate=read(g/'ready.json');ready=g/'services-ready.json';digest=hashlib.sha256(ready.read_bytes()).hexdigest()
  assert gate.get('passed') is True and gate['services_ready_sha256']==digest,'Ready proof differs'
  release=g/'release-request.json'
  if release.exists():
   prior=read(release);assert prior.get('services_ready_sha256')==digest,'Existing release belongs to other services'
   row.update(action='existing_release_preserved',sha256=hashlib.sha256(release.read_bytes()).hexdigest())
  else:
   value=dict(schema='originx_multigpu_service_release_v1',services_ready_sha256=digest,all_assigned_workers_stopped=True,reason='launch_manager_failure_after_actual_owner_and_exact_argv_scan',unix=time.time())
   with release.open('x') as f:json.dump(value,f,indent=2)
   row.update(action='requested_owned_retirement',sha256=hashlib.sha256(release.read_bytes()).hexdigest())
 except Exception as e:row.update(action='preserve',reason='guardian_state_unknown',error=repr(e))
 groups.append(row)
print(json.dumps(dict(campaign_workers_verified_inactive=not blockers,owner_checks=checked,blockers=blockers,groups=groups,no_processes_signalled=True)))
'''
 result=ssh(code,90);once('failure-release.json',result);return result

def watch_deadline(config_sha,pins):
 while utc()<EVAL_END:
  try:
   phase=ssh("from pathlib import Path\nimport json\np=Path("+repr(ROOT+'/results/'+NAME)+")\nprint(json.dumps({'finished':(p/'campaign-finished.json').exists(),'status':json.loads((p/'campaign-status.json').read_text()) if (p/'campaign-status.json').exists() else None}))",30)
   if phase['finished']:
    once('watchdog-terminal.json',dict(parent_utc=utc().isoformat(),campaign=phase,no_signals=True));return
  except Exception:
   status('watchdog_poll_unavailable',error=traceback.format_exc(),deadline_utc=EVAL_END.isoformat(),retry_read_only=True)
  time.sleep(min(30,max(0,(EVAL_END-utc()).total_seconds())))
 helper='deadline_multigpu_v2.py';digest=pins[helper];parent_utc=utc().isoformat()
 code="from pathlib import Path\nimport hashlib,sys,json\nr=Path("+repr(ROOT)+")\np=r/"+repr(helper)+"\nassert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(digest)+"\nsys.path.insert(0,str(r))\nfrom deadline_multigpu_v2 import enforce\nprint(json.dumps(enforce(r/'results'/"+repr(NAME)+","+repr(config_sha)+","+repr(parent_utc)+")))"
 result=ssh(code,90);once('deadline-receipt.json',result)

def main():
 STATE.mkdir(exist_ok=True)
 with ProcessLock(STATE/'launch.lock'):
  once('owner.json',local_owner())
  pins=read(HERE/'v2-source-pins.json');pins_sha=sha(HERE/'v2-source-pins.json')
  while utc()<READY_END:
   phases=[read(HERE/n/'status.json').get('phase') for n in ['finalization_v3','finalization_multigpu_v1']]
   if 'failed' in phases:raise RuntimeError('Prior finalization failed; no new work')
   if phases==['completed','completed']:break
   status('waiting_prior_terminal_archives',prior_phases=phases);time.sleep(30)
  else:raise RuntimeError('No time for an admitted continuation before fixed cutoff')
  gate=ssh("from pathlib import Path\nimport json,hashlib\nr=Path("+repr(ROOT)+")\npins="+repr(pins)+"\nassert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in pins.items())\nfor n in ['originx-confirmatory-20261009-v1','originx-confirmatory-multigpu-20261009-v1']:\n p=r/'results'/n\n assert json.loads((p/'cleanup.json').read_text())['all_owned_services_stopped'] is True\n assert (p/'broker_inventory_final/terminal-inventory.json').exists()\nprint(json.dumps({'passed':True}))")
  once('prior-archive-gate.json',gate)
  lease=max(3600,int((datetime.fromisoformat('2026-10-09T23:40:00+00:00')-utc()).total_seconds()))
  python=ROOT+'/envs/training/bin/python'
  status('starting_new_owned_guardians',lease_seconds=lease)
  for gpu in (3,6):
   check=ssh("import subprocess,json\np=subprocess.run("+repr([python,'-B',ROOT+'/multigpu_services_v2.py','--check','--gpu',str(gpu)])+",capture_output=True,text=True,timeout=60)\nassert p.returncode==0,p.stderr+p.stdout\nprint(p.stdout)",90)
   once('gpu'+str(gpu)+'-check.json',check)
   remote_spawn('v2-gpu'+str(gpu),[python,'-u',ROOT+'/multigpu_services_v2.py','--run','--gpu',str(gpu),'--lease-seconds',str(lease)],ROOT+'/results/'+GROUP+'/gpu'+str(gpu)+'/owner.json')
  while utc()<READY_END:
   groups=ssh("from pathlib import Path\nimport json\nr=Path("+repr(ROOT+'/results/'+GROUP)+")\nrows=[]\nfor gpu in (3,6):\n p=r/('gpu'+str(gpu))\n rows.append({'gpu':gpu,'ready':json.loads((p/'ready.json').read_text()) if (p/'ready.json').exists() else None,'failure':json.loads((p/'first-error.json').read_text()) if (p/'first-error.json').exists() else None})\nprint(json.dumps(rows))")
   if any(x['failure'] for x in groups):raise RuntimeError('Guardian admission failed: '+json.dumps(groups))
   if all(x['ready'] and x['ready']['passed'] for x in groups):break
   status('waiting_both_group_parity',groups=groups);time.sleep(30)
  else:raise RuntimeError('Parity not ready within time allowance')
  duration=min(5400,int((EVAL_END-utc()).total_seconds())-180)
  assert duration>=600,'Insufficient evaluation time; preserve unknowns'
  status('preparing_never_claimed_v2',duration_seconds=duration)
  cmd=[python,'-B',ROOT+'/prepare_reduced_launch_v2.py','--pins-sha256',pins_sha,'--duration-seconds',str(duration)]
  preparation=ssh("import subprocess,json\np=subprocess.run("+repr(cmd)+",cwd="+repr(ROOT)+",capture_output=True,text=True,timeout=300)\nassert p.returncode==0,p.stderr+p.stdout\nprint(p.stdout)",330)
  once('preparation.json',preparation)
  assert utc()<EVAL_END and (EVAL_END-utc()).total_seconds()>=duration+120,'Preparation consumed the cleanup reserve; preserve prepared but unexecuted plan'
  local_spawn('sync_multigpu_v2.py');local_spawn('finalize_multigpu_v2.py');local_spawn('finalize_reduced_v2.py')
  config=ROOT+'/results/'+NAME+'/config.json';cs=preparation['result']['config']['sha256']
  local_spawn('broker_multigpu_v2.py',['--remote-output',ROOT+'/results/'+NAME,'--local-output',str(HERE/'broker_multigpu_v2'),'--run-seconds',str(duration+180),'--startup-wait-seconds','600'])
  campaign=remote_spawn('v2-campaign',[python,'-u',ROOT+'/continue_multigpu_v2.py','campaign','--config',config,'--config-sha256',cs],ROOT+'/results/'+NAME+'/campaign.owner.json')
  status('launched',campaign=campaign,config_sha256=cs,selected_arm_outcomes=preparation['result']['selected_arm_outcomes'],duration_seconds=duration,hard_eval_end_utc=EVAL_END.isoformat(),delivery_deadline_utc='2026-10-09T23:46:19Z',prior_debit=preparation['debit']['prior_usage'])
  watch_deadline(cs,pins)
if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc();STATE.mkdir(exist_ok=True)
  try:release=failure_release()
  except BaseException:release=dict(error=traceback.format_exc(),all_processes_preserved=True,manual_handoff='Inspect exact guardian owner/lease; no automatic kill')
  status('failed',error=error,automatic_retry=False,failure_retirement=release,inspect_exact_owned_guardians_before_any_cleanup=True);raise
