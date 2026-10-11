"""One-shot owned-child scheduler for the frozen sprint. No training or retries."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
sys.path[:0]=[str(HERE),str(ROOT)]
from episode import read,sha,write,require,identity,active,environment,check_sources,native_reset_proof,setup_path
from protocol import make_manifest,validate
SERVICE=ROOT/'results/originx-sprint-xab-services-20261011-v1'
TRACKER_SHA='157dcdde03e64455a6efd7d6595741259afb34350754e41d1d7eaa81b5a7c58a'

def tracker():
    path=ROOT/'start_full_admitted_v2.py';require(sha(path)==TRACKER_SHA,'Tracker changed')
    spec=importlib.util.spec_from_file_location('_sprint_tracker',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.TrackedChild

def prepare(args):
    ready=read(SERVICE/'ready.json');require(ready.get('passed') is True and ready.get('hold_seconds',0)>=360 and ready.get('same_socket_action_rng_parity_passed') is True,'Service parity not admitted')
    require(sha(SERVICE/'services-ready.json')==ready['services_ready_sha256'],'Service profile changed')
    models=read(SERVICE/'services-ready.json')['models']
    require({m['policy_id'] for m in models}=={'X','A','B'},'X/A/B profiles missing')
    for m in models:require(active(m['owner']) and sha(m['server_manifest'])==m['server_manifest_sha256'],'Model owner changed')
    output=args.output.resolve();require(output.parent==ROOT/'results' and output.name.startswith('originx-sprint-'),'New output namespace required')
    require(not output.exists(),'Output already exists, no overwrite')
    ref=read(ROOT/'results/native-reset-b2500-20261009-v1/config.json')
    require(sha(ref['manifest'])==ref['manifest_sha256'],'Historical task manifest changed')
    manifest=make_manifest(read(ref['manifest']),args.development,args.bv);validate(manifest)
    if args.bv:
        core=ROOT/'results/originx-sprint-xab-20261011-v1'
        require(manifest['cases']==read(core/'manifest.json')['cases'],'B/V cases differ from core')
        audit=dict(passed=True,shared_new_core_seed_audit=str(core/'seed-audit.json'),sha256=sha(core/'seed-audit.json'))
    else:
        from originx_confirmatory_20261009.protocol import seed_audit
        audit=seed_audit(ROOT,manifest)
    output.mkdir(parents=True,exist_ok=False);write(output/'seed-audit.json',audit,exclusive=True);write(output/'manifest.json',manifest,exclusive=True)
    sources=dict(ref['source_sha256'])
    for p in list(HERE.glob('*.py'))+list((ROOT/'originx_confirmatory_20261009').glob('*.py'))+[ROOT/'start_full_admitted_v2.py']:
        sources[str(p)]=sha(p)
    c={k:ref[k] for k in ('root','repo','package_root','python','osmesa_library','packages')}
    budget=dict(max_cli_batches=50,max_input_tokens=1000000,max_output_tokens=100000,quota_resets_allowed=0,paid_topup_allowed=False,max_batch=8)
    if args.bv:
        for p in (HERE/'bv_broker').glob('*'):
            if p.is_file():sources[str(p)]=sha(p)
        require((HERE/'bv_broker/broker.py').exists(),'B/V broker source absent')
        require(read(HERE/'bv_broker/budget.json') is not None,'B/V budget absent')
    c.update(schema='originx_confirmatory_config_v1',output=str(output),manifest=str(output/'manifest.json'),manifest_sha256=sha(output/'manifest.json'),
             models=[m for m in models if not args.bv or m['policy_id']=='B'],source_sha256=sources,development=args.development,bv=args.bv,
             response_wait_seconds=1200,intervention_fraction=.5,budget=budget,services_ready=str(SERVICE/'services-ready.json'),services_ready_sha256=ready['services_ready_sha256'],
             service_admission=str(SERVICE/'ready.json'),service_admission_sha256=sha(SERVICE/'ready.json'),
             protocol=dict(ref['protocol'],study='fixed 50 fresh environment cases; X/A/B frozen policies, independent B/V extension',
                           policy_wait_keepalive_seconds=120,seed_replacements=False,training=False,action_budget_matched=True,
                           cross_policy_prefix_equality_required=False,initial_state_equality_required=True,
                           A_actual_updates=1613,B_actual_updates=2000,compute_deadline_utc='2026-10-11T18:00:18Z'),
             original_cohort_preserved=True)
    check_sources(c);write(output/'config.json',c,exclusive=True)
    print(json.dumps(dict(config=str(output/'config.json'),manifest_sha256=c['manifest_sha256'],case_count=manifest['case_count'],jobs=len(manifest['jobs']))))

def preflight(args,c):
    import importlib.metadata
    setup_path();check_sources(c);validate(read(c['manifest']))
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONHASHSEED')=='0','CPU native environment required')
    require({n:importlib.metadata.version(n) for n in c['packages']}==c['packages'],'Packages changed')
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    require(all(get_task_horizon(j['task'])==j['horizon'] for j in read(c['manifest'])['jobs']),'Horizon changed')
    for m in c['models']:require(active(m['owner']) and sha(m['server_manifest'])==m['server_manifest_sha256'],'Model changed')
    write(Path(c['output'])/'preflight.json',dict(passed=True,config_sha256=sha(args.config),manifest_sha256=c['manifest_sha256'],native_reset=native_reset_proof(c)),exclusive=True)

def run(args,c):
    import fcntl
    check_sources(c);out=Path(c['output']);cs=sha(args.config);manifest=read(c['manifest']);jobs=manifest['jobs']
    require(sha(c['manifest'])==c['manifest_sha256'] and read(out/'preflight.json')['config_sha256']==cs,'Authority changed')
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (out/'launch.json').exists(),'Existing campaign cannot restart')
    require(60<=args.hard_seconds<=10800 and 0<=args.dispatch_seconds<=args.hard_seconds,'Invalid monotonic bounds')
    T=tracker();started=time.monotonic();hard=started+args.hard_seconds;dispatch=started+args.dispatch_seconds
    write(out/'launch.json',dict(identity(),config_sha256=cs,hard_seconds=args.hard_seconds,dispatch_seconds=args.dispatch_seconds,authoritative_utc=args.authoritative_utc),exclusive=True)
    pending=list(range(len(jobs)));live=[];done={};stop=[];first_error=None
    signal.signal(signal.SIGTERM,lambda *_:stop.append('requested_drain'));signal.signal(signal.SIGINT,lambda *_:stop.append('requested_drain'))
    def progress():
        write(out/'progress.json',dict(schema='originx_sprint_progress_v1',planned_unique_cases=manifest['case_count'],planned_arm_outcomes=len(jobs),
              finished_arm_outcomes=len(done),active=len(live),pending=len(pending),elapsed_seconds=time.monotonic()-started,stop_reasons=stop,
              by_policy={p:dict(done=sum(jobs[i]['policy_id']==p for i in done),active=sum(jobs[x['index']]['policy_id']==p for x in live),pending=sum(jobs[i]['policy_id']==p for i in pending)) for p in manifest['policy_arms']}))
    def reap(x, forced=None):
        code=x['child'].poll()
        if code is None:return False
        x['tracked'].close();x['log'].close();j=jobs[x['index']];rp=out/'episodes'/j['id']/'result.json'
        result=read(rp) if rp.exists() else dict(status='administrative_deadline' if forced else 'infrastructure_unknown',success=None,reason='owned worker ended without result')
        row=dict(job=j,returncode=code,result_status=result.get('status'),success=result.get('success'),forced_reason=forced,child=x['tracked'].receipt())
        write(out/'worker-exits'/(j['id']+'.json'),row,exclusive=True);done[x['index']]=row;return True
    def end_all(reason):
        # Shared grace, not N sequential grace intervals. Retain each pidfd.
        for x in live:
            if x['child'].poll() is None:x['tracked'].send(signal.SIGTERM)
        limit=time.monotonic()+10
        while any(x['child'].poll() is None for x in live) and time.monotonic()<limit:time.sleep(.1)
        for x in live:
            if x['child'].poll() is None:x['tracked'].send(signal.SIGKILL)
        limit=time.monotonic()+5
        while any(x['child'].poll() is None for x in live) and time.monotonic()<limit:time.sleep(.1)
        for x in live:
            require(x['child'].poll() is not None,'Owned child not reaped within bounded cleanup')
            reap(x,reason)
    try:
        if c.get('bv'):
            core=ROOT/'results/originx-sprint-xab-20261011-v1'
            while True:
                pr=read(core/'progress.json') if (core/'progress.json').exists() else {}
                b=pr.get('by_policy',{}).get('B',{})
                ready=out/'broker-ready.json'
                admitted=(b.get('done')==50 and b.get('active')==0 and b.get('pending')==0 and ready.exists())
                if admitted:
                    require(read(ready)['config_sha256']==cs,'Broker authority mismatch')
                    for m in c['models']:
                        con=subprocess.run(['ss','-Htn','state','established',f"( sport = :{m['port']} )"],capture_output=True,text=True,check=True,timeout=10)
                        require(not con.stdout.strip(),'B has active clients')
                    write(out/'reuse-admission.json',dict(core_progress_sha256=sha(core/'progress.json'),broker_ready_sha256=sha(ready),B_core_all50_terminal=True),exclusive=True);break
                if time.monotonic()>=dispatch or stop or (out/'drain.request.json').exists():stop.append('B_control_or_broker_not_ready_before_dispatch_deadline');break
                progress();time.sleep(5)
        last=0
        while live or (pending and not stop):
            now=time.monotonic()
            if now>=dispatch and not stop:stop.append('administrative_stop_dispatch')
            if (out/'drain.request.json').exists() and not stop:stop.append('broker_or_explicit_drain')
            if (SERVICE/'draining.json').exists() and not stop:stop.append('service_lease_draining')
            if now>=hard:
                stop.append('administrative_compute_deadline')
                end_all('administrative_compute_deadline')
                live=[];break
            used=Counter(x['model_index'] for x in live)
            for mi,model in enumerate(c['models']):
                while used[mi]<model['slots'] and pending and not stop:
                    if time.monotonic()>=dispatch:stop.append('administrative_stop_dispatch');break
                    index=next((i for i in pending if jobs[i]['policy_id']==model['policy_id']),None)
                    if index is None:break
                    require(active(model['owner']),'Model exited');j=jobs[index]
                    command=[c['python'],'-B','-u',str(HERE/'episode.py'),'--config',str(args.config),'--config-sha',cs,'--index',str(index),'--model-index',str(mi)]
                    write(out/'claims'/(j['id']+'.json'),dict(job=j,command=command,config_sha256=cs,model_index=mi),exclusive=True)
                    lp=out/'logs'/(j['id']+'.log');lp.parent.mkdir(exist_ok=True);log=lp.open('x')
                    child=subprocess.Popen(command,cwd=ROOT,env=environment(c),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    tracked=T(child,command,ROOT);x=dict(index=index,model_index=mi,child=child,tracked=tracked,log=log);live.append(x);pending.remove(index)
                    write(out/'spawn-pending'/(j['id']+'.json'),tracked.receipt(),exclusive=True)
                    owner=tracked.await_identity();write(out/'processes'/(j['id']+'.json'),owner,exclusive=True);used[mi]+=1
            remaining=[]
            for x in live:
                if not reap(x):remaining.append(x)
            live=remaining
            if now-last>=10:progress();last=now
            time.sleep(.5)
    except BaseException:
        first_error=traceback.format_exc();write(out/'first-error.json',dict(error=first_error),exclusive=True);stop.append('scheduler_error')
    finally:
        end_all('scheduler_cleanup')
        live=[];progress()
        write(out/'completion.json',dict(schema='originx_confirmatory_completion_v1',config_sha256=cs,manifest_sha256=c['manifest_sha256'],
              all_children_drained=True,finished_arm_outcomes=len(done),never_started=len(pending),stop_reasons=stop,first_error=first_error,elapsed_seconds=time.monotonic()-started),exclusive=True)
    return 0 if not first_error else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','run']);p.add_argument('--output',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--development',action='store_true');p.add_argument('--bv',action='store_true');p.add_argument('--hard-seconds',type=float,default=6000)
    p.add_argument('--dispatch-seconds',type=float,default=4200);p.add_argument('--authoritative-utc',default='')
    args=p.parse_args()
    if args.action=='prepare':return prepare(args)
    c=read(args.config)
    return preflight(args,c) if args.action=='preflight' else run(args,c)
if __name__=='__main__':sys.exit(main())
