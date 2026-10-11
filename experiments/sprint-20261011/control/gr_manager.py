"""Own the GR waiting/admission/campaign sequence. Never executes an Astra CLI.

Lives outside the immutable runtime module inventory. Only this new study's
exact processes are signalled; predecessor identities are read-only gates.
"""
from pathlib import Path
import datetime, hashlib, json, os, signal, subprocess, sys, time, traceback

ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NS='originx_sprint_gr_20261011_v2'
BASE=ROOT/NS
HERE=Path(__file__).resolve().parent
OUT=HERE/'receipts'
STUDY=ROOT/'results/originx-sprint-gr-20261011-v2'
GUARD=ROOT/'results/originx-sprint-gr-services-20261011-v2'
DIRECTOR=ROOT/'originx_sprint_20261011_v1/director'
CUTOFF=1791741618.0
LATEST_SERVICE=CUTOFF-32*60
LATEST_READY=CUTOFF-20*60
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v,new=True):
    with Path(p).open('x' if new else 'w') as f:json.dump(v,f,indent=2);f.write('\n')
def require(v,msg):
    if not v:raise RuntimeError(msg)
def ref(p):return dict(path=str(p),sha256=sha(p))

def validate_transfer(transfer,workers,finished,inactive):
    require(transfer.get('all_workers_inactive') is True and transfer.get('all_models_inactive') is True,'Transfer not terminal')
    require(workers.get('all_workers_inactive') is True and workers.get('owners'),'No worker cleanup identities')
    require(finished.get('all_direct_children_reaped') is True,'Director has unreaped children')
    require(transfer.get('models'),'No model transfer identities')
    owners=workers['owners']+[m['owner'] for m in transfer['models']]
    require(all(m.get('still_active') is False for m in transfer['models']),'Transfer model active')
    require(all(inactive(o) for o in owners),'Actual predecessor owner remains active')
    return owners

def build_plan():
    prior=ROOT/'results/originx-sprint-xab-services-20261011-v1'
    return dict(schema='originx_sprint_gr_service_plan_v1',
        development_admission=ref(ROOT/'results/originx-gr00t-confirmatory-development-20261009-v2/admission-review.json'),
        prior_campaigns=[dict(output=str(ROOT/'results'/n),completion=ref(ROOT/'results'/n/'completion.json'))
                        for n in ('originx-sprint-xab-20261011-v1','originx-sprint-bv-20261011-v1')],
        prior_guard=dict(finished=ref(prior/'finished.json'),cleanup=ref(prior/'cleanup.json')),
        source_freeze=ref(BASE/'source-freeze.json'))

def scan_descendants(root_owner, seen, births, identity):
    """Follow actual PPID/birth links, including children before argv is ready.

    Never classify a process by its name. A retained child birth remains in the
    audit if it is reparented. Callers do not reap/kill its parent before this
    census finishes. A changed birth cannot inherit ownership from a reused PID.
    """
    root_birth=(root_owner['pid'],root_owner['process_start_ticks'])
    owned=set(seen)|{root_birth}
    changed=True
    while changed:
        changed=False
        parent_pids={pid for pid,start in owned if pid in births and births[pid]['process_start_ticks']==start}
        for pid,b in births.items():
            key=(pid,b['process_start_ticks'])
            if b['parent_pid'] in parent_pids and key not in owned:
                owned.add(key);changed=True
    live=[]
    for pid,start in owned-{root_birth}:
        b=births.get(pid)
        if b is None or b['process_start_ticks']!=start or b['state'] in ('Z','X'):continue
        current=identity(pid)
        require(current['process_start_ticks']==start,'Birth changed during descendant identity read')
        seen[(pid,start)]=current;live.append(current)
    return live

def proc_births(birth):
    result={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:result[int(p.name)]=birth(int(p.name))
        except (FileNotFoundError,ProcessLookupError):pass
    return result

def require_model_release(workers_proven):
    require(workers_proven,'Model release blocked: own worker tree unproven')

def final_verdict(failure,errors,workers_proven,all_reaped,service_spawned,guard_finished):
    guard_proven=not service_spawned or (guard_finished is not None and guard_finished.get('all_children_reaped') is True and guard_finished.get('failed') is False)
    return failure is None and not errors and workers_proven and guard_proven and all_reaped,guard_proven

def drain_campaign_tree(tracked, r, birth, receipts, seen):
    """Drain descendants, recheck the registration window, then reap parent."""
    errors=[];start=time.monotonic();frozen=False;stable=0
    if tracked.child.poll() is None:tracked.send(signal.SIGTERM)
    while time.monotonic()-start<45:
        # After a normal 15s drain, freeze the parent so it cannot spawn a fresh
        # aggregate/runner during the final census. Descendants are then frozen
        # before a second census, closing their own pending Popen window too.
        if time.monotonic()-start>=15 and tracked.child.poll() is None and not frozen:
            tracked.send(signal.SIGSTOP);frozen=True
        live=scan_descendants(tracked.owner,seen,proc_births(birth),r.identity)
        if frozen:
            for owner in live:
                if r.active(owner):r.signal_owned(owner,signal.SIGSTOP)
            live=scan_descendants(tracked.owner,seen,proc_births(birth),r.identity)
        for owner in live:
            if r.active(owner):r.signal_owned(owner,signal.SIGKILL if frozen else signal.SIGTERM)
        stable=stable+1 if not live else 0
        if stable>=3:
            if tracked.child.poll() is not None:tracked.close();break
            if frozen:
                # Parent is stopped, no descendant remains live, and three
                # fresh censuses found no new child. Reap only this Popen root.
                tracked.send(signal.SIGKILL);tracked.child.wait(timeout=5);tracked.close();break
        time.sleep(.2)
    live=scan_descendants(tracked.owner,seen,proc_births(birth),r.identity)
    require(not live and tracked.child.poll() is not None,'Campaign tree drain remains unproven')
    receipts.update(owners=list(seen.values()),live=live,root=tracked.receipt(),stable_censuses=stable,
                    frozen_parent=frozen,elapsed_seconds=time.monotonic()-start)
    return True

def main():
    import fcntl
    from originx_sprint_gr_20261011_v2 import service_guard as g, runner as r
    from originx_sprint_gr_20261011_v2.processes import TrackedChild, birth
    require(HERE==ROOT/'originx_sprint_gr_manager_20261011_v1','Wrong exclusive manager path')
    HERE.mkdir(exist_ok=True)
    lock=(HERE/'manager.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False)
    write(OUT/'owner.json',dict(r.identity(),source_sha256=sha(__file__)))
    write(OUT/'deadlines.json',dict(compute_cutoff=CUTOFF,latest_service=LATEST_SERVICE,latest_ready=LATEST_READY,
        stop_dispatch=CUTOFF-900,manager_drain=CUTOFF-120,CLI_started=False))
    stopped=[];children=[];failure=None;phase='waiting';admitted=False;campaign_seen={}
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.append('signal'))
    def status(name,**more):
        nonlocal phase
        phase=name;write(OUT/'status.json',dict(phase=name,unix=time.time(),**more),False)
    def spawn(label,cmd,env):
        require(not stopped and time.time()<CUTOFF-120,'No new process near hard cutoff')
        log=(OUT/(label+'.log')).open('x')
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        t=TrackedChild(p,cmd,ROOT);children.append((label,t,log))
        write(OUT/(label+'.pending.json'),t.receipt())
        write(OUT/(label+'.owner.json'),t.await_identity());return t
    try:
        while True:
            require(not stopped,'Manager interrupted while waiting')
            if time.time()>=LATEST_SERVICE:
                status('not_started',reason='No proven exclusive transfer before service-admission cutoff');return 0
            try:
                transfer=read(DIRECTOR/'transfer-cleanup.json');workers=read(DIRECTOR/'worker-cleanup.json')
                owners=validate_transfer(transfer,workers,read(DIRECTOR/'completion.json'),g.owner_inactive)
                require(g.owner_inactive(read(DIRECTOR/'owner.json')),'Director owner not exited')
                plan=build_plan()
                require(transfer['service_cleanup_sha256']==plan['prior_guard']['cleanup']['sha256'],'Transfer cleanup hash changed')
                g.verify_prerequisites(plan);g.verify_assets();gpu=g.validate_gpu(g.gpu_snapshot())
                write(OUT/'transfer-admission.json',dict(transfer=ref(DIRECTOR/'transfer-cleanup.json'),
                    workers=ref(DIRECTOR/'worker-cleanup.json'),director_completion=ref(DIRECTOR/'completion.json'),
                    director_owner=ref(DIRECTOR/'owner.json'),checked_owners=len(owners),gpu=gpu,unix=time.time()))
                break
            except (FileNotFoundError,RuntimeError,KeyError) as e:
                status('waiting',reason=str(e));time.sleep(10)
        write(BASE/'service-plan.json',plan)
        # 30 seconds absorbs interpreter startup; the guard reserves 60 more
        # seconds for its own children, all within the absolute compute cap.
        lease=int(CUTOFF-time.time()-30)
        require(lease>=660 and time.time()<LATEST_SERVICE,'Admission window closed')
        service=spawn('service',['python3','-B',str(BASE/'service_guard.py'),'--run','--plan',str(BASE/'service-plan.json'),
            '--plan-sha256',sha(BASE/'service-plan.json'),'--lease-seconds',str(lease)],g.environment(False))
        status('service_admission')
        while not (GUARD/'ready.json').exists():
            require(not stopped and time.time()<LATEST_READY,'Fresh admission missed latest ready cutoff')
            require(service.child.poll() is None and not (GUARD/'first-error.json').exists(),'GR service admission failed')
            time.sleep(2)
        require(time.time()<LATEST_READY,'Admission completed after cutoff')
        ready=read(GUARD/'ready.json');parity=read(ready['parity_path'])
        require(parity.get('passed') is True and parity.get('clients')==6 and parity.get('hold_seconds',0)>=360,'Missing real six-stream 360s probe')
        admitted=True
        reference=ROOT/'results/native-reset-b2500-20261009-v1/config.json';cfg=read(reference)
        manifest=ROOT/'results/originx-sprint-xab-20261011-v1/manifest.json'
        require(sha(manifest)==read(BASE/'frozen-manifest.json')['source_manifest_sha256'],'Frozen shared cases changed')
        dev=ROOT/'results/originx-gr00t-confirmatory-development-20261009-v2/admission-review.json'
        cmd=[cfg['python'],'-B','-u','-m',NS+'.campaign','--reference-config',str(reference),'--reference-config-sha256',sha(reference),
            '--reference-manifest',str(manifest),'--reference-manifest-sha256',sha(manifest),
            '--service-manifest',ready['service_profile'],'--service-manifest-sha256',ready['service_profile_sha256'],
            '--source-freeze',str(BASE/'source-freeze.json'),'--source-freeze-sha256',sha(BASE/'source-freeze.json'),
            '--checkpoint',str(ROOT/'originx_gr00t_confirmation_20261009/checkpoint-120000'),
            '--parity-result',ready['parity_path'],'--parity-result-sha256',ready['parity_sha256'],
            '--development-review',str(dev),'--development-review-sha256',sha(dev),
            '--duration-seconds',str(int(CUTOFF-time.time()-150))]
        campaign=spawn('campaign',cmd,r.environment(cfg));status('campaign_waiting_for_root_broker')
        while campaign.child.poll() is None:
            scan_descendants(campaign.owner,campaign_seen,proc_births(birth),r.identity)
            require(not stopped and time.time()<CUTOFF-120,'Administrative compute drain')
            time.sleep(2)
        campaign.close();status('campaign_terminal',returncode=campaign.child.returncode)
        require(campaign.child.returncode==0,'Campaign ended nonzero; fixed denominator retains unknown outcomes')
    except BaseException:
        failure=traceback.format_exc();write(OUT/'error.json',dict(phase=phase,error=failure))
    finally:
        status('draining',admitted=admitted)
        # Ask the campaign/runner to drain before stopping the model owner.
        if (STUDY/'config.json').exists() and not (STUDY/'drain.request.json').exists():
            write(STUDY/'drain.request.json',dict(reason='manager_finalization',unix=time.time()))
        errors=[];trees=[];workers_proven=True
        for label,t,log in children:
            if label!='campaign':continue
            proof={}
            try:
                # Durable identities also cover a supervisor which exited
                # between two /proc samples, reparenting its known child.
                control=BASE/'campaigns'/STUDY.name
                for p in list((STUDY/'processes').glob('*.json'))+list(control.glob('*.owner.json')):
                    o=read(p);campaign_seen[(o['pid'],o['process_start_ticks'])]=o
                drain_campaign_tree(t,r,birth,proof,campaign_seen)
            except BaseException:
                workers_proven=False;errors.append(dict(label=label,error=traceback.format_exc()))
            trees.append(proof)
        # Independently census durable owner records, including preflight,
        # rollout and aggregate supervisors, in addition to episode workers.
        control=BASE/'campaigns'/STUDY.name
        owners=[]
        for p in list((STUDY/'processes').glob('*.json'))+list(control.glob('*.owner.json')):
            try:
                owner=read(p);owners.append(owner)
                require(g.owner_inactive(owner),'Recorded own process still active: '+str(p))
            except BaseException:
                workers_proven=False;errors.append(dict(path=str(p),error=traceback.format_exc()))
        write(OUT/'worker-tree-cleanup.json',dict(trees=trees,recorded_owners=owners,workers_proven=workers_proven,errors=errors))
        for label,t,log in reversed(children):
            try:
                if label=='service':require_model_release(workers_proven)
                if label=='service' and t.child.poll() is None and (GUARD/'ready.json').exists():
                    if not (GUARD/'release-request.json').exists():
                        write(GUARD/'release-request.json',dict(service_profile_sha256=read(GUARD/'ready.json')['service_profile_sha256']))
                    try:t.child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                if t.child.poll() is None:t.drain(grace=35 if label=='service' else 10,kill_timeout=5)
                else:t.close()
            except BaseException:errors.append(dict(label=label,error=traceback.format_exc()))
            finally:log.close();write(OUT/(label+'.exit.json'),t.receipt())
        guard_finished=read(GUARD/'finished.json') if (GUARD/'finished.json').exists() else None
        service_spawned=any(label=='service' for label,_,_ in children)
        all_reaped=all(t.reaped for _,t,_ in children)
        success,guard_proven=final_verdict(failure,errors,workers_proven,all_reaped,service_spawned,guard_finished)
        if not success and failure is None:failure='Final owned-process cleanup or guard proof incomplete'
        write(OUT/'completion.json',dict(phase=phase,failed=not success,admitted=admitted,
            all_direct_children_reaped=all_reaped,all_workers_inactive=workers_proven,
            guard_finished=guard_finished,guard_cleanup_proven=guard_proven,
            errors=errors,unix=time.time(),planned_arms=100,CLI_started=False))
        status('completed' if success else 'failed',admitted=admitted)
    return 1 if failure else 0

if __name__=='__main__':sys.exit(main())
