"""One-shot CPU evidence aggregation, only after all study owners terminate."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
BASE=ROOT/'originx_sprint_gr_20261011_v2'
STUDY=ROOT/'results/originx-sprint-gr-20261011-v2'
MANAGER=ROOT/'originx_sprint_gr_manager_20261011_v1/receipts'
GUARD=ROOT/'results/originx-sprint-gr-services-20261011-v2'
HERE=Path(__file__).resolve().parent
DEST=STUDY/'analysis-final-usage-v1'
BROKER=BASE/'broker_terminal_state'
DEADLINE=1791743418.0 # 2026-10-11 18:30:18 UTC, CPU evidence only
FINAL_END=1791744018.0 # 18:40:18 leaves five minutes before delivery
FREEZE='a148b90e4fa752fb455e35078a22a11db78b0c85e57f65acdd3389c2b160e5f4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v,new=True):
    with Path(p).open('x' if new else 'w') as f:json.dump(v,f,indent=2);f.write('\n')
def require(v,s):
    if not v:raise RuntimeError(s)
def owners(value):
    result=[]
    if isinstance(value,dict):
        if {'pid','process_start_ticks','command','cwd'}<=set(value):result.append(value)
        for v in value.values():result.extend(owners(v))
    elif isinstance(value,list):
        for v in value:result.extend(owners(v))
    return result
def verify_seal(seal,attestation):
    require(attestation.get('broker_inactive') is True and attestation.get('all_file_hashes_verified') is True,'Broker host has not attested terminal identity')
    require(attestation['remote_seal']==seal and seal['remote_path']==str(BROKER),'Sealer/attestation identity differs')
    expected=seal['files'];actual={str(p.relative_to(BROKER)):sha(p) for p in BROKER.rglob('*') if p.is_file()}
    require(actual==expected and not any(p.is_symlink() for p in BROKER.rglob('*')),'Sealed broker inventory changed')
    require(read(BROKER/'owner.json')==attestation['broker_owner'],'Broker owner differs from Windows inactive proof')
    require(read(BROKER/'broker_status.json').get('state') in ('finished','stopped'),'Broker status nonterminal')
    return len(actual)
def main():
    import fcntl
    require(HERE==ROOT/'originx_sprint_gr_final_usage_20261011_v1','Wrong one-shot namespace')
    lock=(HERE/'finalizer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (HERE/'owner.json').exists(),'Finalizer owner already exists; no reissue')
    require(sha(BASE/'source-freeze.json')==FREEZE,'Frozen runtime authority changed')
    for n,h in read(BASE/'source-freeze.json')['source_sha256'].items():require(sha(BASE/n)==h,'Frozen source changed: '+n)
    sys.path.insert(0,str(ROOT))
    from originx_sprint_gr_20261011_v2 import runner as r,service_guard as g
    from originx_sprint_gr_20261011_v2.processes import TrackedChild
    write(HERE/'owner.json',dict(r.identity(),source_sha256=sha(__file__)))
    child=None;error=None
    try:
        while True:
            require(time.time()<DEADLINE,'Evidence-only finalization wait deadline')
            if not (MANAGER/'completion.json').exists():
                write(HERE/'status.json',dict(phase='waiting_terminal_manager_and_broker_seal',unix=time.time()),False);time.sleep(30);continue
            completion=read(MANAGER/'completion.json');guard=read(GUARD/'finished.json')
            require(completion.get('all_direct_children_reaped') is True and completion.get('all_workers_inactive') is True
                    and completion.get('guard_cleanup_proven') is True and not completion.get('errors'),'Manager cleanup anomaly; do not aggregate')
            require(guard.get('all_children_reaped') is True and guard.get('failed') is False,'Model guard cleanup anomaly')
            require(read(STUDY/'completion.json').get('all_children_drained') is True,'Study workers not drained')
            if not (HERE/'broker-host-attestation.json').exists():
                write(HERE/'status.json',dict(phase='waiting_broker_host_seal',unix=time.time()),False);time.sleep(30);continue
            paths=[MANAGER/'owner.json',MANAGER/'worker-tree-cleanup.json',GUARD/'owner.json',GUARD/'drain.json',STUDY/'campaign.owner.json']
            paths+=list((STUDY/'processes').glob('*.json'))+list((BASE/'campaigns'/STUDY.name).glob('*.owner.json'))
            identities=[o for p in paths for o in owners(read(p))]
            if not all(g.owner_inactive(o) for o in identities):
                write(HERE/'status.json',dict(phase='waiting_actual_owner_exit',unix=time.time()),False);time.sleep(10);continue
            seal=read(BASE/'broker_terminal_state.seal.json');attestation=read(HERE/'broker-host-attestation.json')
            sealed_files=verify_seal(seal,attestation)
            config=read(STUDY/'config.json');manifest=read(STUDY/'manifest.json')
            require(manifest['case_count']==50 and len(manifest['jobs'])==100,'Fixed 50 case / 100 arm denominator changed')
            require(not DEST.exists() and not (HERE/'aggregation-started.json').exists(),'One-shot output already exists; never rerun')
            write(HERE/'admission.json',dict(manager_completion_sha256=sha(MANAGER/'completion.json'),guard_finished_sha256=sha(GUARD/'finished.json'),
                broker_host_attestation_sha256=sha(HERE/'broker-host-attestation.json'),broker_seal_sha256=sha(BASE/'broker_terminal_state.seal.json'),
                checked_identities=len(identities),sealed_files=sealed_files,config_sha256=sha(STUDY/'config.json'),manifest_sha256=sha(STUDY/'manifest.json'),
                manager_reported_failed=completion.get('failed'),administrative_unknowns_preserved=True,unix=time.time()))
            cmd=[config['python'],'-B','-u','-m','originx_sprint_gr_20261011_v2.aggregate','--config',str(STUDY/'config.json'),
                 '--broker-state',str(BROKER),'--output-directory',str(DEST)]
            write(HERE/'aggregation-started.json',dict(command=cmd,source_freeze_sha256=FREEZE,unix=time.time()))
            with (HERE/'aggregate.stdout').open('x') as stdout,(HERE/'aggregate.stderr').open('x') as stderr:
                p=subprocess.Popen(cmd,cwd=ROOT,env=r.environment(config),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                child=TrackedChild(p,cmd,ROOT);write(HERE/'aggregate.owner.json',child.await_identity())
                write(HERE/'status.json',dict(phase='aggregating_once',unix=time.time()),False)
                code=p.wait(timeout=min(900,max(1,FINAL_END-time.time())));child.close();require(code==0,'Frozen aggregator exited nonzero; preserve output, no retry')
            records=read(DEST/'normalized_records.json');report=read(DEST/'report.json');usage=read(DEST/'cli_usage.json')
            require(len(records)==100 and report['planned_arm_outcomes']==100,'Final fixed denominator mismatch')
            outputs={p.name:sha(p) for p in DEST.iterdir() if p.is_file()}
            write(HERE/'completion.json',dict(passed=True,all_children_reaped=child.reaped,fixed_arms=100,output=str(DEST),outputs_sha256=outputs,
                authority_valid=report['authority_valid'],inventory_complete=usage['inventory_complete'],
                complete_input_output_token_accounting=usage['complete_input_output_token_accounting'],unix=time.time()))
            write(HERE/'status.json',dict(phase='completed',output=str(DEST),unix=time.time()),False);return 0
    except BaseException:
        error=traceback.format_exc();write(HERE/'error.json',dict(error=error,unix=time.time()))
        write(HERE/'status.json',dict(phase='blocked_or_failed',error=error,unix=time.time()),False)
        return 1
    finally:
        if child:
            if child.child.poll() is None:child.drain(grace=5,kill_timeout=5)
            else:child.close()
            write(HERE/'aggregate.exit.json',child.receipt())
if __name__=='__main__':sys.exit(main())
