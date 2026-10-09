"""Continuation-only Astra broker with lifecycle admission and terminal draining.

The frozen broker and v1 retry wrapper are imported, never edited. Lifecycle
metadata is used only for technical admission and is never added to model input.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import broker
from run_broker_resilient import ResilientQueue, resolve_network_stop, FROZEN_SHA
from start_full_broker import WindowsProcesses

CODEX = Path(r'C:\Users\LuckyQinzhen\AppData\Local\OpenAI\Codex\bin\2e5e00daee91c61d\codex.exe')
HERE = Path(__file__).resolve().parent
CONTINUATION = broker.PROJECT + "/results/astra-native-reset-continuation-20261009-v2"
PRIOR = broker.PROJECT + "/results/astra-native-reset-full-20261009-v1"

REMOTE_GUARD = broker.REMOTE_COMMON + r'''
import time
mode, expected_config_sha, payload_json = sys.argv[2:]
payload = json.loads(payload_json)
def require(value, message):
    if not value: raise ValueError(message)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    require(not path.is_symlink() and path.is_file(), 'Missing/symbolic-link metadata file: '+str(path))
    return json.loads(path.read_text(encoding='utf-8'))
def project_file(name):
    path = pathlib.Path(name)
    require(path.is_absolute() and not path.is_symlink() and path.resolve(strict=True).is_relative_to(project), 'Artifact escaped approved project')
    return path
def publish(path, value):
    require(path.parent.resolve().is_relative_to(resolved), 'Publication escaped output')
    require(not path.is_symlink(), 'Symbolic-link publication target')
    if path.exists(): return {'result':'already_present','sha256':sha(path)}
    temporary = path.with_name('.'+path.name+'.'+os.urandom(16).hex()+'.tmp')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        try: os.link(temporary,path); result='created'
        except FileExistsError: result='already_present'
    finally: temporary.unlink()
    return {'result':result,'sha256':sha(path)}
if not (root/'config.json').exists() or not (root/'preflight.json').exists():
    require(mode=='bind','Prepared configuration disappeared')
    print(json.dumps({'state':'waiting_preflight'})); sys.exit(0)
config=read(root/'config.json'); config_sha=sha(root/'config.json')
require(config.get('schema')=='astra_native_reset_control_rescue_config_v1' and config.get('output')==str(root),'Config identity mismatch')
require(not expected_config_sha or expected_config_sha==config_sha,'Configuration changed after binding')
preflight=read(root/'preflight.json')
require(preflight.get('passed') is True,'Preflight did not pass')
continuation=config.get('continuation',{})
require(str(root)==str(project/'results/astra-native-reset-continuation-20261009-v2')
        and continuation.get('schema')=='astra_native_reset_continuation_v2'
        and continuation.get('prior_output')==str(project/'results/astra-native-reset-full-20261009-v1')
        and continuation.get('selected_cases')==968 and continuation.get('excluded_cases')==36
        and continuation.get('selection_uses_success') is False and continuation.get('quota_retry_included') is False,
        'Only the fixed 968-case unattempted continuation may start new CLI calls')
require(config.get('response_wait_seconds')==1200 and config.get('protocol',{}).get('policy_wait_keepalive_seconds')==120,'Continuation wait/keepalive contract differs')
ledger_path=project_file(continuation['exclusion_ledger'])
require(ledger_path==root/'continuation-exclusion-ledger.json' and sha(ledger_path)==continuation['exclusion_ledger_sha256'],'Exclusion ledger identity/hash mismatch')
ledger=read(ledger_path)
require(ledger.get('schema')=='astra_continuation_exclusion_ledger_v2' and ledger.get('fixed_denominator')==1004
        and ledger.get('selection_uses_success') is False and ledger.get('quota_retry_included') is False,'Exclusion ledger protocol differs')
selected=set(ledger['selected_case_ids']); excluded=set(ledger['excluded_case_ids'])
require(len(selected)==968 and len(excluded)==36 and not selected.intersection(excluded),'Prior cases overlap continuation')
manifest_path=project_file(config['manifest'])
require(manifest_path==root/'manifest.json' and sha(manifest_path)==config['manifest_sha256'],'Manifest identity/hash mismatch')
manifest=read(manifest_path); jobs={job['id']:job for job in manifest['jobs']}
require(len(manifest['jobs'])==len(jobs)==1936 and manifest.get('arms')==['control','astra']
        and {job['id'] for job in manifest['selected_original_jobs']}==selected,'Continuation assignments differ')
require(all(job.get('original_id') in selected and job.get('rescue_arm') in ('control','astra')
            and job['id']==job['original_id']+'--'+job['rescue_arm'] for job in jobs.values()),'Unexpected continuation job')
if mode in ('bind','ready'):
    sources=config.get('source_sha256',{}); require(bool(sources),'Empty source manifest')
    for name, digest in sources.items(): require(sha(project_file(name))==digest,'Source hash mismatch: '+name)
    require(sha(project_file(continuation['prior_output']+'/config.json'))==continuation['prior_config_sha256'],'Prior config changed')
    for name,digest in ledger.get('artifacts_sha256',{}).items(): require(sha(project_file(name))==digest,'Prior claim-ledger artifact changed')
    require(sha(project_file(config['failure_manifest']))==config['failure_manifest_sha256'],'Frozen failure manifest changed')
def alive(name):
    owner_path=root/'episodes'/name/'owner.json'
    if not owner_path.exists(): return False,'owner_metadata_missing'
    owner=read(owner_path); pid=owner.get('pid'); start=owner.get('process_start_ticks')
    if type(pid) is not int or type(start) is not int: return False,'owner_identity_invalid'
    try:
        fields=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return fields[0] not in ('Z','X') and int(fields[19])==start,'pid_and_start_ticks_checked'
    except FileNotFoundError: return False,'owner_process_not_present'
def terminal():
    completed=root/'completion.json'; finished=root/'campaign-finished.json'
    candidate=False; reason=None
    if completed.exists() and read(completed).get('all_children_drained') is True:
        candidate=True; reason='all_children_drained'
    elif finished.exists():
        value=read(finished)
        candidate=value.get('schema')=='astra_continuation_campaign_finished_v2' and value.get('config_sha256')==config_sha
        reason='campaign_finished'
    if candidate and not any(alive(name)[0] for name in jobs): return True,reason
    return False,None
if mode=='bind':
    ended,why=terminal()
    print(json.dumps({'state':'terminal' if ended else 'ready','terminal_reason':why,'config_sha256':config_sha,
          'preflight_sha256':sha(root/'preflight.json'),'manifest_sha256':config['manifest_sha256'],
          'selected_cases':968,'episodes':1936,'source_files_verified':len(config['source_sha256']),
          'existing_ready':read(root/'broker-ready.json') if (root/'broker-ready.json').exists() else None,
          'request_count':sum(1 for p in queue.glob('*/request.json')) if queue.exists() else 0})); sys.exit(0)
if mode=='ready':
    require(not (root/'drain.request.json').exists() and not terminal()[0],'Stopped campaign cannot accept readiness')
    require(payload.get('schema')=='astra_broker_ready_v1' and payload.get('model')=='gpt-6-astra'
            and payload.get('reasoning_effort')=='high' and payload.get('remote_output')==str(root)
            and payload.get('config_sha256')==config_sha and payload.get('queue_read_succeeded') is True
            and type(payload.get('local_owner',{}).get('pid')) is int and payload['local_owner']['pid']>0,'Bad ready receipt')
    if (root/'broker-ready.json').exists():
        existing=read(root/'broker-ready.json')
        require(existing.get('config_sha256')==config_sha and existing.get('remote_output')==str(root),'Foreign existing readiness')
    print(json.dumps(publish(root/'broker-ready.json',payload))); sys.exit(0)
if mode=='skip':
    target=folder(payload['request_id']); request=target/'request.json'
    require(sha(request)==payload.get('request_sha256') and payload.get('config_sha256')==config_sha
            and payload.get('schema')=='astra_observation_technical_skip_v1' and payload.get('cli_executed') is False,'Invalid technical skip identity')
    if (target/'response.json').exists():
        print(json.dumps({'result':'response_already_exists'})); sys.exit(0)
    if (target/'technical-skip.json').exists():
        existing=read(target/'technical-skip.json')
        require(existing.get('request_sha256')==payload['request_sha256'],'Existing skip references another request')
    print(json.dumps(publish(target/'technical-skip.json',payload))); sys.exit(0)
require(mode=='gate','Unknown lifecycle operation')
ended,why=terminal(); rows=[]; now=time.time()
names=payload.get('request_ids')
entries=[queue/name for name in names] if names is not None else list(queue.iterdir()) if queue.exists() else []
for entry in entries:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,180}--astra',entry.name): continue
    target=folder(entry.name); request=target/'request.json'
    if not request.exists() or (target/'response.json').exists(): continue
    require(entry.name in jobs and jobs[entry.name]['rescue_arm']=='astra','Request not in continuation manifest')
    require(not request.is_symlink(),'Request is a symlink')
    age=max(0.0,now-request.stat().st_mtime)  # Immutable request timestamp, same server clock.
    live,owner_reason=alive(entry.name)
    assistance_path=root/'episodes'/entry.name/'assistance.json'
    assistance=read(assistance_path) if assistance_path.exists() else {}
    status=assistance.get('status'); reason=None; classification=None
    if (target/'technical-skip.json').exists(): reason='already_technically_skipped'; classification='existing_skip'
    elif (root/'episodes'/entry.name/'result.json').exists() or (root/'errors'/(entry.name+'.json')).exists():
        reason='episode_terminal_artifact_present'; classification='skipped_terminal_request'
    elif ended: reason='queue_terminal'; classification='skipped_terminal_request'
    elif not live: reason=owner_reason; classification='skipped_unverified_request'
    elif age>=config['response_wait_seconds']: reason='response_wait_budget_expired'; classification='skipped_expired_request'
    elif status!='waiting': reason='assistance_not_waiting'; classification='skipped_terminal_request'
    rows.append({'request_id':entry.name,'request_sha256':sha(request),'mtime_ns':request.stat().st_mtime_ns,
                 'eligible':reason is None,'technical_status':classification,'reason':reason,
                 'request_age_seconds':age,'remaining_wait_seconds':max(0,config['response_wait_seconds']-age),
                 'owner_live':live,'owner_check':owner_reason,'assistance_status':status,'observed_remote_unix':now})
print(json.dumps({'state':'terminal' if ended else 'active','terminal_reason':why,'config_sha256':config_sha,
                  'requests':sorted(rows,key=lambda row:(row['mtime_ns'],row['request_id']))}))
'''


class LiveQueue(ResilientQueue):
    def __init__(self, remote, local, sleep=time.sleep):
        super().__init__(remote, local, sleep)
        self.local=Path(local); self.config_sha=""; self.latest=None

    def guard(self, mode, payload=None):
        return self.retry("ssh_lifecycle_"+mode, lambda: broker.SSHQueue.remote(
            self, REMOTE_GUARD, mode, self.config_sha, json.dumps(payload or {},ensure_ascii=False)))

    def scan(self, names=None):
        snapshot=self.guard("gate", {} if names is None else {"request_ids":names})
        self.latest=snapshot
        broker.write_json(self.local/"queue-lifecycle-latest.json",snapshot)
        for row in snapshot["requests"]:
            if row["eligible"] or row["technical_status"]=="existing_skip": continue
            path=self.local/"technical_skips"/(row["request_id"]+".json")
            if path.exists(): value=broker.read_json(path)
            else:
                value=dict(schema="astra_observation_technical_skip_v1",status=row["technical_status"],
                           request_id=row["request_id"],request_sha256=row["request_sha256"],
                           config_sha256=self.config_sha,reason=row["reason"],evidence=row,
                           cli_executed=False,model_failure=False,counts_as_rescue=False,created_at=broker.utc())
                broker.write_json(path,value)
            publication=self.guard("skip",value)
            broker.write_json(path.with_name(path.stem+".publication.json"),publication)
        return snapshot

    def pending(self):
        return [row["request_id"] for row in self.scan()["requests"] if row["eligible"]]


class LiveBroker(broker.Broker):
    def finish(self, batch):
        if batch is None: return
        if not (batch/"receipt.json").exists() and not (batch/"cli_started.json").exists():
            if time.monotonic()>=self.deadline:
                broker.write_json(batch/"deferred-run-budget.json",dict(at=broker.utc(),cli_executed=False))
                return
            manifest=broker.read_json(batch/"batch.json")
            snapshot=self.queue.scan([row["request_id"] for row in manifest["requests"]])
            if time.monotonic()>=self.deadline:
                broker.write_json(batch/"deferred-run-budget.json",dict(at=broker.utc(),cli_executed=False))
                return
            eligible={row["request_id"] for row in snapshot["requests"] if row["eligible"]}
            kept=[row for row in manifest["requests"] if row["request_id"] in eligible]
            if len(kept)!=len(manifest["requests"]):
                if not (batch/"manifest-before-lifecycle-filter.json").exists():
                    broker.write_json(batch/"manifest-before-lifecycle-filter.json",manifest)
                manifest["excluded_before_cli"]=[row["request_id"] for row in manifest["requests"] if row["request_id"] not in eligible]
                manifest["requests"]=kept
                broker.write_json(batch/"batch.json",manifest)
            if not kept:
                broker.write_json(batch/"completed.json",dict(completed_at=broker.utc(),cli_executed=False,technical_skip_only=True))
                return
        super().finish(batch)  # Existing receipts always publish; they never invoke CLI again.

    def run(self, run_seconds, once=False, poll_seconds=3):
        self.deadline=time.monotonic()+run_seconds
        if (self.local/"stop.json").exists():
            self.status("stopped",stop=broker.read_json(self.local/"stop.json")); return 2
        self.recover()
        names=[]; fill_started=None; reason="run_budget_exhausted"
        while time.monotonic()<self.deadline and not (self.local/"stop.json").exists():
            processed={path.stem for path in (self.local/"receipts").glob("*.json")}
            pending=self.queue.pending()
            if self.queue.latest["state"]=="terminal": reason="queue_terminal"; break
            names=[name for name in names if name in pending]
            if not names: fill_started=None
            for name in pending:
                if name not in processed and name not in names and len(names)<self.max_batch: names.append(name)
            now=time.monotonic()
            if names and fill_started is None: fill_started=now
            if now>=self.deadline: break
            if names and (len(names)>=self.max_batch or now-fill_started>=self.batch_wait):
                self.finish(self.prepare(names)); names=[]; fill_started=None
                if once: reason="one_batch_finished"; break
            self.status("waiting",pending_batch_size=len(names),termination_policy="no_new_cli_after_deadline_or_queue_terminal")
            time.sleep(min(poll_seconds,max(0,self.deadline-time.monotonic())))
        stopped=(self.local/"stop.json").exists()
        self.status("stopped" if stopped else "finished",termination_reason="local_stop" if stopped else reason,
                    technical_skips=len(list((self.local/"technical_skips").glob("*.json")))//2)
        return 2 if stopped else 0


def resolve_operator_pause(local, expected_sha):
    path=local/"stop.json"; raw=path.read_bytes(); value=json.loads(raw)
    if broker.digest(raw)!=expected_sha or value.get("kind")!="operator_transport_review_pause":
        raise RuntimeError("Operator pause kind/hash differs from the explicitly reviewed stop")
    archive=local/"resolved_operator_pauses"/uuid.uuid4().hex
    archive.mkdir(parents=True)
    (archive/"stop.original.json").write_bytes(raw)
    broker.write_json(archive/"resolution.json",dict(explicit_stop_sha256=expected_sha,at=broker.utc(),
                      remote_drain_preserved=True,phase="saved_before_move"))
    if path.read_bytes()!=raw: raise RuntimeError("Stop changed during explicit review")
    path.rename(archive/"stop.moved.json")


def wait_preflight(queue, seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        try: value=queue.guard("bind")
        except Exception as error:
            # Network failures have already exhausted their bounded retries.
            broker.write_json(queue.local/"startup-error.json",dict(at=broker.utc(),error=repr(error)))
            raise
        broker.write_json(queue.local/"startup-status.json",dict(at=broker.utc(),binding=value))
        if value["state"] in ("ready","terminal"): return value
        time.sleep(min(60,max(0,deadline-time.monotonic())))
    raise RuntimeError("Continuation preflight was not ready within the loading budget")


def self_test():
    root=HERE/"broker_tests"/("live-v2-self-test-"+uuid.uuid4().hex[:12]); root.mkdir(parents=True)
    project=root/"project"; output=project/"results/astra-native-reset-continuation-20261009-v2"
    output.mkdir(parents=True); prior=project/"results/astra-native-reset-full-20261009-v1"; prior.mkdir()
    def write(path,value): broker.write_json(path,value)
    def sha(path): return broker.digest(path.read_bytes())
    prior_config=prior/"config.json"; write(prior_config,{"fixture":True})
    source=project/"source.py"; source.write_text("# fixture\n")
    failure=project/"failure.json"; write(failure,{"fixture":True})
    selected=[f"native-B-{index:04d}-Fixture-1" for index in range(968)]
    excluded=[f"prior-{index}" for index in range(36)]
    ledger_path=output/"continuation-exclusion-ledger.json"
    ledger=dict(schema="astra_continuation_exclusion_ledger_v2",fixed_denominator=1004,selected_case_ids=selected,
                excluded_case_ids=excluded,selection_uses_success=False,quota_retry_included=False,artifacts_sha256={})
    write(ledger_path,ledger)
    manifest_path=output/"manifest.json"
    jobs=[dict(id=name+"--"+arm,original_id=name,rescue_arm=arm) for name in selected for arm in ("control","astra")]
    write(manifest_path,dict(jobs=jobs,arms=["control","astra"],selected_original_jobs=[dict(id=name) for name in selected]))
    config=dict(schema="astra_native_reset_control_rescue_config_v1",output=str(output),response_wait_seconds=1200,
                protocol=dict(policy_wait_keepalive_seconds=120),manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
                source_sha256={str(source):sha(source)},failure_manifest=str(failure),failure_manifest_sha256=sha(failure),
                continuation=dict(schema="astra_native_reset_continuation_v2",prior_output=str(prior),
                prior_config_sha256=sha(prior_config),selected_cases=968,excluded_cases=36,selection_uses_success=False,
                quota_retry_included=False,exclusion_ledger=str(ledger_path),exclusion_ledger_sha256=sha(ledger_path)))
    write(output/"config.json",config); write(output/"preflight.json",dict(passed=True))
    proc=root/"proc"/"123"; proc.mkdir(parents=True)
    (proc/"stat").write_text("123 (fixture) "+" ".join(["S"]+["0"]*18+["777"]))
    names=[name+"--astra" for name in selected[:3]]
    for name in names:
        write(output/"requests"/name/"request.json",dict(fixture=True))
        write(output/"episodes"/name/"owner.json",dict(pid=123,process_start_ticks=777))
        write(output/"episodes"/name/"assistance.json",dict(status="waiting"))
    write(output/"episodes"/names[1]/"result.json",dict(technical_fixture=True))
    os.utime(output/"requests"/names[2]/"request.json",(time.time()-1300,time.time()-1300))
    script=REMOTE_GUARD.replace("pathlib.Path('"+broker.PROJECT+"')","pathlib.Path("+repr(str(project))+")")
    script=script.replace("pathlib.Path('/proc')","pathlib.Path("+repr(str(root/"proc"))+")")
    def run(mode,payload=None):
        result=subprocess.run([os.sys.executable,"-",str(output),mode,"",json.dumps(payload or {})],
                              input=script.encode(),capture_output=True,timeout=10,check=False)
        if result.returncode: raise AssertionError(result.stderr.decode())
        return json.loads(result.stdout)
    assert run("bind")["selected_cases"]==968
    rows=run("gate")["requests"]; byid={row["request_id"]:row for row in rows}
    assert byid[names[0]]["eligible"] and byid[names[1]]["technical_status"]=="skipped_terminal_request"
    assert byid[names[2]]["technical_status"]=="skipped_expired_request"  # Fresh keepalive file must not reset request age.
    skip=dict(schema="astra_observation_technical_skip_v1",request_id=names[1],request_sha256=byid[names[1]]["request_sha256"],
              config_sha256=sha(output/"config.json"),cli_executed=False,status="skipped_terminal_request")
    assert run("skip",skip)["result"]=="created"
    assert not (output/"requests"/names[1]/"response.json").exists()
    write(output/"completion.json",dict(all_children_drained=True))
    assert run("gate")["state"]=="active"
    (proc/"stat").unlink()
    assert run("gate")["state"]=="terminal"
    local=root/"pause"; local.mkdir(); write(local/"stop.json",dict(kind="operator_transport_review_pause",message="fixture"))
    resolve_operator_pause(local,sha(local/"stop.json")); assert not (local/"stop.json").exists()
    from unittest.mock import patch
    class FakeQueue:
        remote_output=CONTINUATION
        def scan(self,names): return dict(requests=[dict(request_id=name,eligible=False) for name in names])
    state=root/"model_gate"; instance=LiveBroker(state,FakeQueue(),["NEVER_EXECUTE"])
    batch=state/"batches"/"fixture"; write(batch/"batch.json",dict(batch_id="fixture",requests=[dict(request_id=names[0])]))
    instance.deadline=time.monotonic()+30
    with patch.object(broker,"run_cli",side_effect=AssertionError("No CLI for expired/terminal input")):
        instance.finish(batch)
    assert (batch/"completed.json").exists() and not (batch/"cli_started.json").exists()
    receipt=dict(status="passed",wrapper_sha256=sha(Path(__file__)),frozen_broker_sha256=sha(HERE/"broker.py"),
                 no_real_ssh_or_model_calls=True,tests=["968/1936 continuation binding","live owner admission",
                 "terminal request exclusion","immutable request age despite fresh keepalive","technical skip leaves response absent",
                 "terminal requires no live owners","explicit operator pause archive","pre-CLI lifecycle filter prevents model call"])
    write(root/"self_test_result.json",receipt)
    print(json.dumps(dict(receipt,receipt_path=str(root/"self_test_result.json"))),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--remote-output",default=CONTINUATION)
    parser.add_argument("--initialize-new-state",action="store_true")
    resolutions=parser.add_mutually_exclusive_group()
    resolutions.add_argument("--resolve-network-stop",action="store_true")
    resolutions.add_argument("--resolve-operator-pause-sha256")
    parser.add_argument("--recover-saved-only",action="store_true")
    parser.add_argument("--run-seconds",type=float,default=30000)
    parser.add_argument("--wait-preflight-seconds",type=float,default=1800)
    parser.add_argument("--batch-wait-seconds",type=float,default=60)
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    if args.self_test: return self_test()
    if args.local_output is None: parser.error("--local-output is required")
    local=args.local_output.resolve(); remote=broker.safe_remote(args.remote_output)
    if os.name=="nt" and local.drive.upper()!="D:": parser.error("State must remain on D:")
    if broker.digest((HERE/"broker.py").read_bytes())!=FROZEN_SHA: parser.error("Frozen broker changed")
    if args.run_seconds<=0 or not 0<=args.batch_wait_seconds<=60 or not 0<args.wait_preflight_seconds<=1800:
        parser.error("Invalid bounded timing options")
    if args.initialize_new_state:
        if remote!=CONTINUATION or local.exists() or args.recover_saved_only:
            parser.error("New state requires a nonexistent local directory and exact continuation output")
        local.mkdir(parents=True)
    elif not (local/"identity.json").exists(): parser.error("Existing identity required; use explicit initialization for the new continuation")
    if not CODEX.is_file(): parser.error("Pinned Codex executable unavailable")
    os.environ["PATH"]=str(CODEX.parent)+os.pathsep+os.environ.get("PATH","")
    with broker.ProcessLock(local/"broker.lock"):
        if not args.initialize_new_state:
            identity=broker.read_json(local/"identity.json")
            if args.recover_saved_only: remote=identity["remote_output"]
            expected=dict(schema="astra_broker_identity_v1",remote_output=remote,ssh_host=broker.HOST,model=broker.MODEL,reasoning_effort=broker.EFFORT)
            if identity!=expected: parser.error("Existing state/model/remote identity differs")
        if args.resolve_network_stop: resolve_network_stop(local)
        if args.resolve_operator_pause_sha256: resolve_operator_pause(local,args.resolve_operator_pause_sha256)
        if (local/"stop.json").exists():
            print(json.dumps(dict(state="stopped",stop=broker.read_json(local/"stop.json")))); return 2
        if args.recover_saved_only:
            instance=broker.Broker(local,ResilientQueue(remote,local),[str(CODEX)])
            for manifest in (local/"batches").glob("*/batch.json"):
                if not (manifest.parent/"completed.json").exists() and not (manifest.parent/"receipt.json").exists():
                    raise RuntimeError("Saved-only recovery refuses a batch without a completed CLI receipt")
            instance.recover(); instance.status("finished",termination_reason="saved_response_recovery_only"); return 0
        if remote!=CONTINUATION: parser.error("New CLI calls are forbidden for the old campaign")
        queue=LiveQueue(remote,local)
        binding=wait_preflight(queue,args.wait_preflight_seconds)
        if binding["state"]=="terminal":
            broker.write_json(local/"startup-status.json",dict(state="queue_already_terminal",binding=binding)); return 0
        if args.initialize_new_state and (binding["existing_ready"] is not None or binding["request_count"]):
            raise RuntimeError("Remote queue already started; do not initialize another state and repeat prior calls")
        queue.config_sha=binding["config_sha256"]
        instance=LiveBroker(local,queue,[str(CODEX)],batch_wait=args.batch_wait_seconds)
        broker.write_json(local/"continuation-binding.json",binding)
        try:
            pending=queue.pending()
            if queue.latest["state"]=="terminal":
                instance.status("finished",termination_reason="queue_terminal_before_ready"); return 0
            owner=WindowsProcesses().identity(os.getpid())
            ready=dict(schema="astra_broker_ready_v1",model=broker.MODEL,reasoning_effort=broker.EFFORT,
                       remote_output=remote,config_sha256=queue.config_sha,queue_read_succeeded=True,
                       local_owner=owner,wrapper_sha256=broker.digest(Path(__file__).read_bytes()),
                       frozen_broker_sha256=FROZEN_SHA,checked_at=broker.utc(),pending_count=len(pending))
            broker.write_json(local/"v2-ready-payload.json",ready)
            broker.write_json(local/"v2-ready-receipt.json",queue.guard("ready",ready))
            return instance.run(args.run_seconds,args.once)
        except Exception as error:
            instance.stop("broker_infrastructure_error",repr(error)); return 1


if __name__=="__main__":
    raise SystemExit(main())
