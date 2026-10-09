"""External GR00T formal-service guard; --check is read-only, --run explicit.

Only the frozen original GR00T server and its socket parity probe can be spawned.
The guard never trains, starts a study/rollout/broker, invokes Astra, grants
scientific approval, or mutates any frozen policy/study source. Linux only for
execution; all preconditions must be backed by externally supplied real files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import traceback
sys.dont_write_bytecode=True
# Exact implementation below copied from the independently tested GR00T v2
# processes.py; construction performs no reads before child registration.
TRACKED_CHILD_ORIGIN_SHA256='89bf3178e1e71ff0cbb19e0d3bbc1c6bc880b27450a12e9ec424ff6b41d2c6d8'
from pathlib import Path
import os
import signal
import time
import subprocess


def require(value,message):
    if not value:raise RuntimeError(message)


def birth(pid):
    fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,parent_pid=int(fields[1]),process_start_ticks=int(fields[19]),state=fields[0])


def identity(pid):
    p=Path('/proc')/str(pid);b=birth(pid)
    return dict(pid=pid,process_start_ticks=b['process_start_ticks'],
        command=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),cwd=str((p/'cwd').resolve(strict=True)))


def open_pidfd(pid):
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux pidfd support required')
    return os.pidfd_open(pid)


def send_pidfd(fd,sig):signal.pidfd_send_signal(fd,sig)


def close_pidfd(fd):os.close(fd)


class TrackedChild:
    def __init__(self,child,command,cwd):
        self.child=child;self.command=list(command);self.cwd=str(cwd);self.pid=child.pid
        self.parent_pid=os.getpid();self.start=None;self.pidfd=None;self.owner=None
        self.samples=[];self.signals=[];self.reaped=False;self.startup_error=None

    def _birth(self):
        current=birth(self.pid)
        require(current['pid']==self.pid and current['parent_pid']==self.parent_pid,
                'Spawn is no longer the direct unreaped child; no signal permitted')
        if self.start is None:self.start=current['process_start_ticks']
        require(current['process_start_ticks']==self.start,'Spawn start ticks changed; no signal permitted')
        return current

    def _attach(self):
        if self.child.poll() is not None:return False
        if self.pidfd is None:
            # Popen has not reaped this still-live direct child, preventing PID
            # reuse while the kernel handle is acquired and its birth checked.
            self.pidfd=open_pidfd(self.pid)
        self._birth()
        return True

    def await_identity(self,timeout=5.0,interval=.02):
        deadline=time.monotonic()+timeout;previous=None;matches=0
        try:
            if not self._attach():raise RuntimeError('Child exited before startup identity binding')
            while time.monotonic()<deadline:
                if self.child.poll() is not None:raise RuntimeError('Child exited before startup identity binding')
                self._birth()
                try:
                    observed=identity(self.pid);self._birth()
                    valid=(observed['pid']==self.pid and observed['process_start_ticks']==self.start
                        and observed['command']==self.command and observed['cwd']==self.cwd)
                    self.samples.append(dict(observed=observed,expected_identity=valid))
                    self.samples=self.samples[-16:]
                    matches=matches+1 if valid and previous==observed else 1 if valid else 0
                    previous=observed
                    if matches>=2:self.owner=observed;return observed
                except OSError as error:
                    matches=0;previous=None
                    self.samples.append(dict(read_error=str(error)));self.samples=self.samples[-16:]
                time.sleep(interval)
            raise TimeoutError('Bounded child identity handshake expired; pending spawn must be drained')
        except BaseException as error:
            self.startup_error=repr(error)
            raise

    def send(self,sig):
        if not self._attach():return False
        self._birth()
        send_pidfd(self.pidfd,sig)
        self.signals.append(dict(signal=int(sig),pidfd=True,bound_owner=self.owner is not None))
        return True

    def drain(self,grace=15,kill_timeout=10):
        """Wait/reap every spawn, even if the complete identity was never bound."""
        try:
            if self.child.poll() is None:self.send(signal.SIGTERM)
            try:code=self.child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.send(signal.SIGKILL);code=self.child.wait(timeout=kill_timeout)
            self.reaped=True
            return code
        finally:
            if self.child.returncode is not None:self.close()

    def close(self):
        require(self.child.poll() is not None,'Cannot release a live tracked child')
        self.child.wait(timeout=0);self.reaped=True
        if self.pidfd is not None:close_pidfd(self.pidfd);self.pidfd=None

    def receipt(self):
        return dict(schema='originx_gr00t_spawn_handshake_v2',pid=self.pid,parent_pid=self.parent_pid,
            process_start_ticks=self.start,requested_command=self.command,requested_cwd=self.cwd,
            owner=self.owner,identity_bound=self.owner is not None,samples=self.samples,
            signals=self.signals,reaped=self.reaped,returncode=self.child.returncode,startup_error=self.startup_error)


ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
BASE=ROOT/'originx_gr00t_confirmation_20261009'
OUT=ROOT/'results/originx-gr00t-formal-service-guard-20261009-v1'
MAIN=ROOT/'results/originx-confirmatory-20261009-v1'
MAIN_SERVICES=ROOT/'results/originx-confirmatory-services-20261009-v2'
DEV=ROOT/'results/originx-gr00t-confirmatory-development-20261009-v2'
DIRECT=BASE/'parity-direct-v2/direct.json'
DIRECT_SHA='ae36943e5db8d9d27532dbea74b6a731e890c7374d2145f05a4aa8bca945ddf9'
IDENTITY_SHA='90811c0ed4668f634c2beb04ba13bc208ab120a5391aa972c30b67e8b07bee93'
SOURCE_COMMIT='9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10'
GPU_UUID='GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'
PROFILE=BASE/'services/port-27903/server.json'
PARITY=BASE/'parity-socket-formal-v1'
MAX_LEASE=5*86400+3600
DRAIN_RESERVE=60  # two owned children, each TERM 15s + KILL 10s, plus loop margin
CAPS=dict(cli_calls=1500,input_tokens=25000000,output_tokens=2000000)
STUDIES=('originx-confirmatory-development-20261009-v2','originx-confirmatory-20261009-v1',
 'originx-gr00t-confirmatory-development-20261009-v1','originx-gr00t-confirmatory-development-20261009-v2')
SERVICE_SHA={
 '__init__.py':'fb769a894e28f0ffd1ac6081745d1a776e90473cc339102a3569595c613e5f85',
 'adapter.py':'daf0b5333cc8430e2330a7e6d4d2482269be51eaec14a405d33aea03c6e24b5b',
 'wire.py':'5e94665f8fe25a49abf732ee335ff579fa2f2583c13917a2677db6d84a7ee5b8',
 'core.py':'3a0df5483786611743500330038d1291bca7141e4d6a13da5e3a5f4dba006bd9',
 'server.py':'dd9564cf9a4be486de866cc631f79528d707c1e8f3a307999c9b5a40f74f6565',
 'client.py':'8d50db9fa8fc0d3193658a7bb7124d310ab342cd0874ec0b379ef1d87fe58d38',
 'parity_probe.py':'e703d4db2dfb39d4e85dd987bbc80d6b201644624159ddcf3278d649621684c1'}
ASSETS={
 'config.json':'6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526',
 'model.safetensors.index.json':'bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746',
 'experiment_cfg/metadata.json':'8be0fc606c9220356bad497feefc4ae4daaa05670acccc31caecaa7ae5590b69',
 'model-00001-of-00002.safetensors':'08f1891947973e2e5ec2422201cd90261806f77f2634f9ec0477c27aa5a4fe42',
 'model-00002-of-00002.safetensors':'deb9c9cf40cd8983a7779af85341f6344db15f65bf3307073a0e3c3085450435'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def scoped(path,root=None):
    root=ROOT if root is None else root
    p=Path(path)
    require(p.is_absolute() and p.resolve()==p and p.is_relative_to(root),'Unscoped/symlink evidence path: '+str(p))
    return p


def pinned(ref,root=None):
    require(set(ref)=={'path','sha256'},'Evidence reference must contain path and SHA256')
    p=scoped(ref['path'],root);require(sha(p)==ref['sha256'],'Evidence hash differs: '+str(p))
    return read(p)


def run_read(argv):return subprocess.check_output(argv,text=True,timeout=30).strip()


def owner_inactive(owner):
    required={'pid','process_start_ticks','command','cwd'}
    require(required<=set(owner) and type(owner['pid']) is int and type(owner['process_start_ticks']) is int,'Malformed owner')
    try:
        current=birth(owner['pid'])
        if current['state'] in ('Z','X'):return True
        actual=identity(owner['pid'])
    except (FileNotFoundError,ProcessLookupError):return True
    if actual['process_start_ticks']!=owner['process_start_ticks']:return True
    require(actual['pid']==owner['pid'] and actual['command']==owner['command'] and actual['cwd']==owner['cwd'],
        'Live owner with same birth identity changed argv/cwd; termination is not proven')
    return False


def validate_gpu(snapshot,allowed=()):
    require(snapshot['gpu_uuid']==GPU_UUID and snapshot['ecc']==0,'GPU6 UUID/ECC mismatch')
    require(set(snapshot['occupants'])<=set(allowed),'GPU6 has another or unregistered process; no launch')
    require(snapshot['free_mib']>=16384,'GPU6 needs at least 16 GiB free')
    return snapshot


def gpu_snapshot():
    row=run_read(['nvidia-smi','-i','6','--query-gpu=uuid,memory.free,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits']).split(',')
    occupants=[]
    for line in run_read(['nvidia-smi','-i','6','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits']).splitlines():
        if line.strip():
            pid,uuid=map(str.strip,line.split(','));require(uuid==GPU_UUID,'GPU occupant belongs to wrong UUID');occupants.append(int(pid))
    return dict(gpu_uuid=row[0].strip(),free_mib=int(row[1]),ecc=int(row[2]),occupants=occupants)


def terminal_inventory(ref,terminal):
    parent=scoped(ref['path']).parent;expected=terminal.get('files_sha256',{})
    require(bool(expected) and terminal.get('excluded_non_evidence_files')==['broker.lock'],'Unsealed broker inventory')
    actual={}
    for path in parent.rglob('*'):
        require(not path.is_symlink(),'Symlink in sealed broker inventory')
        if path.is_file() and path!=Path(ref['path']) and path!=parent/'broker.lock':
            actual[path.relative_to(parent).as_posix()]=sha(path)
    require(actual==expected,'Terminal inventory incomplete, stale, or changed')
    return expected


def validate_budget(debit,load=pinned,inventory=terminal_inventory):
    require(debit.get('schema')=='originx_shared_budget_debit_v1' and debit.get('all_prior_brokers_inactive') is True
        and debit.get('complete_input_output_token_accounting') is True,'Shared debit incomplete or brokers live')
    prior=debit.get('prior_usage',{});require(set(prior)==set(CAPS) and all(type(v)is int and v>=0 for v in prior.values()),'Invalid prior counters')
    studies=debit.get('studies',[])
    require(len(studies)==len(STUDIES) and {s['study'] for s in studies}==set(STUDIES),'All four prior studies, including failed v1, must be accounted')
    seen=set();total={k:0 for k in CAPS};zero_proofs=[]
    for study in studies:
        usage=load(study['usage']);terminal=load(study['terminal'])
        require(terminal.get('schema')=='originx_confirmatory_broker_terminal_audit_v1'
            and all(terminal.get(k)is True for k in ('broker_owner_inactive','all_cli_wait_returns_terminal','all_owned_cli_processes_absent','no_cli_invoked_by_audit'))
            and terminal.get('scoped_cli_process_matches')==[],'Missing terminal broker audit')
        require(terminal.get('broker_identity',{}).get('remote_output')==str(ROOT/'results'/study['study']),'Budget terminal belongs to another study')
        files=inventory(study['terminal'],terminal)
        calls=usage.get('calls',[]);n=usage.get('unique_observed_cli_invocations')
        require(type(n)is int and n==len(calls)==terminal.get('cli_starts')==len(terminal.get('verified_cli_wait_returns',[])),'CLI count incomplete')
        require(usage.get('broker_inventory_provided') is True and usage.get('invalid_receipts')==[]
            and usage.get('interrupted_cli_starts_without_receipt')==[],'Invalid or interrupted usage inventory')
        # The first GR development aborted before any CLI; its raw aggregate
        # retains null totals/false completeness. Only the separately sealed,
        # empty terminal inventory can establish actual zero calls/tokens.
        zero=(study['study']==STUDIES[2] and n==0 and not any(
            Path(name).name in ('cli_started.json','receipt.json') or any(part in ('batches','receipts') for part in Path(name).parts) for name in files))
        if zero:
            terminal_root=Path(study['terminal']['path']).parent
            require('broker_status.json' in files,'Zero calls lack a sealed broker status')
            status=load(dict(path=str(terminal_root/'broker_status.json'),sha256=files['broker_status.json']))
            require(type(status.get('cli_calls'))is int and status['cli_calls']==0,'Zero-call broker status is not explicit')
            proof=load(study['zero_invocation_evidence'])
            require(proof.get('schema')=='originx_gr00t_failed_development_terminal_audit_v1'
                and all(proof.get(k)is True for k in ('campaign_failed','original_completion_preserved','all_recorded_worker_and_model_identities_inactive'))
                and bool(proof.get('checks')),'Missing failed development zero-call proof')
            for check in proof['checks']:
                require(check.get('inactive') is True and owner_inactive(check['recorded_owner']),'Failed development owner still live')
            zero_proofs.append(study['study'])
        else:
            require(usage.get('inventory_complete') is True and usage.get('complete_input_output_token_accounting') is True,
                'Incomplete actual usage inventory')
        known=usage.get('known_usage_totals',{});subtotal=dict(input_tokens=0,output_tokens=0)
        receipt_hashes={r['receipt_sha256'] for r in terminal['verified_cli_wait_returns']}
        require(len(receipt_hashes)==n,'Duplicate terminal receipts')
        for call in calls:
            key=call.get('invocation_key');require(isinstance(key,str) and len(key)==64 and key not in seen,'Duplicate/invalid invocation across shared budget')
            seen.add(key);digest=call.get('receipt_sha256')
            require(digest in receipt_hashes and call.get('transcript_problems')==[],'Unverified invocation receipt')
            terminal_root=Path(study['terminal']['path']).parent
            relocated=study.get('raw_receipts',{}).get(digest)
            if relocated is not None:
                require(set(relocated)=={'path','sha256'} and relocated['sha256']==digest,'Malformed relocated raw receipt binding')
                paths=[Path(relocated['path'])]
            else:paths=[Path(path) for path in call.get('source_paths',[]) if Path(path).is_relative_to(terminal_root)]
            require(bool(paths) and all(path.is_relative_to(terminal_root) for path in paths),'No raw receipt in sealed broker inventory')
            for path in paths:
                require(files.get(path.relative_to(terminal_root).as_posix())==digest,'Receipt not sealed in terminal inventory')
                raw=load(dict(path=str(path),sha256=digest))
                require(raw.get('schema')=='astra_cli_receipt_v1' and raw.get('cli_executed') is True
                    and raw.get('model')=='gpt-6-astra' and raw.get('reasoning_effort')=='high','Wrong CLI receipt identity')
                require(all(raw.get('usage_totals',{}).get(k)==call.get('usage',{}).get(k) for k in subtotal),
                    'Actual receipt token counters differ')
            for k in subtotal:
                v=call.get('usage',{}).get(k);require(type(v)is int and v>=0,'Missing actual token counter');subtotal[k]+=v
        require(all(known.get(k) in (None,0) if zero else known.get(k)==v for k,v in subtotal.items()),'Usage totals do not equal actual invocations')
        require(all(usage.get('observed_calls_missing_counters',{}).get(k)==0 for k in subtotal),'Unknown token counters are not zero')
        total['cli_calls']+=n
        for k,v in subtotal.items():total[k]+=v
    require(total==prior,'Shared debit differs from complete deduplicated inventory')
    remaining={k:CAPS[k]-v for k,v in total.items()};require(all(v>0 for v in remaining.values()),'Shared budget exhausted; no model start')
    return dict(caps=CAPS,prior_usage=total,remaining=remaining,zero_calls_proven_by_sealed_inventory=zero_proofs)


def verify_prerequisites(plan):
    require(plan.get('schema')=='originx_gr00t_formal_service_plan_v1','Wrong guard plan')
    require(set(plan)=={'schema','development_admission','main_finished','main_cleanup','main_services','shared_budget'},'Unexpected plan fields')
    require(Path(plan['development_admission']['path'])==DEV/'admission-review.json','Only independent GR development v2 admission')
    gate=pinned(plan['development_admission'])
    require(gate.get('schema')=='originx_gr00t_development_admission_v1' and gate.get('passed') is True
        and gate.get('study_namespace')=='originx_gr00t_confirmation_20261009_v2'
        and gate.get('no_confirmatory_outcomes_seen') is True and gate.get('all_owned_models_stopped') is True
        and gate.get('raw_verified_arm_outcomes')==4 and gate.get('matched_case_pairs')==2 and gate.get('verified_V_applications',0)>=1,
        'Independent development admission not passed')
    bindings=gate.get('bindings_sha256',{});require(bool(bindings),'Missing development evidence bindings')
    for name,digest in bindings.items():require(sha(scoped(name))==digest,'Development evidence changed: '+name)
    for check in gate.get('owned_worker_checks',[]):
        require(check.get('stopped') is True and check['path'] in bindings,'Invalid development owner audit')
        require(owner_inactive(read(check['path'])),'Development owned process still live')
    require(bool(gate.get('owned_worker_checks')),'Empty development owner inventory')
    expected={'main_finished':MAIN/'campaign-finished.json','main_cleanup':MAIN/'cleanup.json','main_services':MAIN_SERVICES/'services-ready.json'}
    for key,path in expected.items():require(Path(plan[key]['path'])==path,'Wrong main study evidence path')
    finished=pinned(plan['main_finished']);cleanup=pinned(plan['main_cleanup']);ready=pinned(plan['main_services'])
    require(finished.get('development') is False and finished.get('phase') in ('completed','failed')
        and type(finished.get('failed')) is bool,'Main full study has no terminal receipt')
    require(cleanup.get('namespace')=='originx_confirmatory_20261009' and cleanup.get('all_owned_services_stopped') is True
        and cleanup.get('still_active_owned_pids')==[] and cleanup.get('evidence_preserved') is True,'Main models not cleaned up')
    require(ready.get('namespace')=='originx_confirmatory_20261009' and ready.get('gpu_uuid')==GPU_UUID and bool(ready.get('models')),'Wrong main services')
    owners=[]
    for model in ready['models']:
        owner=model['owner'];require(owner.get('namespace')=='originx_confirmatory_20261009' and owner.get('gpu_uuid')==GPU_UUID,'Foreign main owner')
        require(owner_inactive(owner),'Main owned model still live');owners.append(owner)
    require((MAIN/'campaign.owner.json').exists(),'Main campaign owner missing')
    owner_paths=set(MAIN.glob('*.owner.json'))|set((MAIN/'processes').glob('*.json'))|set((MAIN_SERVICES/'services').glob('*/owner.json'))
    for path in sorted(owner_paths):
        owner=read(path);require(owner_inactive(owner),'Main owned campaign/worker still live: '+str(path));owners.append(owner)
    budget=validate_budget(pinned(plan['shared_budget']))
    return dict(development_admission_sha256=plan['development_admission']['sha256'],main_owners_checked=len(owners),shared_budget=budget)


def verify_assets():
    require(sha(scoped(DIRECT))==DIRECT_SHA,'Pinned direct reference changed')
    direct=read(DIRECT);require(direct.get('passed') is True and direct.get('ambient_rng_unchanged') is True,'Invalid direct reference')
    ident=direct['identity'];require(ident.get('asset_source_identity_sha256')==IDENTITY_SHA
        and ident.get('service_sources_sha256')==SERVICE_SHA and ident.get('model_assets_sha256')==ASSETS
        and ident.get('official_source_commit')==SOURCE_COMMIT,'Wrong official policy/source binding')
    for name,digest in SERVICE_SHA.items():require(sha(scoped(BASE/name))==digest,'Frozen service source changed: '+name)
    for name,digest in ASSETS.items():require(sha(scoped(BASE/'checkpoint-120000'/name))==digest,'Official asset changed: '+name)
    source=BASE/'source';require(run_read(['git','-C',str(source),'rev-parse','HEAD'])==SOURCE_COMMIT,'Official commit changed')
    require(not run_read(['git','-C',str(source),'diff','HEAD','--','gr00t','pyproject.toml']),'Official source modified')
    for name,digest in ident['official_source_sha256'].items():require(sha(scoped(source/name))==digest,'Official processor/model source differs')
    fixture=Path(direct['fixtures']);require(sha(scoped(fixture))==direct['fixtures_sha256'],'Native parity fixture manifest changed')
    for row in read(fixture)['fixtures']:require(sha(scoped(row['path']))==row['sha256'],'Native parity fixture changed')
    return dict(asset_source_identity_sha256=IDENTITY_SHA,direct_reference_sha256=DIRECT_SHA)


def commands(profile_sha=None):
    python=str(BASE/'env/bin/python');module=BASE.name
    service=[python,'-u','-m',module+'.server','--port','27903','--max-clients','6']
    parity=None if profile_sha is None else [python,'-u','-m',module+'.parity_probe','socket','--output',str(PARITY),
        '--profile',str(PROFILE),'--profile-sha256',profile_sha,'--reference',str(DIRECT),'--reference-sha256',DIRECT_SHA,
        '--hold-seconds','360','--interval-seconds','120']
    return service,parity


def environment(gpu):
    # Build an allowlist rather than inherit LD_PRELOAD/PYTHONSTARTUP/CUDA overrides.
    allowed=('PATH','HOME','USER','LANG','LC_ALL','TMPDIR','CONDA_PREFIX','SSL_CERT_FILE')
    result={k:os.environ[k] for k in allowed if k in os.environ}
    result.update(PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=GPU_UUID if gpu else '',PYTHONHASHSEED='0',
        OMP_NUM_THREADS='4' if gpu else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    return result


def validate_profile(profile,tracked):
    hello=profile.get('hello',{});require(profile.get('ready') is True and profile.get('schema')=='originx_gr00t_service_profile_v1','Not a ready GR00T profile')
    require(profile.get('port')==27903 and profile.get('slots')==6 and profile.get('gpu_uuid')==GPU_UUID,'Wrong formal port/device/capacity')
    require(all(profile['owner'].get(k)==v for k,v in tracked.owner.items()),'Published owner does not match held Popen/pidfd')
    require(hello.get('protocol')=='originx-gr00t-rng-v1' and hello.get('policy_id')=='gr00t_n15_robocasa_multitask120000'
        and hello.get('asset_source_identity_sha256')==IDENTITY_SHA and hello.get('model_assets_sha256')==ASSETS
        and hello.get('adapters_loaded') is False and hello.get('denoising_steps')==4 and hello.get('replan_steps')==16,
        'Policy semantics changed')


class Supervisor:
    def __init__(self,output,lease_seconds):
        self.output=Path(output);self.deadline=time.monotonic()+lease_seconds;self.children=[];self.stop=False
    def interrupted(self,*_):self.stop=True
    def check_time(self):
        require(not self.stop,'Guard received termination signal')
        require(time.monotonic()<self.deadline,'Formal service lease expired')
    def spawn(self,label,command,gpu):
        service,parity=commands(sha(PROFILE) if PROFILE.exists() else None)
        require(type(gpu)is bool and gpu==(label=='service'),'Fixed child CUDA visibility differs')
        require(isinstance(command,list),'Child argv must be explicit list')
        require(command==(service if label=='service' else parity if label=='parity' else None),'Only fixed service/parity argv may spawn')
        path=self.output/label;path.mkdir()
        log=(path/'stdout.log').open('xb');error=(path/'stderr.log').open('xb')
        try:
            child=subprocess.Popen(command,cwd=BASE,env=environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=error,start_new_session=True)
            tracked=TrackedChild(child,command,BASE);self.children.append((label,tracked))
            try:tracked.await_identity(timeout=5)
            finally:write_new(path/'handshake.json',tracked.receipt())
            return tracked
        finally:log.close();error.close()
    def await_profile(self,service):
        deadline=min(self.deadline,time.monotonic()+900)
        while time.monotonic()<deadline:
            self.check_time();require(service.child.poll() is None,'Service exited before readiness')
            if PROFILE.exists():
                try:profile=read(PROFILE)
                except (json.JSONDecodeError,OSError):time.sleep(.2);continue
                validate_profile(profile,service);return profile
            time.sleep(.2)
        raise TimeoutError('Service readiness deadline expired')
    def wait_phase(self,tracked,timeout):
        deadline=min(self.deadline,time.monotonic()+timeout)
        while time.monotonic()<deadline:
            self.check_time()
            try:code=tracked.child.wait(timeout=min(1,max(.01,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:continue
            tracked.close();require(code==0,'Fixed child phase exited '+str(code));return
        raise TimeoutError('Fixed child phase timeout')
    def drain(self):
        failures=[]
        # Parity clients first; only this guard's unreaped direct children.
        for label,tracked in reversed(self.children):
            try:tracked.drain()
            except BaseException as error:failures.append(dict(label=label,error=repr(error)))
        receipt=dict(children=[dict(label=l,**t.receipt()) for l,t in self.children],failures=failures,
            all_children_reaped=all(t.reaped for _,t in self.children))
        write_new(self.output/'drain.json',receipt);require(receipt['all_children_reaped'] and not failures,'Owned child drain incomplete; preserve exact identities')
        return receipt


def execute(plan,plan_sha,lease_seconds):
    require(os.name=='posix' and hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux pidfd required')
    require(600<=lease_seconds<=MAX_LEASE,'Lease must be 600..435600 seconds')
    require(Path(__file__).resolve()==ROOT/'gr00t_formal_service_guard.py','Execute only the external project-root helper')
    require(not OUT.exists() and not PROFILE.parent.exists() and not PARITY.exists(),'Formal guard/27903/parity already exists; never overwrite or duplicate')
    prerequisites=verify_prerequisites(plan);assets=verify_assets();gpu=validate_gpu(gpu_snapshot())
    OUT.mkdir(parents=True,exist_ok=False);supervisor=Supervisor(OUT,lease_seconds-DRAIN_RESERVE);failure=None;result=None
    write_new(OUT/'owner.json',dict(identity(os.getpid()),helper_sha256=sha(__file__),plan_sha256=plan_sha,lease_seconds=lease_seconds))
    write_new(OUT/'preflight.json',dict(prerequisites=prerequisites,assets=assets,gpu=gpu,no_rollouts_or_Astra_started=True))
    old_handlers={sig:signal.signal(sig,supervisor.interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        # Recheck the real evidence and empty GPU immediately before first Popen.
        verify_prerequisites(plan);validate_gpu(gpu_snapshot())
        service=supervisor.spawn('service',commands()[0],True);profile=supervisor.await_profile(service)
        require(service.child.poll() is None,'Service exited before parity')
        validate_gpu(gpu_snapshot(),[service.pid]);profile_sha=sha(PROFILE)
        parity=supervisor.spawn('parity',commands(profile_sha)[1],False);supervisor.wait_phase(parity,900)
        result=read(PARITY/'result.json')
        require(result.get('passed') is True and result.get('clients')==6 and result.get('hold_seconds',0)>=360
            and result.get('profile_sha256')==profile_sha and result.get('reference_sha256')==DIRECT_SHA,
            'Fresh exact six-stream parity failed')
        write_new(OUT/'ready.json',dict(schema='originx_gr00t_formal_service_guard_ready_v1',service_profile=str(PROFILE),
            service_profile_sha256=profile_sha,service_owner=profile['owner'],parity_path=str(PARITY/'result.json'),
            parity_sha256=sha(PARITY/'result.json'),plan_sha256=plan_sha,retained_parent_pidfd=True,
            study_approval_created=False,starts_rollouts=False,starts_broker=False))
        # Root/heartbeat may now launch the separately admitted frozen study.
        # Its normal cleanup may terminate this service; retain the pidfd until reaping.
        while service.child.poll() is None:
            supervisor.check_time()
            release=OUT/'release-request.json'
            if release.exists():
                request=read(release);require(request.get('service_profile_sha256')==profile_sha,'Release request profile differs')
                require(not run_read(['ss','-Htn','( sport = :27903 )']),'Release refuses active or closing client sockets')
                break
            time.sleep(1)
    except BaseException:
        failure=traceback.format_exc();write_new(OUT/'first-error.json',dict(error=failure))
    finally:
        try:supervisor.drain()
        except BaseException:
            if failure is None:failure=traceback.format_exc();write_new(OUT/'first-error.json',dict(error=failure))
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        write_new(OUT/'finished.json',dict(failed=failure is not None,all_children_reaped=all(t.reaped for _,t in supervisor.children),
            parity_passed=bool(result and result.get('passed')),no_rollouts_or_Astra_started=True))
    return 1 if failure else 0


def main():
    p=argparse.ArgumentParser(description=__doc__);actions=p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check',action='store_true');actions.add_argument('--run',action='store_true')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--lease-seconds',type=int,default=MAX_LEASE);args=p.parse_args()
    plan=pinned(dict(path=str(args.plan),sha256=args.plan_sha256))
    if args.check:
        report=dict(prerequisites=verify_prerequisites(plan),assets=verify_assets(),gpu=validate_gpu(gpu_snapshot()),
            guard_source_sha256=sha(__file__),read_only=True,no_approval_written=True)
        print(json.dumps(report,indent=2));return 0
    return execute(plan,args.plan_sha256,args.lease_seconds)


if __name__=='__main__':raise SystemExit(main())
