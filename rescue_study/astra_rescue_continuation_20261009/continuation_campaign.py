"""Independent bounded supervisor for the remaining 968-case paired study; original denominator stays 1004.

Does not modify the frozen runner, broker, reference results or service launcher.
Run once from the remote project root with envs/training/bin/python. The broker
is a separate root-owned local process. A drain request stops new trial dispatch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
HERE = ROOT/'astra_rescue_continuation_20261009'
OUTPUT = ROOT/'results/astra-native-reset-continuation-20261009-v2'
SERVICES = ROOT/'results/astra-rescue-services-20261009-v2'
EXPAND_OWNER = SERVICES/'launcher-owner.json'
TRAINING_PYTHON = ROOT/'envs/training/bin/python'
BINDING = ROOT/'results/continued-ab2000-v1/B-endpoint.json'
BINDING_SHA = 'c3138b49dcf51c0514f129073870701b5cd59cb88df384ac5a4b9ac24a77c601'
GPU = {3:'GPU-d0f25b35-5423-72d4-9f45-047dfe7ed87c',
       6:'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'}
PILOT_INDICES = (48,97,249,345)
RUN_SECONDS = 28800
WAIT_SECONDS = 1800


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, *, exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
    else:
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temporary.replace(path)


def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    path=Path('/proc')/str(pid)
    fields=(path/'stat').read_text().rsplit(')',1)[1].split()
    if fields[0] in ('Z','X'):
        raise ProcessLookupError('Process is no longer active')
    return dict(pid=pid,process_start_ticks=int(fields[19]),
                command=(path/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str((path/'cwd').resolve()))


def active(owner):
    try:
        actual=identity(owner['pid'])
    except (FileNotFoundError,ProcessLookupError):
        return False
    require(actual=={key:owner[key] for key in actual},'PID/start/argv/cwd identity differs; preserve process')
    return True


def signal_owned(owner, sig):
    if not active(owner):
        return False
    fd=os.pidfd_open(owner['pid'])
    try:
        require(active(owner),'Process identity changed after pidfd open')
        signal.pidfd_send_signal(fd,sig)
    finally:
        os.close(fd)
    return True


def argument(command, flag):
    require(command.count(flag)==1 and command.index(flag)+1<len(command),'Missing/repeated argument '+flag)
    return command[command.index(flag)+1]


def validate_expander(owner):
    command=owner['command']
    require(owner['cwd']==str(ROOT) and (str(HERE/'launch_models.py') in command
            or 'astra_rescue_continuation_20261009.launch_models' in command), 'Unexpected expansion launcher identity')
    require(argument(command,'--output')==str(SERVICES) and argument(command,'--slots-per-gpu')=='6',
            'Expansion launcher is not this exact 12-service campaign')
    return owner


def validate_service_owner(path):
    require(path.parent.parent==SERVICES/'services' and path.name=='owner.json', 'Unexpected service owner location')
    owner=read(path); command=owner['command']; directory=path.parent
    names={f'B-gpu{gpu}-r{slot}':gpu for gpu in GPU for slot in range(6)}
    require(directory.name in names and owner.get('gpu')==GPU[names[directory.name]], 'Service directory/GPU differs')
    require(owner['cwd']==str(ROOT) and 'continuous_eval2000_v3.server' in command
            and argument(command,'--mode')=='serve'
            and argument(command,'--server-manifest')==str(directory/'servers.json')
            and argument(command,'--binding')==str(BINDING)
            and argument(command,'--binding-sha256')==BINDING_SHA,
            'Service is not bound to this directory and frozen B2000')
    require(all(key in owner for key in ('pid','process_start_ticks','command','cwd')), 'Incomplete service identity')
    return owner


class Drained(RuntimeError):
    pass


class Campaign:
    def __init__(self):
        self.started=time.monotonic(); self.owner=identity(); self.stage='created'
        self.child=None; self.child_owner=None; self.config=None; self.stop_signal=None
        self.sources={}; self.pilot_jobs=[]; self.expander=None

    def initialize(self):
        self.sources={str(HERE/name):sha(HERE/name) for name in
                      ('continuation_campaign.py','pilot_warmup.py','runner.py','assistance.py','broker.py',
                       'launch_models.py','__init__.py','failure_manifest.json','support.py','keepalive_probe.py')}
        fixed=read(HERE/'failure_manifest.json')
        self.pilot_jobs=[j for j in fixed['jobs'] if j['assignment_index'] in PILOT_INDICES]
        require(len(self.pilot_jobs)==4,'Frozen pilot-development assignment membership changed')
        self.expander=validate_expander(read(EXPAND_OWNER))

    def status(self, stage, **extra):
        self.stage=stage
        value=dict(schema='astra_continuation_campaign_status_v2',stage=stage,owner=self.owner,
                   elapsed_seconds=time.monotonic()-self.started,unix=time.time(),
                   output=str(OUTPUT),services_output=str(SERVICES),expand_owner=self.expander,
                   source_sha256=self.sources,pilot_development_indices=list(PILOT_INDICES),
                   pilot_development_jobs=self.pilot_jobs,original_failure_denominator=1004,
                   planned_episodes=1936,run_seconds=RUN_SECONDS,
                   config_sha256=sha(OUTPUT/'config.json') if (OUTPUT/'config.json').exists() else None,
                   stop_signal=self.stop_signal,**extra)
        write(OUTPUT/'campaign-status.json',value)

    def check_sources(self):
        for path,digest in self.sources.items():
            require(sha(path)==digest,'Pinned campaign source changed: '+path)
        require(sha(BINDING)==BINDING_SHA,'Frozen B2000 binding changed')

    def drained(self):
        return self.stop_signal is not None or (OUTPUT/'drain.request.json').exists()

    def check_drain(self):
        if self.drained():
            raise Drained('Drain requested; no new trial or phase was started')

    def handle_signal(self, signum, _frame):
        self.stop_signal=signum
        path=OUTPUT/'drain.request.json'
        if not path.exists():
            try:write(path,dict(reason='campaign_signal',signal=signum,owner=self.owner,unix=time.time()),exclusive=True)
            except FileExistsError:pass

    def wait_services(self):
        self.status('waiting_for_12_services',wait_budget_seconds=WAIT_SECONDS)
        deadline=time.monotonic()+WAIT_SECONDS; last_status=0
        while time.monotonic()<deadline:
            self.check_drain(); self.check_sources()
            live=active(self.expander)
            ready=None
            try:ready=read(SERVICES/'services-ready.json')
            except (FileNotFoundError,json.JSONDecodeError):pass
            if ready and ready.get('ready') is True and ready.get('models')==12:
                manifests=ready.get('manifests',[])
                expected={str(SERVICES/'services'/f'B-gpu{g}-r{s}'/'servers.json') for g in GPU for s in range(6)}
                require(len(manifests)==12 and set(manifests)==expected,'Full readiness list is not the exact 12 owned slots')
                require(ready.get('binding_sha256')==BINDING_SHA,'Service list frozen binding differs')
                for manifest in manifests:
                    owner=validate_service_owner(Path(manifest).parent/'owner.json')
                    require(active(owner),'A full-campaign service is no longer active')
                self.status('services_ready',models=12,expand_owner_active=live,
                            services_ready_sha256=sha(SERVICES/'services-ready.json'))
                return manifests
            require(live,'Expansion launcher completed/exited before publishing all 12 services')
            if time.monotonic()-last_status>=15:
                self.status('waiting_for_12_services',observed_models=ready.get('models') if ready else None,
                            remaining_wait_seconds=max(0,deadline-time.monotonic()))
                last_status=time.monotonic()
            time.sleep(2)
        raise TimeoutError('12-service readiness was not published within 1800 seconds')

    def phase(self, name, command, env, timeout):
        self.check_drain(); self.check_sources()
        self.status(name,command=command,phase_timeout_seconds=timeout)
        logpath=OUTPUT/(name+'.log')
        with logpath.open('xb') as log:
            self.child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.child_owner=identity(self.child.pid)
            write(OUTPUT/'campaign-processes'/(name+'.json'),self.child_owner,exclusive=True)
            self.status(name,command=command,phase_owner=self.child_owner,log_path=str(logpath))
            deadline=time.monotonic()+timeout; last_status=time.monotonic()
            while self.child.poll() is None:
                if time.monotonic()>=deadline:
                    raise TimeoutError(name+' exceeded bounded wall-clock budget')
                if time.monotonic()-last_status>=30:
                    self.status(name,command=command,phase_owner=self.child_owner,
                                drain_requested=self.drained(),remaining_phase_seconds=max(0,deadline-time.monotonic()))
                    last_status=time.monotonic()
                # Evaluation's own loop observes drain.request.json. Other phases
                # cannot launch episodes; they finish before the next drain gate.
                time.sleep(1)
        code=self.child.returncode; self.child=None; self.child_owner=None
        if code:
            if self.drained():raise Drained(name+' drained with incomplete execution')
            raise RuntimeError(name+' exited with code '+str(code))
        return code

    def wait_broker(self):
        self.status('waiting_for_broker',wait_budget_seconds=600)
        deadline=time.monotonic()+600; last_status=0; path=OUTPUT/'broker-ready.json'
        while time.monotonic()<deadline:
            self.check_drain(); self.check_sources()
            if path.exists():
                ready=read(path)
                require(ready.get('schema')=='astra_broker_ready_v1' and ready.get('model')=='gpt-6-astra'
                        and ready.get('reasoning_effort')=='high' and ready.get('remote_output')==str(OUTPUT)
                        and ready.get('config_sha256')==sha(OUTPUT/'config.json')
                        and ready.get('queue_read_succeeded') is True,
                        'Broker readiness receipt does not bind this output/config/model')
                local_owner=ready.get('local_owner',{})
                require(type(local_owner.get('pid')) is int and local_owner['pid']>0,
                        'Broker readiness receipt has no local PID identity')
                self.status('broker_ready',broker_ready=ready,broker_ready_sha256=sha(path))
                self.check_drain()
                return ready
            if time.monotonic()-last_status>=15:
                self.status('waiting_for_broker',remaining_wait_seconds=max(0,deadline-time.monotonic()))
                last_status=time.monotonic()
            time.sleep(2)
        raise TimeoutError('Broker readiness receipt was not published within 600 seconds; no episodes started')

    def run(self):
        manifests=self.wait_services(); self.check_drain()
        env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'remote_processor_candidate_v1/deps'),
                 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        self.phase('full_warmup',[str(TRAINING_PYTHON),'-u','-m','astra_rescue_continuation_20261009.pilot_warmup',
                   '--services-ready',str(SERVICES/'services-ready.json'),
                   '--output',str(SERVICES/'full-warmup.json')],env,900)
        warm=read(SERVICES/'full-warmup.json')
        require(warm.get('passed') is True and warm.get('models')==12 and warm.get('concurrent_clients')==72,
                'Full 72-client B2000 action/RNG admission failed')
        self.phase('keepalive_probe',[str(TRAINING_PYTHON),'-u','-m',
                   'astra_rescue_continuation_20261009.keepalive_probe',
                   '--services-ready',str(SERVICES/'services-ready.json'),
                   '--output',str(SERVICES/'keepalive-probe.json')],env,900)
        keepalive=read(SERVICES/'keepalive-probe.json')
        require(keepalive.get('passed') is True, 'Same-socket >300-second keepalive validation failed')
        self.phase('prepare',[str(TRAINING_PYTHON),'-u','-m','astra_rescue_continuation_20261009.runner','prepare',
                   '--output',str(OUTPUT),'--failure-manifest',str(HERE/'failure_manifest.json'),
                   '--servers',*manifests,'--arms','control','astra','--response-wait-seconds','1200'],env,300)
        self.config=read(OUTPUT/'config.json'); manifest=read(self.config['manifest'])
        require(len(manifest['jobs'])==1936 and len(manifest['selected_original_jobs'])==968
                and manifest['full_failure_subset'] is False and manifest['arms']==['control','astra'],
                'Continuation paired assignment count differs')
        sys.path.insert(0,str(ROOT))
        from astra_rescue_continuation_20261009 import runner
        cpuenv=runner.environment(self.config)
        self.phase('preflight',[self.config['python'],'-u','-m','astra_rescue_continuation_20261009.runner','preflight',
                   '--config',str(OUTPUT/'config.json')],cpuenv,180)
        self.wait_broker()
        self.check_drain()
        self.phase('evaluating',[self.config['python'],'-u','-m','astra_rescue_continuation_20261009.runner','run',
                   '--config',str(OUTPUT/'config.json'),'--parallel','72','--duration-seconds',str(RUN_SECONDS),
                   '--cleanup-servers'],cpuenv,RUN_SECONDS+300)
        report=read(OUTPUT/'analysis/report.json')
        require(report.get('attempt_coverage_complete') is True and report.get('planned_episodes')==1936
                and report.get('denominators',{}).get('fixed_original_failure_cases')==1004,
                'Continuation lacks all 1936 terminal records or fixed denominator')
        self.status('completed',report_path=str(OUTPUT/'analysis/report.json'),
                    strict_rescued=report.get('strict_rescued'),classification_selected=report.get('classification_selected'),attempt_coverage_complete=report.get('attempt_coverage_complete'))
        return 'completed'

    def cleanup(self):
        """Only exact identities under this service/output pair may be signaled."""
        evidence=[]; targets=[]
        drain=OUTPUT/'drain.request.json'
        if not drain.exists():
            try:write(drain,dict(reason='campaign_cleanup',owner=self.owner,unix=time.time()),exclusive=True)
            except FileExistsError:pass
        def add(owner,kind):
            targets.append(dict(owner=owner,kind=kind))
        if self.child_owner is not None:
            require(self.child_owner['cwd']==str(ROOT),'Phase child cwd changed')
            add(self.child_owner,'campaign_phase')
        # A killed supervisor may leave its exact episode children. Admit only
        # explicit episode commands bound to this independent full config.
        for path in (OUTPUT/'processes').glob('*.json'):
            try:
                owner=read(path); cmd=owner['command']
                require(owner['cwd']==str(ROOT) and str(HERE/'runner.py') in cmd and 'episode' in cmd
                        and argument(cmd,'--config')==str(OUTPUT/'config.json'), 'Unrelated episode owner; preserve')
                add(owner,'campaign_episode')
            except Exception as error:evidence.append(dict(path=str(path),preserved=True,error=repr(error)))
        try:add(validate_expander(read(EXPAND_OWNER)),'expansion_launcher')
        except Exception as error:evidence.append(dict(path=str(EXPAND_OWNER),preserved=True,error=repr(error)))
        self._stop_targets(targets,evidence)
        services=[]
        for path in (SERVICES/'services').glob('*/owner.json'):
            try:
                owner=validate_service_owner(path)
                if not active(owner):
                    evidence.append(dict(owner=owner,kind='service',already_inactive=True));continue
                port=int(argument(owner['command'],'--port'))
                connections=subprocess.run(['ss','-Htn','state','established',f'( sport = :{port} )'],
                                           capture_output=True,text=True,check=True,timeout=10)
                if connections.stdout.strip():
                    evidence.append(dict(owner=owner,kind='service',preserved=True,reason='active_client_connections'));continue
                services.append(dict(owner=owner,kind='service'))
            except Exception as error:evidence.append(dict(path=str(path),preserved=True,error=repr(error)))
        self._stop_targets(services,evidence)
        write(OUTPUT/'campaign-cleanup.json',dict(unix=time.time(),evidence=evidence,
                                               foreign_processes_untouched=True))
        return evidence

    @staticmethod
    def _stop_targets(targets,evidence):
        sent=[]
        for item in targets:
            try:
                item['sigterm_sent']=signal_owned(item['owner'],signal.SIGTERM)
                if item['sigterm_sent']:sent.append(item)
            except Exception as error:item.update(preserved=True,error=repr(error))
            evidence.append(item)
        deadline=time.monotonic()+15
        while sent and time.monotonic()<deadline:
            pending=[]
            for item in sent:
                try:
                    if active(item['owner']):pending.append(item)
                except Exception as error:item.update(preserved=True,error=repr(error))
            sent=pending
            if sent:time.sleep(.25)
        for item in sent:
            try:item['sigkill_sent']=signal_owned(item['owner'],signal.SIGKILL)
            except Exception as error:item.update(preserved=True,error=repr(error))

    def failure_tails(self):
        tails={}
        for path in [OUTPUT/(name+'.log') for name in ('full_warmup','prepare','preflight','evaluating')]+[SERVICES/'launcher.log']:
            if path.is_file():
                with path.open('rb') as stream:
                    stream.seek(0,os.SEEK_END); size=stream.tell(); stream.seek(max(0,size-6000))
                    tails[str(path)]=stream.read(6000).decode('utf-8',errors='replace')
        return tails


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args()
    require(OUTPUT.parent==ROOT/'results' and SERVICES.parent==ROOT/'results','Unexpected fixed campaign paths')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (OUTPUT/'config.json').exists() and not (OUTPUT/'campaign-finished.json').exists(),
                'Campaign already configured/finished; duplicate launch is forbidden')
        require(not (OUTPUT/'campaign-started.json').exists(),'Previous campaign launch exists; preserve it for explicit review')
        require(not (OUTPUT/'campaign-owner.json').exists(),'Previous campaign owner exists; preserve it for explicit review')
        campaign=None; stage='incomplete'; error=None; cleanup_error=None
        try:
            campaign=Campaign()
            write(OUTPUT/'campaign-owner.json',campaign.owner,exclusive=True)
            campaign.initialize()
            write(OUTPUT/'campaign-started.json',dict(owner=campaign.owner,source_sha256=campaign.sources,
                  output=str(OUTPUT),unix=time.time(),pilot_development_jobs=campaign.pilot_jobs),exclusive=True)
            signal.signal(signal.SIGTERM,campaign.handle_signal); signal.signal(signal.SIGINT,campaign.handle_signal)
            stage=campaign.run()
        except BaseException as exc:
            error=traceback.format_exc(); stage='drained' if isinstance(exc,Drained) else 'incomplete'
            if campaign is not None:
                try:
                    campaign.status(stage,error=error,failure_log_tails=campaign.failure_tails())
                    campaign.cleanup()
                except BaseException:cleanup_error=traceback.format_exc()
            write(OUTPUT/'campaign-error.json',dict(error=error,cleanup_error=cleanup_error,unix=time.time()),exclusive=True)
        finally:
            report_path=OUTPUT/'analysis/report.json'
            try:report=read(report_path) if report_path.exists() else {}
            except Exception:report={}
            final=dict(schema='astra_continuation_campaign_finished_v2',stage=stage,complete=report.get('complete',False),attempt_coverage_complete=report.get('attempt_coverage_complete',False),
                       owner=campaign.owner if campaign else None,unix=time.time(),error=error,cleanup_error=cleanup_error,
                       elapsed_seconds=time.monotonic()-campaign.started if campaign else None,
                       source_sha256=campaign.sources if campaign else {},
                       config_sha256=sha(OUTPUT/'config.json') if (OUTPUT/'config.json').exists() else None,
                       original_failure_denominator=1004,planned_episodes=1936,pilot_development_indices=list(PILOT_INDICES),
                       strict_rescued=report.get('strict_rescued'),classification_selected=report.get('classification_selected'),
                       report_path=str(report_path) if report_path.exists() else None,
                       submitted_to_leaderboard=False)
            write(OUTPUT/'campaign-finished.json',final,exclusive=True)
            print(json.dumps(final),flush=True)
        return 0 if stage=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
