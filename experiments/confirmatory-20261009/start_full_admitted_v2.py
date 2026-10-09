"""External, explicitly invoked full-study admission/launch wrapper.

Default action --check is read-only. --launch is the sole mutating entry point:
it revalidates the already approved development evidence, extends the existing
GPU6-only namespace from 1 B + 1 base to 4 B + 2 base, then starts the unchanged
432000-second campaign supervisor. It does not invoke Astra or start a broker.
Use the local Windows entry point so the actual local broker source is hashed
immediately before the SSH request. This file must first be copied unchanged to
the project root; it is deliberately outside the frozen runtime directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NAMESPACE = 'originx_confirmatory_20261009'
DEV = 'originx-confirmatory-development-20261009-v2'
FULL = 'originx-confirmatory-20261009-v1'
SERVICES = 'originx-confirmatory-services-20261009-v2'
BROKER_SHA = '706d85e48a2655e05cb857a2fbf98b1f8a9c1714e81a373a5e60b7eb1211cd99'
ADMIT_SHA = 'a9cc287845d188be23f2bf26cbb94b82d4915be5134669cd5415537b9a4fa170'
GPU_UUID = 'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'
DURATION = 432000


def child_birth(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, parent_pid=int(fields[1]), process_start_ticks=int(fields[19]), state=fields[0])


def child_identity(pid):
    path = Path('/proc') / str(pid); birth = child_birth(pid)
    return dict(pid=pid, process_start_ticks=birth['process_start_ticks'],
                command=(path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str((path / 'cwd').resolve(strict=True)))


def open_pidfd(pid):
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfd support required')
    return os.pidfd_open(pid)


def send_pidfd(fd, sig):
    signal.pidfd_send_signal(fd, sig)


def close_pidfd(fd):
    os.close(fd)


class TrackedChild:
    """Registered immediately after Popen; ownership never needs early argv."""
    def __init__(self, child, command, cwd):
        self.child=child; self.command=list(command); self.cwd=str(cwd); self.pid=child.pid
        self.parent_pid=os.getpid(); self.start=None; self.pidfd=None; self.owner=None
        self.samples=[]; self.signals=[]; self.reaped=False; self.handed_off=False; self.startup_error=None

    def _birth(self):
        value=child_birth(self.pid)
        require(value['pid']==self.pid and value['parent_pid']==self.parent_pid,
                'Spawn is not the direct unreaped child; no signal permitted')
        if self.start is None:self.start=value['process_start_ticks']
        require(value['process_start_ticks']==self.start,'Spawn birth changed; no signal permitted')
        return value

    def _attach(self):
        if self.child.poll() is not None:return False
        if self.pidfd is None:self.pidfd=open_pidfd(self.pid)
        self._birth();return True

    def await_identity(self, timeout=10.0, interval=.02):
        deadline=time.monotonic()+timeout;previous=None;matches=0
        try:
            require(self._attach(),'Child exited before identity binding')
            while time.monotonic()<deadline:
                require(self.child.poll() is None,'Child exited before identity binding')
                self._birth()
                try:
                    observed=child_identity(self.pid);self._birth()
                    valid=(observed['pid']==self.pid and observed['process_start_ticks']==self.start
                           and observed['command']==self.command and observed['cwd']==self.cwd)
                    self.samples.append(dict(observed=observed,expected_identity=valid));self.samples=self.samples[-16:]
                    matches=matches+1 if valid and previous==observed else 1 if valid else 0
                    previous=observed
                    if matches>=2:self.owner=observed;return observed
                except OSError as error:
                    self.samples.append(dict(read_error=str(error)));self.samples=self.samples[-16:]
                    matches=0;previous=None
                time.sleep(interval)
            raise TimeoutError('Bounded child identity handshake expired')
        except BaseException as error:
            self.startup_error=repr(error);raise

    def send(self, sig):
        if not self._attach():return False
        self._birth();send_pidfd(self.pidfd,sig)
        self.signals.append(dict(signal=int(sig),pidfd=True,identity_bound=self.owner is not None));return True

    def drain(self, grace=15, kill_timeout=10):
        try:
            if self.child.poll() is None:self.send(signal.SIGTERM)
            try:code=self.child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.send(signal.SIGKILL);code=self.child.wait(timeout=kill_timeout)
            self.reaped=True;return code
        finally:
            if self.child.returncode is not None:self.close()

    def close(self):
        require(self.child.poll() is not None,'Cannot close live unhanded child')
        self.child.wait(timeout=0);self.reaped=True
        if self.pidfd is not None:close_pidfd(self.pidfd);self.pidfd=None

    def handoff(self, owner):
        require(self.owner is not None and all(owner.get(k)==self.owner[k] for k in self.owner),
                'Handoff requires matching complete published owner')
        require(self._attach() and child_identity(self.pid)==self.owner,'Child changed before owner handoff')
        close_pidfd(self.pidfd);self.pidfd=None;self.handed_off=True

    def receipt(self):
        return dict(schema='originx_external_spawn_handshake_v2',pid=self.pid,parent_pid=self.parent_pid,
                    process_start_ticks=self.start,requested_command=self.command,requested_cwd=self.cwd,
                    owner=self.owner,identity_bound=self.owner is not None,samples=self.samples,signals=self.signals,
                    reaped=self.reaped,handed_off=self.handed_off,returncode=self.child.returncode,startup_error=self.startup_error)


class SubprocessProxy:
    """A per-module proxy; never mutate the shared subprocess module."""
    def __init__(self, original, popen):self.original=original;self.Popen=popen
    def __getattr__(self,name):return getattr(self.original,name)


class OwnedServiceSpawns:
    def __init__(self, services, location, original_profiles):
        self.services=services;self.location=location;self.original=services.subprocess;self.children=[]
        existing={p['service_id'] for p in original_profiles}
        self.expected={tuple(services.service_command(spec)):spec for spec in services.layout(4,2)
                       if spec['service_id'] not in existing}

    def __enter__(self):
        self.services.subprocess=SubprocessProxy(self.original,self.popen);return self

    def __exit__(self,*_):self.services.subprocess=self.original

    def popen(self,command,*args,**kwargs):
        # Unrelated helpers are deliberately delegated unchanged, including
        # nvidia-smi. Only the exact four admitted new service commands qualify.
        spec=self.expected.get(tuple(command)) if isinstance(command,(list,tuple)) else None
        if spec is None:return self.original.Popen(command,*args,**kwargs)
        require(str(kwargs.get('cwd'))==str(ROOT),'Expected service has unexpected cwd')
        child=self.original.Popen(command,*args,**kwargs)
        tracked=TrackedChild(child,command,ROOT);self.children.append((spec,tracked))
        path=self.location/'spawns'/spec['service_id'];path.mkdir(parents=True,exist_ok=False)
        try:tracked.await_identity()
        finally:write_new(path/'handshake.json',tracked.receipt())
        return child

    def drain(self):
        rows=[]
        for spec,child in reversed(self.children):
            if not child.handed_off:
                try:child.drain()
                except BaseException as error:rows.append(dict(service_id=spec['service_id'],error=repr(error),receipt=child.receipt()));continue
            rows.append(dict(service_id=spec['service_id'],receipt=child.receipt()))
        safe=all(t.reaped or t.handed_off for _,t in self.children)
        write_new(self.location/'spawn-drain.json',dict(children=rows,all_pending_spawns_reaped=safe,
                  handed_off_services_require_frozen_owner_cleanup=True))
        require(safe,'Some owned service spawns not drained; preserve exact handles')

    def handoff(self,profiles):
        owners={p['service_id']:p['owner'] for p in profiles}
        for spec,child in self.children:child.handoff(owners[spec['service_id']])
        write_new(self.location/'service-owner-handoff.json',dict(children=[t.receipt() for _,t in self.children],
            frozen_service_source_unmodified=True,launcher_source_sha256=sha(__file__)))


def prebroker_campaign_commands():
    output=ROOT/'results'/FULL;ready=ROOT/'results'/SERVICES/'services-ready.json'
    return [
        [str(ROOT/'envs/training/bin/python'),'-u','-m',NAMESPACE+'.probe','--services',str(ready),
         '--output',str(ready.parent/'probes/confirmatory-v1')],
        [str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/NAMESPACE/'runner.py'),'prepare','--services-ready',str(ready),
         '--output',str(output),'--duration-seconds',str(DURATION)],
        [str(ROOT/'envs/sim/bin/python'),'-u',str(ROOT/NAMESPACE/'runner.py'),'preflight','--config',str(output/'config.json')],
    ]


def require_no_live_prebroker_children():
    """Read-only exact-command scan also catches a reparented startup child.

    Matching orphan processes are never signalled without a parent handle. Their
    presence instead blocks model cleanup and leaves explicit recovery evidence.
    """
    matches=[];expected=prebroker_campaign_commands()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            command=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if command not in expected:continue
            owner=child_identity(int(proc.name));birth=child_birth(int(proc.name))
            if owner['cwd']==str(ROOT) and birth['state'] not in ('Z','X'):matches.append(owner)
        except (OSError,ValueError):continue
    require(not matches,'Exact pre-broker children remain live or orphaned; preserve models: '+repr(matches))


def await_campaign_owner(child,path,timeout=15):
    deadline=time.monotonic()+timeout
    while True:
        require(child.poll() is None,'Campaign exited before recording its own owner')
        if path.exists():
            try:return read(path)
            except (json.JSONDecodeError,OSError):pass
        require(time.monotonic()<deadline,'Campaign owner publication timed out; preserve launcher identity')
        time.sleep(.05)


def drain_campaign(child,location):
    """Before broker launch, drain only this owned supervisor and exact children.

    Stop its spawning thread briefly while binding any just-spawned CPU probe or
    preparation child. PID handles and direct ancestry identify them; no UID,
    process-name wildcard, model process or already-scored rollout is targeted.
    """
    checks=[];paused=False
    try:
        if child.child.poll() is None:
            require(not (ROOT/'results'/FULL/'broker-ready.json').exists(),
                    'Broker already admitted; preserve running experiment for explicit recovery')
            child.send(signal.SIGSTOP);paused=True
            deadline=time.monotonic()+5
            while True:
                if child.child.poll() is not None:break
                stopped=child._birth()
                if stopped['state'] in ('T','t'):break
                require(time.monotonic()<deadline,'Supervisor did not enter stopped state; do not enumerate a spawning parent')
                time.sleep(.01)
            enumerate_children=child.child.poll() is None
            if enumerate_children:require(child._birth()['state'] in ('T','t'),'Supervisor resumed before descendant snapshot')
            for proc in (Path('/proc').iterdir() if enumerate_children else ()):
                if not proc.name.isdigit():continue
                try:birth=child_birth(int(proc.name))
                except (OSError,ValueError):continue
                if birth['parent_pid']!=child.pid or birth['state'] in ('Z','X'):continue
                fd=open_pidfd(birth['pid'])
                try:
                    deadline=time.monotonic()+3;owner=None
                    while time.monotonic()<deadline:
                        now=child_birth(birth['pid'])
                        require(now['process_start_ticks']==birth['process_start_ticks'] and now['parent_pid']==child.pid,
                                'Supervisor descendant birth changed')
                        if now['state'] in ('Z','X'):break
                        try:observed=child_identity(birth['pid'])
                        except OSError:time.sleep(.02);continue
                        if observed['command'] in prebroker_campaign_commands() and observed['cwd']==str(ROOT):
                            owner=observed;break
                        time.sleep(.02)
                    if owner is None:
                        try:now=child_birth(birth['pid'])
                        except OSError:now={'state':'X'}
                        require(now['state'] in ('Z','X'),'Unrecognized supervisor descendant; no signal issued')
                        checks.append(dict(birth=birth,already_inactive=True));continue
                    send_pidfd(fd,signal.SIGTERM);deadline=time.monotonic()+15
                    while time.monotonic()<deadline:
                        try:now=child_birth(birth['pid'])
                        except OSError:break
                        if now['process_start_ticks']!=birth['process_start_ticks'] or now['state'] in ('Z','X'):break
                        time.sleep(.05)
                    else:
                        send_pidfd(fd,signal.SIGKILL);deadline=time.monotonic()+5
                        while time.monotonic()<deadline:
                            try:now=child_birth(birth['pid'])
                            except OSError:break
                            if now['process_start_ticks']!=birth['process_start_ticks'] or now['state'] in ('Z','X'):break
                            time.sleep(.05)
                        else:raise RuntimeError('Owned supervisor descendant did not terminate')
                    checks.append(dict(owner=owner,direct_parent_pid=child.pid,pidfd=True,inactive=True))
                finally:close_pidfd(fd)
            child.send(signal.SIGCONT);paused=False
        child.drain()
        require_no_live_prebroker_children()
        write_new(location/'campaign-drain.json',dict(campaign=child.receipt(),descendants=checks,
                  no_broker_started_by_launcher=True,original_campaign_records_preserved=True))
    except BaseException:
        if paused:
            try:child.send(signal.SIGCONT)
            except BaseException:pass
        write_new(location/'campaign-drain-error.json',dict(error=traceback.format_exc(),campaign=child.receipt(),descendants=checks))
        raise


def cleanup_known_services(services,spawned,location,original_profiles):
    """Failure-only cleanup tolerates preserved, already-reaped partial spawns.

    The frozen cleanup refuses partial directories. Never manufacture an owner
    to bypass it: pending children were separately reaped through their handles;
    only exact admitted or this-launch published owners can receive a signal.
    """
    specs={s['service_id']:s for s in services.layout(4,2)}
    originals={p['service_id']:p['owner'] for p in original_profiles}
    new={s['service_id']:t for s,t in spawned.children}
    records=[];handles=[];inactive=[];partials=[]
    def stopped(owner):
        try:now=child_birth(owner['pid'])
        except (OSError,ProcessLookupError):return True
        return now['process_start_ticks']!=owner['process_start_ticks'] or now['state'] in ('Z','X')
    try:
        # The original mutex prevents the namespace lease racing this cleanup.
        with services.service_lock(services.SERVICE_OUTPUT):
            for directory in (services.SERVICE_OUTPUT/'services').iterdir():
                require(directory.is_dir() and not directory.is_symlink() and directory.name in specs,
                        'Unexpected service directory; preserve foreign artifacts')
                name=directory.name;spec=specs[name];owner_path=directory/'owner.json'
                if not owner_path.exists():
                    require(name in new and new[name].reaped,'Untracked partial service may still be alive')
                    partials.append(dict(service_id=name,spawn=new[name].receipt(),directory_preserved=True));continue
                owner=read(owner_path)
                if name in originals:require(owner==originals[name],'Original admitted owner changed')
                else:
                    require(name in new and new[name].owner is not None and
                            all(owner.get(k)==v for k,v in new[name].owner.items()),'New owner is not this launcher child')
                records.append((spec,owner))
                if stopped(owner):inactive.append(owner['pid']);continue
                fd=open_pidfd(owner['pid']);handles.append((spec,owner,fd))
                services.assert_cleanup_safe(owner,spec,services.process_identity(owner['pid']),
                                             services.tcp_connections(owner['pid'],spec['port']))
            require(set(originals)<={s['service_id'] for s,_ in records},'Admitted original owner record is missing')
            draining=services.SERVICE_OUTPUT/'draining.json'
            if not draining.exists():write_new(draining,dict(namespace=NAMESPACE,requested_unix=time.time(),
                  reason='External failed-launch cleanup; all own startup descendants drained',launcher_source_sha256=sha(__file__)))
            time.sleep(2)
            for spec,owner,fd in handles:
                if not stopped(owner):services.assert_cleanup_safe(owner,spec,services.process_identity(owner['pid']),
                                                                   services.tcp_connections(owner['pid'],spec['port']))
            signalled=[]
            for spec,owner,fd in handles:
                if stopped(owner):continue
                services.assert_cleanup_safe(owner,spec,services.process_identity(owner['pid']),
                                             services.tcp_connections(owner['pid'],spec['port']))
                send_pidfd(fd,signal.SIGTERM);signalled.append(owner['pid'])
            deadline=time.monotonic()+30
            while any(not stopped(o) for _,o in records) and time.monotonic()<deadline:time.sleep(.1)
            remaining=[o for _,o in records if not stopped(o)]
            receipt=dict(schema='originx_failed_full_launch_cleanup_v2',all_owned_services_stopped=not remaining,
                         remaining_owned=remaining,signalled_owned_pids=signalled,previously_inactive_pids=inactive,
                         partial_spawn_directories=partials,original_profile_ids=sorted(originals),pidfd=True,
                         frozen_service_source_unmodified=True,launcher_source_sha256=sha(__file__),
                         forced_model_kill=False,foreign_processes_untouched=True)
            write_new(location/'cleanup.json',receipt)
            require(not remaining,'Some exact owned models did not stop; no forced model kill')
            return receipt
    finally:
        for _,_,fd in handles:close_pidfd(fd)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or symbolic-link evidence: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_no_full(root):
    output = Path(root) / 'results' / FULL
    require(not output.is_symlink(), 'Full output must not be symbolic link')
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            'Full output already contains an owner, configuration or other work; no restart')


def validate_extension(previous, current):
    for ready in (previous, current):
        require(ready.get('schema') == 'originx_confirmatory_services_v1'
                and ready.get('namespace') == NAMESPACE and ready.get('ready') is True
                and ready.get('gpu_index') == 6 and ready.get('gpu_uuid') == GPU_UUID,
                'Service namespace or GPU differs')
        require(ready.get('profiles') == ready.get('models'), 'Service profile inventories differ')
        require(ready.get('model_count') == len(ready['models']) and ready.get('clients') == 6 * len(ready['models']),
                'Service capacity/count differs')
        for model in ready['models']:
            require(model.get('gpu_index') == 6 and model.get('gpu_uuid') == GPU_UUID
                    and model.get('slots') == 6 and model.get('namespace') == NAMESPACE,
                    'A profile escaped GPU6-only capacity')
    old = {x['service_id']: x for x in previous['models']}
    new = {x['service_id']: x for x in current['models']}
    require(len(old) == len(previous['models']) == 2 and set(old) == {'B-gpu6-r0', 'base-gpu6-r0'},
            'Development must retain its exact original two profiles')
    require(len(new) == len(current['models']) == 6 and set(new) ==
            {f'B-gpu6-r{i}' for i in range(4)} | {f'base-gpu6-r{i}' for i in range(2)},
            'Expansion must be exactly 4 B + 2 base')
    require(all(new[name] == profile for name, profile in old.items()), 'Expansion replaced an original model/profile')
    require(Counter(m['policy_id'] for m in new.values()) == {'B': 4, 'base': 2}, 'Wrong policy allocation')
    require(len({m['port'] for m in new.values()}) == 6, 'Duplicated service port')


def validate_bindings(approval, root, *, previous_ready=None, current_ready=None):
    root = Path(root).resolve()
    ready_path = root / 'results' / SERVICES / 'services-ready.json'
    require(approval.get('schema') == 'originx_confirmatory_admission_review_v1'
            and approval.get('passed') is True and approval.get('no_confirmatory_outcomes_seen') is True
            and approval.get('raw_verified_outcomes') == 14 and approval.get('complete_cli_usage') is True,
            'Missing strict full development approval')
    require(approval.get('admission_tool_sha256') == ADMIT_SHA
            and approval.get('broker_source_sha256') == BROKER_SHA, 'Approved tool/broker version differs')
    require(sha(root / 'admit_full.py') == ADMIT_SHA, 'Installed admission tool changed')
    require(bool(approval.get('bindings_sha256')), 'Approval has no evidence bindings')
    for name, expected in approval['bindings_sha256'].items():
        path = Path(name)
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), 'Bound evidence escapes project')
        if current_ready is not None and path == ready_path:
            require(previous_ready is not None, 'Unreviewed service-ready mutation')
            validate_extension(previous_ready, current_ready)
            continue
        require(sha(path) == expected, 'Approved immutable evidence changed: ' + str(path))
    config = read(root / 'results' / DEV / 'config.json')
    require(sha(root / 'results' / DEV / 'config.json') == approval['config_sha256']
            and config['manifest_sha256'] == approval['manifest_sha256']
            and config['source_freeze_sha256'] == approval['source_freeze_sha256'], 'Approved configuration pins differ')
    require(config['broker_source_sha256'] == BROKER_SHA, 'Configuration expects another broker')
    return config


def precheck(root, broker_check, self_sha, *, owner_active, owner_stopped):
    root = Path(root).resolve(); output = root / 'results' / DEV
    require(broker_check.get('broker_source_sha256') == BROKER_SHA
            and broker_check.get('launcher_source_sha256') == self_sha
            and broker_check.get('performed_on_local_host') is True, 'Local broker-source launch check missing')
    ensure_no_full(root)
    require(not any((root / 'results' / SERVICES / name).exists() for name in
                    ('full-launch-admission-v1','full-launch-admission-v2')),
            'Prior external full-launch attempt exists; preserve it and review instead of restarting')
    approval_path = output / 'admission-review.json'; approval = read(approval_path)
    config = validate_bindings(approval, root)
    ready_path = root / 'results' / SERVICES / 'services-ready.json'; ready = read(ready_path)
    require(len(ready.get('models', [])) == 2 and ready['models'] == config['models'],
            'Current original services differ from admitted development profiles')
    require(not (ready_path.parent / 'draining.json').exists()
            and not (ready_path.parent / 'lease-expired.json').exists(), 'Services are draining/lease expired')
    lease = read(ready_path.parent / 'lease.owner.json')
    require(owner_active(lease), 'Existing bounded service lease is not alive')
    require(all(owner_active(model['owner']) for model in ready['models']), 'Original development model owner no longer alive')
    for check in approval['owned_development_processes']:
        require(owner_stopped(read(check['path'])), 'Development process revived or still live')
    return dict(schema='originx_confirmatory_full_start_precheck_v1', passed=True,
                approval_sha256=sha(approval_path), approval=approval,
                original_services_ready_sha256=sha(ready_path), original_services=ready,
                local_broker_source_check=broker_check, launcher_source_sha256=self_sha,
                duration_seconds=DURATION, requested_allocation={'B': 4, 'base': 2},
                gpu_index=6, gpu_uuid=GPU_UUID, no_rollout_started=True)


def remote(action, broker_check, expected_self_sha):
    require(os.name == 'posix' and Path(__file__).resolve() == ROOT / 'start_full_admitted_v2.py',
            'Remote execution is restricted to the independent project-root launcher')
    require(sha(__file__) == expected_self_sha, 'Remote launcher differs from local reviewed source')
    require(sha(ROOT / 'admit_full.py') == ADMIT_SHA, 'Installed admission tool changed')
    # Verify pinned executable bytes before importing any frozen runtime module.
    validate_bindings(read(ROOT / 'results' / DEV / 'admission-review.json'), ROOT)
    sys.path.insert(0, str(ROOT))
    admission = load(ROOT / 'admit_full.py', 'frozen_admission_for_launch')
    # The frozen services module does not load a model at import.
    from originx_confirmatory_20261009 import services
    require(services.SERVICE_OUTPUT == ROOT / 'results' / SERVICES and services.GPU_INDEX == 6
            and services.GPU_UUID == GPU_UUID, 'Frozen service namespace/GPU differs')
    checked = precheck(ROOT, broker_check, expected_self_sha, owner_active=services.active,
                       owner_stopped=admission.linux_owner_stopped)
    if action == 'check':
        return {k: v for k, v in checked.items() if k not in ('approval', 'original_services')}
    require(action == 'launch', 'Unknown remote action')
    import fcntl
    location = ROOT / 'results' / SERVICES / 'full-launch-admission-v2'
    # A prior attempt, including a failed one, requires explicit review instead
    # of silently expanding/relaunching the same scored experiment.
    location.mkdir(exist_ok=False)
    lock = (location / 'lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_new(location / 'launcher.owner.json', services.process_identity(os.getpid()))
    write_new(location / 'pre-expansion.json', checked)
    campaign_owner = None; campaign_spawn = None; expanded = False
    spawned = OwnedServiceSpawns(services,location,checked['original_services']['models'])
    try:
        # Frozen service launch owns its own mutex, validates all model/PID/GPU
        # identities and refuses foreign GPU users before each new allocation.
        expanded = True
        with spawned:
            current = services.launch(4, 2)
        validate_extension(checked['original_services'], current)
        require(all(services.active(m['owner']) for m in current['models']), 'Expanded services not all live')
        validate_bindings(checked['approval'], ROOT, previous_ready=checked['original_services'], current_ready=current)
        require(sha(ROOT / 'results' / DEV / 'admission-review.json') == checked['approval_sha256'], 'Approval changed during expansion')
        require(sha(__file__) == expected_self_sha, 'Launcher changed during expansion')
        ensure_no_full(ROOT)
        extension = dict(schema='originx_confirmatory_service_extension_v1', passed=True,
                         old_services_ready_sha256=checked['original_services_ready_sha256'],
                         new_services_ready_sha256=sha(services.SERVICE_OUTPUT / 'services-ready.json'),
                         original_profiles_exactly_preserved=True, models=current['models'],
                         documented_mutable_binding=str(services.SERVICE_OUTPUT / 'services-ready.json'),
                         all_other_approved_bindings_unchanged=True)
        write_new(location / 'service-extension.json', extension)
        command = [str(ROOT / 'envs/sim/bin/python'), '-B', '-u', '-m', NAMESPACE + '.campaign',
                   '--duration-seconds', str(DURATION)]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                           PYTHONPATH=str(ROOT) + ':' + str(ROOT / 'remote_processor_candidate_v1/deps'),
                           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (location / 'campaign.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            campaign_spawn = TrackedChild(child,command,ROOT)
        try:campaign_owner = campaign_spawn.await_identity()
        finally:write_new(location/'campaign-handshake.json',campaign_spawn.receipt())
        write_new(location / 'campaign-launch.owner.json', campaign_owner)
        owner_path = ROOT / 'results' / FULL / 'campaign.owner.json'
        owner = await_campaign_owner(child,owner_path)
        require(all(owner.get(k) == campaign_owner[k] for k in ('pid', 'process_start_ticks', 'command', 'cwd'))
                and services.active(owner), 'Campaign owner does not match launched process')
        result = dict(schema='originx_confirmatory_full_start_v1', started=True, campaign_owner=owner,
                      duration_seconds=DURATION, approval_sha256=checked['approval_sha256'],
                      extension_sha256=sha(location / 'service-extension.json'), launcher_source_sha256=expected_self_sha,
                      broker_source_sha256=BROKER_SHA, broker_started_by_tool=False,
                      model_count=6, gpu_index=6, gpu_uuid=GPU_UUID,
                      next_phase='Campaign performs full socket probe/preflight and waits for separately started fixed broker')
        spawned.handoff(current['models'])
        campaign_spawn.handoff(owner)
        result.update(external_spawn_handshakes=True,frozen_service_source_unmodified=True,
                      new_service_children=len(spawned.children))
        write_new(location / 'started.json', result)
        return result
    except BaseException:
        write_new(location / 'error.json', dict(error=traceback.format_exc(), expansion_attempted=expanded,
                   campaign_owner=campaign_owner, no_automatic_restart=True,
                   campaign_spawn=campaign_spawn.receipt() if campaign_spawn else None))
        campaign_drained = campaign_spawn is None
        if campaign_spawn is not None:
            try:drain_campaign(campaign_spawn,location);campaign_drained=True
            except BaseException:campaign_drained=False
        if expanded and campaign_drained:
            try:
                spawned.drain()
                cleanup_known_services(services,spawned,location,checked['original_services']['models'])
            except BaseException:
                write_new(location / 'cleanup-error.json', dict(error=traceback.format_exc(),
                          existing_namespace_lease_retained=True))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--launch', action='store_true')
    action.add_argument('--remote-check', action='store_true', help=argparse.SUPPRESS)
    action.add_argument('--remote-launch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--broker-source', type=Path, default=Path(__file__).with_name('broker.py'))
    parser.add_argument('--expected-self-sha', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.remote_check or args.remote_launch:
            result = remote('launch' if args.remote_launch else 'check', json.load(sys.stdin), args.expected_self_sha)
            print(json.dumps(result, indent=2)); return 0
        require(os.name == 'nt', 'Use the local Windows entry point for a fresh actual broker-source check')
        digest = sha(__file__)
        require(sha(args.broker_source) == BROKER_SHA, 'Actual local frozen broker source differs; launch refused')
        receipt = dict(performed_on_local_host=True, broker_source_sha256=BROKER_SHA,
                       local_broker_path=str(args.broker_source.resolve()), launcher_source_sha256=digest,
                       checked_at=datetime.now(timezone.utc).isoformat(), no_cli_invoked=True)
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'hkust-cluster',
                   (ROOT / 'envs/sim/bin/python').as_posix(), '-B', (ROOT / 'start_full_admitted_v2.py').as_posix(),
                   '--remote-launch' if args.launch else '--remote-check', '--expected-self-sha', digest]
        completed = subprocess.run(command, input=json.dumps(receipt).encode('utf-8'), timeout=4000 if args.launch else 120)
        return completed.returncode
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, error=str(error))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
