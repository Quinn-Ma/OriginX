"""Start one hidden Windows broker after the fixed full campaign passes preflight.

No GPU management, quota reset, marker removal, automatic broker restart, or model
invocation occurs in this helper. --self-test uses local fixtures and fake owners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import broker

HERE = Path(__file__).resolve().parent
PROJECT = "/ephemeral/qinzhen/robocasa-xr1-20261003"
REMOTE_OUTPUT = PROJECT + "/results/astra-native-reset-full-20261009-v1"
LOCAL_OUTPUT = HERE / "broker_state_full_v1"
CODEX_EXE = Path(r"C:\Users\LuckyQinzhen\AppData\Local\OpenAI\Codex\bin\9691020b546a15b2\codex.exe")
BROKER_SHA = "452bc7c53c63155ee250987312bccaf2b4f3d3d831e224aa8749d0cd0eac6ca4"
PINNED_SOURCES = {
    PROJECT + "/astra_rescue_20261009/runner.py": "1eaa370fe16d57a2b93eab7216cf011ea1b135c53fb2f367d822afc1bc0c9e4f",
    PROJECT + "/astra_rescue_20261009/assistance.py": "82fcada288b85d20e23d4bf9e2256f0a6cd30e2b550c0c3190cf29741ab01162",
    PROJECT + "/astra_rescue_20261009/__init__.py": "0fc1bd5b367d9f73651802bf9854f89320c4f8341560193a8054ec7e6cba9f91",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_exclusive(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    broker.write_json(temporary, value)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


# Read-only until REMOTE_PUBLISH_READY is selected. All paths are checked before
# reading source files; no model inputs or authentication files are inspected.
REMOTE_BASE = r'''
import hashlib, json, os, pathlib, re, sys
PROJECT = '/ephemeral/qinzhen/robocasa-xr1-20261003'
EXPECTED_OUTPUT = PROJECT + '/results/astra-native-reset-full-20261009-v1'
root = pathlib.Path(sys.argv[1])
pinned = json.loads(sys.argv[2])
def require(condition, message):
    if not condition: raise ValueError(message)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    require(not path.is_symlink() and path.is_file(), 'Missing or symbolic-link artifact: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))
def checked_project_file(name):
    path = pathlib.Path(name)
    require(path.is_absolute() and not path.is_symlink() and path.resolve(strict=True).is_relative_to(project),
            'Source/artifact escaped approved project')
    require(path.is_file(), 'Not a regular file')
    return path
require(str(root) == EXPECTED_OUTPUT, 'Only the exact full campaign output is authorized')
project = pathlib.Path(PROJECT).resolve(strict=True)
require(not root.is_symlink() and root.resolve().is_relative_to(project), 'Output escaped approved project')
def inspect():
    if not root.exists(): return {'state': 'waiting_output'}
    if (root/'campaign-finished.json').exists():
        return {'state': 'campaign_finished', 'finished': read(root/'campaign-finished.json')}
    if (root/'drain.request.json').exists(): return {'state': 'drained'}
    if not (root/'config.json').exists(): return {'state': 'waiting_config'}
    config = read(root/'config.json')
    require(config.get('schema') == 'astra_native_reset_control_rescue_config_v1'
            and config.get('output') == str(root), 'Config schema/output mismatch')
    if not (root/'preflight.json').exists(): return {'state': 'waiting_preflight'}
    preflight = read(root/'preflight.json')
    require(preflight.get('passed') is True and preflight.get('episodes') == 2008
            and preflight.get('original_cases') == 1004, 'Full preflight did not pass')
    require(len(config.get('models', [])) == 12, 'Full campaign requires exactly 12 prepared services')
    sources = config.get('source_sha256')
    require(isinstance(sources, dict) and bool(sources), 'Missing frozen source map')
    require(all(sources.get(name) == expected for name, expected in pinned.items()), 'Pinned rescue source differs')
    for name, expected in sources.items():
        require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Invalid source digest')
        require(sha(checked_project_file(name)) == expected, 'Configured source hash mismatch: ' + name)
    require(config.get('manifest') == str(root/'manifest.json'), 'Manifest is not bound to this output')
    manifest_path = checked_project_file(config['manifest'])
    require(sha(manifest_path) == config.get('manifest_sha256'), 'Manifest digest mismatch')
    manifest = read(manifest_path)
    require(manifest.get('schema') == 'astra_native_reset_control_rescue_v1'
            and manifest.get('full_failure_subset') is True and manifest.get('arms') == ['control', 'astra']
            and len(manifest.get('jobs', [])) == 2008
            and len(manifest.get('selected_original_jobs', [])) == 1004, 'Not the full fixed paired campaign')
    failure_path = checked_project_file(config['failure_manifest'])
    require(sha(failure_path) == config.get('failure_manifest_sha256'), 'Fixed failure-manifest digest mismatch')
    ready = read(root/'broker-ready.json') if (root/'broker-ready.json').exists() else None
    return {'state': 'ready', 'config_sha256': sha(root/'config.json'),
            'preflight_sha256': sha(root/'preflight.json'), 'verified_sources': len(sources),
            'models': 12, 'episodes': 2008, 'existing_ready': ready}
'''

REMOTE_INSPECT = REMOTE_BASE + r'''
try:
    print(json.dumps(inspect()))
except Exception as error:
    print(json.dumps({'state': 'invalid', 'error': str(error)}))
'''

REMOTE_PUBLISH_READY = REMOTE_BASE + r'''
snapshot = inspect()
require(snapshot['state'] == 'ready', 'Campaign no longer accepts broker readiness')
payload = json.loads(sys.argv[3])
require(payload.get('schema') == 'astra_broker_ready_v1' and payload.get('model') == 'gpt-6-astra'
        and payload.get('reasoning_effort') == 'high' and payload.get('remote_output') == str(root)
        and payload.get('config_sha256') == snapshot['config_sha256']
        and payload.get('queue_read_succeeded') is True, 'Ready schema/config binding mismatch')
owner = payload.get('local_owner', {})
require(type(owner.get('pid')) is int and owner['pid'] > 0 and bool(owner.get('start_utc')),
        'Missing local broker process identity')
final = root/'broker-ready.json'
if final.exists():
    require(read(final) == payload, 'Existing broker-ready marker differs; refusing overwrite')
    result = 'already_present'
else:
    temporary = root/('.broker-ready-' + os.urandom(16).hex() + '.tmp')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
        stream.flush(); os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, final)
            result = 'created'
        except FileExistsError:
            require(read(final) == payload, 'Concurrent broker-ready marker differs')
            result = 'already_present'
    finally:
        temporary.unlink()
print(json.dumps({'result': result, 'broker_ready_sha256': sha(final), 'config_sha256': snapshot['config_sha256']}))
'''


class Remote:
    def __init__(self):
        broker.safe_remote(REMOTE_OUTPUT)
        self.queue = broker.SSHQueue(REMOTE_OUTPUT)

    def request(self, script, *args, timeout=55):
        arguments = (REMOTE_OUTPUT, json.dumps(PINNED_SOURCES), *args)
        command = "python3 - " + " ".join(shlex.quote(value) for value in arguments)
        result = subprocess.run(["ssh", *broker.SSH_OPTIONS, broker.HOST, command],
                                input=script.encode("utf-8"), capture_output=True, timeout=timeout, check=False)
        require(result.returncode == 0, "SSH failure: " + result.stderr.decode("utf-8", "replace")[-4000:])
        return json.loads(result.stdout)

    def inspect(self, timeout):
        return self.request(REMOTE_INSPECT, timeout=timeout)

    def pending(self):
        return self.queue.pending()

    def publish(self, payload):
        return self.request(REMOTE_PUBLISH_READY, json.dumps(payload, ensure_ascii=False), timeout=60)


class WindowsProcesses:
    def identity(self, pid):
        require(type(pid) is int and pid > 0, "Invalid process ID")
        script = ("$p = Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "'; "
                  "if ($null -eq $p) { Write-Output 'null'; exit 0 }; "
                  "[pscustomobject]@{pid=[int]$p.ProcessId; parent_pid=[int]$p.ParentProcessId; "
                  "start_utc=$p.CreationDate.ToUniversalTime().ToString('o'); "
                  "command_line=$p.CommandLine; executable_path=$p.ExecutablePath} | ConvertTo-Json -Compress")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=20, check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        require(result.returncode == 0, "Could not inspect owned Windows process: " + result.stderr[-2000:])
        return json.loads(result.stdout)

    def launch(self, command, local):
        require(CODEX_EXE.is_file(), "Pinned Codex executable is missing")
        env = dict(os.environ)
        env["PATH"] = str(CODEX_EXE.parent) + os.pathsep + env.get("PATH", "")
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        with (local/"starter-child.stdout.log").open("xb") as stdout, (local/"starter-child.stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, cwd=str(HERE), env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, startupinfo=startup,
                                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                                     close_fds=True)
        return child.pid


class Starter:
    def __init__(self, local, remote, processes, clock=time):
        self.local, self.remote, self.processes, self.clock = Path(local), remote, processes, clock
        self.local.mkdir(parents=True, exist_ok=True)
        self.owner_path = self.local/"starter-owner.json"
        self.payload_path = self.local/"starter-ready-payload.json"
        self.helper_started_at = utc()

    def status(self, state, **extra):
        record = dict(schema="astra_full_broker_starter_status_v1", state=state, updated_at=utc(),
                      helper_pid=os.getpid(), helper_started_at_utc=self.helper_started_at,
                      remote_output=REMOTE_OUTPUT, local_output=str(self.local), **extra)
        broker.write_json(self.local/"starter-status.json", record)
        with (self.local/"starter-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def no_stop(self):
        require(not (self.local/"stop.json").exists(), "Broker stop.json exists; inspect it manually before any resume")

    def inspect_owner(self, owner):
        identity = owner.get("local_owner")
        require(isinstance(identity, dict) and type(identity.get("pid")) is int,
                "Prior launch is incomplete/uncertain; automatic restart is forbidden")
        actual = self.processes.identity(identity["pid"])
        require(actual is not None and all(actual.get(k) == identity.get(k)
                for k in ("pid", "start_utc", "command_line", "executable_path")),
                "Recorded broker exited or process identity changed; automatic restart is forbidden")
        require(actual["command_line"] == subprocess.list2cmdline(owner["command"]), "Broker command line differs")
        return actual

    def start_or_reuse(self, snapshot):
        self.no_stop()
        if self.owner_path.exists():
            owner = broker.read_json(self.owner_path)
            require(owner.get("remote_output") == REMOTE_OUTPUT and owner.get("config_sha256") == snapshot["config_sha256"]
                    and owner.get("broker_source_sha256") == BROKER_SHA, "Existing broker owner belongs to another run")
            self.inspect_owner(owner)
            self.status("reusing_existing_broker", owner=owner)
            return owner
        require(snapshot.get("existing_ready") is None, "Remote broker-ready exists without a matching local owner")
        # A broker launched outside this helper must not be duplicated either.
        with broker.ProcessLock(self.local/"broker.lock"):
            pass
        command = [sys.executable, "-u", str(HERE/"broker.py"), "--remote-output", REMOTE_OUTPUT,
                   "--local-output", str(self.local.resolve()), "--run-seconds", "30000",
                   "--max-batch", "8", "--batch-wait-seconds", "60", "--workers", "1"]
        owner = dict(schema="astra_full_broker_owner_v1", stage="launch_intent", created_at_utc=utc(),
                     remote_output=REMOTE_OUTPUT, config_sha256=snapshot["config_sha256"],
                     broker_source_sha256=BROKER_SHA, helper_source_sha256=sha(__file__),
                     command=command, codex_executable=str(CODEX_EXE), local_owner=None)
        write_exclusive(self.owner_path, owner)
        pid = self.processes.launch(command, self.local)
        owner.update(stage="spawned", spawned_pid=pid, spawned_at_utc=utc())
        broker.write_json(self.owner_path, owner)
        identity = self.processes.identity(pid)
        require(identity is not None and identity.get("command_line") == subprocess.list2cmdline(command)
                and bool(identity.get("start_utc")), "New broker process identity could not be confirmed")
        owner.update(stage="running", local_owner=identity)
        broker.write_json(self.owner_path, owner)
        self.status("broker_started", owner=owner)
        return owner

    def wait_broker(self, owner):
        deadline = self.clock.monotonic() + 60
        while self.clock.monotonic() < deadline:
            self.no_stop()
            self.inspect_owner(owner)
            path = self.local/"broker_status.json"
            if path.exists():
                value = broker.read_json(path)
                require(value.get("remote_output") == REMOTE_OUTPUT and value.get("model") == "gpt-6-astra"
                        and value.get("reasoning_effort") == "high", "Broker status identity differs")
                require(value.get("state") not in ("stopped", "finished"), "Broker stopped before readiness")
                if value.get("state") == "waiting":
                    return
            self.clock.sleep(2)
        raise RuntimeError("Broker did not reach waiting state within 60 seconds; preserved owner forbids automatic restart")

    def run(self, wait_seconds=1800):
        require(sha(HERE/"broker.py") == BROKER_SHA, "Frozen broker source changed")
        deadline = self.clock.monotonic() + wait_seconds
        self.status("waiting_for_full_preflight")
        while self.clock.monotonic() < deadline:
            try:
                snapshot = self.remote.inspect(timeout=max(1, min(55, deadline-self.clock.monotonic())))
            except Exception as error:
                self.status("waiting_transport", error=repr(error))
                snapshot = None
            if snapshot is not None:
                self.status("remote_checked", snapshot=snapshot)
                if snapshot["state"] == "campaign_finished":
                    self.status("campaign_already_finished", snapshot=snapshot)
                    return 0
                require(snapshot["state"] not in ("invalid", "drained"), "Remote preparation cannot proceed: " + repr(snapshot))
                self.no_stop()
                if snapshot["state"] == "ready":
                    break
            self.clock.sleep(min(60, max(0, deadline-self.clock.monotonic())))
        else:
            raise RuntimeError("Full campaign did not pass preflight within the 1800-second loading budget")
        owner = self.start_or_reuse(snapshot)
        self.wait_broker(owner)
        pending = self.remote.pending()  # Real SSHQueue.pending must succeed before the ready signal.
        self.no_stop()
        self.inspect_owner(owner)
        latest = self.remote.inspect(timeout=55)
        if latest["state"] == "campaign_finished":
            self.status("campaign_finished_during_startup", owner=owner, snapshot=latest)
            return 0
        require(latest["state"] == "ready" and latest["config_sha256"] == snapshot["config_sha256"],
                "Campaign changed while starting broker")
        if self.payload_path.exists():
            payload = broker.read_json(self.payload_path)
            require(payload.get("local_owner") == owner["local_owner"]
                    and payload.get("config_sha256") == snapshot["config_sha256"], "Stored ready payload differs")
        else:
            payload = dict(schema="astra_broker_ready_v1", model="gpt-6-astra", reasoning_effort="high",
                           remote_output=REMOTE_OUTPUT, config_sha256=snapshot["config_sha256"],
                           queue_read_succeeded=True, local_owner=owner["local_owner"],
                           broker_started_at_utc=owner["local_owner"]["start_utc"],
                           local_owner_sha256=sha(self.owner_path), broker_source_sha256=BROKER_SHA,
                           preflight_sha256=snapshot["preflight_sha256"], queue_pending_count=len(pending),
                           checked_at_utc=utc())
            write_exclusive(self.payload_path, payload)
        if latest.get("existing_ready") is not None:
            require(latest["existing_ready"] == payload, "Existing remote readiness belongs to another launch")
        result = self.remote.publish(payload)
        broker.write_json(self.local/"starter-ready-receipt.json", dict(payload=payload, publication=result, confirmed_at_utc=utc()))
        self.status("broker_ready", local_owner=owner["local_owner"], publication=result)
        return 0


def self_test():
    root = HERE/"broker_tests"/("full-starter-self-test-"+uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    class Clock:
        now = 0
        def monotonic(self): return self.now
        def sleep(self, seconds): self.now += seconds
    class FakeRemote:
        def __init__(self, clock): self.clock, self.ready, self.finished, self.reads = clock, None, False, 0
        def inspect(self, timeout):
            if self.finished: return dict(state="campaign_finished", finished={})
            if self.clock.now < 120: return dict(state="waiting_preflight")
            return dict(state="ready", config_sha256="c"*64, preflight_sha256="d"*64, existing_ready=self.ready)
        def pending(self): self.reads += 1; return []
        def publish(self, payload):
            if self.ready is not None: require(self.ready == payload, "Fake marker conflict")
            self.ready = payload
            return dict(result="created")
    class FakeProcesses:
        def __init__(self): self.launches, self.current = 0, None
        def launch(self, command, local):
            self.launches += 1
            self.current = dict(pid=12345, start_utc="2026-10-09T00:00:00Z", parent_pid=100,
                                command_line=subprocess.list2cmdline(command), executable_path=sys.executable)
            broker.write_json(local/"broker_status.json", dict(state="waiting", remote_output=REMOTE_OUTPUT,
                                                               model="gpt-6-astra", reasoning_effort="high"))
            return 12345
        def identity(self, pid): return self.current
    clock, processes = Clock(), FakeProcesses()
    remote = FakeRemote(clock)
    starter = Starter(root/"normal", remote, processes, clock)
    require(starter.run() == 0 and clock.now == 120 and processes.launches == 1 and remote.reads == 1,
            "Delayed preflight and one successful launch")
    require(starter.run() == 0 and processes.launches == 1, "Existing broker is reused without another launch")
    processes.current = None
    try: starter.run()
    except RuntimeError: pass
    else: raise AssertionError("Dead owner must prevent restart")
    require(processes.launches == 1, "Dead owner triggered a duplicate")
    finished_remote = FakeRemote(clock); finished_remote.finished = True
    finished = Starter(root/"finished", finished_remote, FakeProcesses(), clock)
    require(finished.run() == 0 and not finished.owner_path.exists(), "Finished campaign must never launch")
    broker.write_json(root/"normal"/"stop.json", {"kind":"synthetic stop"})
    try: starter.run()
    except RuntimeError: pass
    else: raise AssertionError("Existing stop must be preserved and block startup")

    # Run exactly the SSH Python source locally, with only the fixture root substituted.
    project = root/"remote_project"
    output = project/"results"/"astra-native-reset-full-20261009-v1"
    output.mkdir(parents=True)
    source = project/"source.py"; source.write_text("# fixture only\n")
    pinned = {str(source):sha(source)}
    failure = project/"failure.json"; broker.write_json(failure, {"fixture":True})
    manifest = dict(schema="astra_native_reset_control_rescue_v1", full_failure_subset=True,
                    arms=["control","astra"], jobs=[{}]*2008, selected_original_jobs=[{}]*1004)
    broker.write_json(output/"manifest.json", manifest)
    config = dict(schema="astra_native_reset_control_rescue_config_v1", output=str(output), models=[{}]*12,
                  source_sha256=pinned, manifest=str(output/"manifest.json"), manifest_sha256=sha(output/"manifest.json"),
                  failure_manifest=str(failure), failure_manifest_sha256=sha(failure))
    broker.write_json(output/"config.json", config)
    broker.write_json(output/"preflight.json", dict(passed=True, episodes=2008, original_cases=1004))
    def local_script(script, *args):
        script = script.replace("PROJECT = '"+PROJECT+"'", "PROJECT = "+repr(str(project)))
        script = script.replace("EXPECTED_OUTPUT = PROJECT + '/results/astra-native-reset-full-20261009-v1'",
                                "EXPECTED_OUTPUT = "+repr(str(output)))
        return subprocess.run([sys.executable,"-",str(output),json.dumps(pinned),*args],
                              input=script.encode(),capture_output=True,timeout=10,check=False)
    checked = local_script(REMOTE_INSPECT)
    require(checked.returncode == 0 and json.loads(checked.stdout)["state"] == "ready", "Remote verification fixture")
    payload = dict(remote.ready, remote_output=str(output), config_sha256=sha(output/"config.json"))
    published = local_script(REMOTE_PUBLISH_READY,json.dumps(payload))
    require(published.returncode == 0 and json.loads(published.stdout)["result"] == "created", "Exclusive readiness publication")
    original = (output/"broker-ready.json").read_bytes()
    conflict = local_script(REMOTE_PUBLISH_READY,json.dumps(dict(payload,local_owner=dict(payload["local_owner"],pid=54321))))
    require(conflict.returncode != 0 and (output/"broker-ready.json").read_bytes() == original, "Readiness conflict never overwrites")
    source.write_text("# changed fixture\n")
    rejected = local_script(REMOTE_INSPECT)
    require(json.loads(rejected.stdout)["state"] == "invalid", "Changed frozen source rejected")
    receipt = dict(status="passed", helper_sha256=sha(__file__), frozen_broker_sha256=sha(HERE/"broker.py"),
                   no_model_or_ssh_calls=True, tests=["delayed preflight", "single hidden-launch path via fake process",
                   "existing owner reused", "dead owner blocks restart", "finished campaign skips launch",
                   "stop retained", "remote source/config verification", "exclusive ready publication",
                   "ready conflict preserved", "source mutation rejected"])
    broker.write_json(root/"self_test_result.json",receipt)
    print(json.dumps(dict(receipt,receipt_path=str(root/"self_test_result.json"))),flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test: return self_test()
    require(os.name == "nt", "The full broker launcher is Windows-only")
    require(LOCAL_OUTPUT.resolve().drive.upper() == "D:", "Local outputs must remain on D:")
    LOCAL_OUTPUT.mkdir(parents=True,exist_ok=True)
    with broker.ProcessLock(LOCAL_OUTPUT/"starter.lock"):
        starter = Starter(LOCAL_OUTPUT,Remote(),WindowsProcesses())
        try:
            return starter.run()
        except BaseException as error:
            record = dict(error=repr(error),traceback=traceback.format_exc(),created_at_utc=utc(),
                          remote_output=REMOTE_OUTPUT,local_output=str(LOCAL_OUTPUT),helper_pid=os.getpid(),
                          automatic_restart_attempted=False,markers_removed=False)
            broker.write_json(LOCAL_OUTPUT/"starter-error.json",record)
            starter.status("error",error=repr(error))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
