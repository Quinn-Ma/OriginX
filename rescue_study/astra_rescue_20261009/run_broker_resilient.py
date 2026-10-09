"""Bounded transport retries around the frozen broker, retaining its CLI receipts.

Use an existing --local-output. Resolving a reviewed network stop is explicit;
quota/auth/model stops and remote drain markers are never cleared here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import broker

HERE = Path(__file__).resolve().parent
FROZEN_SHA = "452bc7c53c63155ee250987312bccaf2b4f3d3d831e224aa8749d0cd0eac6ca4"
CODEX = Path(r"C:\Users\LuckyQinzhen\AppData\Local\OpenAI\Codex\bin\9691020b546a15b2\codex.exe")


def transient(error):
    if isinstance(error, subprocess.TimeoutExpired):
        command = error.cmd
        return isinstance(command, (list, tuple)) and Path(str(command[0])).name.lower() in ("ssh", "ssh.exe", "scp", "scp.exe")
    message = str(error)
    if re.search(r"permission denied|authentication failed|host key verification|remote host identification|quota|usage.limit|rate.?limit|hash mismatch|digest mismatch", message, re.I):
        return False
    return bool(re.search(r"\b(?:ssh|scp)\b", message, re.I)
                and re.search(r"connection (?:timed out|reset|closed|refused)|connect(?:ion)? timeout|broken pipe|no route to host|network is unreachable|TimeoutExpired", message, re.I))


class ResilientQueue(broker.SSHQueue):
    def __init__(self, remote_output, local, sleep=time.sleep):
        super().__init__(remote_output)
        self.log = Path(local)/"transport-retries.jsonl"
        self.sleep = sleep

    def retry(self, operation, call):
        for attempt in range(1, 4):
            started = time.monotonic()
            try:
                value = call()
            except Exception as error:
                retrying = attempt < 3 and transient(error)
                self.record(operation, attempt, started, status="retrying" if retrying else "failed", error=repr(error))
                if not retrying:
                    raise
                self.sleep((2, 5)[attempt-1])
            else:
                self.record(operation, attempt, started, status="ok")
                return value

    def record(self, operation, attempt, started, **extra):
        row = dict(at=broker.utc(), operation=operation, attempt=attempt,
                   latency_seconds=time.monotonic()-started, **extra)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def remote(self, source, *arguments):
        call = lambda: super(ResilientQueue, self).remote(source, *arguments)
        # Publication retries must upload fresh temporary artifacts again. A
        # previous successful commit may already have removed its temp files.
        if source == broker.REMOTE_PUBLISH:
            return call()
        return self.retry("ssh_drain" if source == broker.REMOTE_DRAIN else "ssh_read", call)

    def download(self, name, destination):
        broker.safe_name(name)
        destination.mkdir(parents=True, exist_ok=False)
        sources = [f"{broker.HOST}:{self.remote_output}/requests/{name}/{filename}"
                   for filename in ("request.json", "left.png", "right.png", "wrist.png")]
        def transfer():
            result = subprocess.run(["scp", *broker.SSH_OPTIONS, *sources, str(destination)],
                                    capture_output=True, timeout=180, check=False)
            if result.returncode:
                raise RuntimeError("SCP download failed: " + result.stderr.decode("utf-8", "replace")[-4000:])
        return self.retry("scp_download:"+name, transfer)

    def publish(self, name, source, request_sha, audit_files):
        return self.retry("publish_saved_response:"+name,
                          lambda: super(ResilientQueue, self).publish(name, source, request_sha, audit_files))


def resolve_network_stop(local):
    path = local/"stop.json"
    if not path.is_file():
        raise RuntimeError("Explicit stop resolution requires an existing stop.json")
    original = path.read_bytes()
    value = json.loads(original)
    if value.get("kind") != "broker_infrastructure_error" or not transient(value.get("message", "")):
        raise RuntimeError("Only a reviewed SSH/SCP transient infrastructure stop may be resolved")
    # An interrupted CLI cannot be rerun simply because a later transport failed.
    for manifest in (local/"batches").glob("*/batch.json"):
        batch = manifest.parent
        if not (batch/"completed.json").exists() and (batch/"cli_started.json").exists() and not (batch/"receipt.json").exists():
            raise RuntimeError("Unfinished CLI has no saved receipt; automatic retry is forbidden")
    archive = local/"resolved_network_stops"/(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())+"-"+uuid.uuid4().hex[:12])
    archive.mkdir(parents=True, exist_ok=False)
    if not archive.resolve().is_relative_to(local.resolve()):
        raise RuntimeError("Resolution archive escaped existing local state")
    with (archive/"stop.original.json").open("xb") as stream:
        stream.write(original); stream.flush(); os.fsync(stream.fileno())
    record = dict(schema="astra_network_stop_resolution_v1", explicitly_requested=True,
                  resolved_at=broker.utc(), stop_sha256=hashlib.sha256(original).hexdigest(),
                  original_kind=value["kind"], original_message=value["message"],
                  local_output=str(local), remote_drain_preserved=True, cli_receipts_preserved=True,
                  phase="backed_up_before_move", wrapper_sha256=broker.digest(Path(__file__).read_bytes()))
    broker.write_json(archive/"resolution.json", record)
    if path.read_bytes() != original:
        raise RuntimeError("Stop changed during review; preserved without resolution")
    path.rename(archive/"stop.moved.json")
    record["phase"] = "resolved"
    broker.write_json(archive/"resolution.json", record)
    return record


def self_test():
    from unittest.mock import patch
    root = HERE/"broker_tests"/("resilient-self-test-"+uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    waits, calls = [], []
    queue = ResilientQueue(broker.PROJECT+"/self-test-only", root, waits.append)
    def unstable():
        calls.append(1)
        if len(calls) < 3: raise RuntimeError("SSH failed: Connection reset by peer")
        return "saved response published"
    assert queue.retry("fixture", unstable) == "saved response published" and waits == [2, 5]
    assert not transient("SCP failed: Permission denied; Connection closed")
    assert not transient("CLI quota exhausted")
    assert transient(subprocess.TimeoutExpired(["scp", "fixture"], 10))
    stop = dict(kind="broker_infrastructure_error", message="RuntimeError('SCP upload failed: Connection timed out')")
    broker.write_json(root/"stop.json", stop)
    original = (root/"stop.json").read_bytes()
    resolution = resolve_network_stop(root)
    assert not (root/"stop.json").exists() and resolution["phase"] == "resolved"
    assert next((root/"resolved_network_stops").glob("*/stop.original.json")).read_bytes() == original
    broker.write_json(root/"stop.json", dict(kind="cli_quota_or_rate_limit", message="quota"))
    try: resolve_network_stop(root)
    except RuntimeError: pass
    else: raise AssertionError("Quota stop must never be resolved")

    # A saved successful batch is replayed through Broker.finish. Any CLI call
    # fails this test; only one still-unpublished response may reach the queue.
    (root/"stop.json").rename(root/"synthetic-quota-stop-preserved.json")
    name = "native-B-0001-Case-1--astra"
    batch = root/"batches"/"saved-fixture"
    (batch/"requests"/name).mkdir(parents=True)
    request_path = batch/"requests"/name/"request.json"
    request_path.write_text('{"fixture":"saved request"}', encoding="utf-8")
    token = "a"*32
    manifest = dict(batch_id=batch.name, requests=[dict(request_id=name, request_token=token,
                     request_sha256=broker.digest(request_path.read_bytes()), validation_error=None)])
    broker.write_json(batch/"batch.json", manifest)
    broker.write_json(batch/"validated_response.json", {name:dict(subgoal_instruction="Previously generated advice.", rationale="Saved.")})
    broker.write_json(batch/"receipt.json", dict(status="ok", cli_executed=True,
                      batch_manifest_sha256=broker.digest((batch/"batch.json").read_bytes()),
                      validated_response_sha256=broker.digest((batch/"validated_response.json").read_bytes())))
    published = []
    queue.publish = lambda *args: (published.append(args[0]) or dict(same_response=True,result="published"))
    instance = broker.Broker(root, queue, ["NEVER_EXECUTE"])
    with patch.object(broker, "run_cli", side_effect=AssertionError("CLI must not run during saved recovery")):
        instance.recover()
        instance.recover()
    assert published == [name]
    receipt = dict(status="passed", wrapper_sha256=broker.digest(Path(__file__).read_bytes()),
                   frozen_broker_sha256=broker.digest((HERE/"broker.py").read_bytes()),
                   no_real_ssh_or_cli_calls=True, tests=["bounded transient retries", "2/5-second backoff",
                   "authentication/quota excluded", "network stop preserved before move", "quota stop cannot resolve",
                   "saved batch publishes without CLI", "second recovery does not republish"])
    broker.write_json(root/"self_test_result.json", receipt)
    print(json.dumps(dict(receipt, receipt_path=str(root/"self_test_result.json"))), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-output", type=Path)
    parser.add_argument("--resolve-network-stop", action="store_true")
    parser.add_argument("--run-seconds", type=float, default=30000)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: return self_test()
    if not args.local_output: parser.error("--local-output must select the existing broker state")
    local = args.local_output.resolve()
    if not (local/"identity.json").is_file(): parser.error("Existing identity.json is required; a new state is forbidden")
    if args.run_seconds <= 0: parser.error("--run-seconds must be positive")
    if os.name == "nt" and local.drive.upper() != "D:": parser.error("Existing state must be on D:")
    if broker.digest((HERE/"broker.py").read_bytes()) != FROZEN_SHA: parser.error("Frozen broker source changed")
    identity = broker.read_json(local/"identity.json")
    remote = broker.safe_remote(identity["remote_output"])
    expected = dict(schema="astra_broker_identity_v1", remote_output=remote, ssh_host=broker.HOST,
                    model=broker.MODEL, reasoning_effort=broker.EFFORT)
    if identity != expected: parser.error("Existing state identity differs from pinned model/queue")
    if CODEX.is_file():
        os.environ["PATH"] = str(CODEX.parent)+os.pathsep+os.environ.get("PATH", "")
        cli = str(CODEX)
    else:
        cli = shutil.which("codex")
    if not cli: parser.error("Codex executable unavailable")
    with broker.ProcessLock(local/"broker.lock"):
        if args.resolve_network_stop:
            print(json.dumps(resolve_network_stop(local)), flush=True)
        instance = broker.Broker(local, ResilientQueue(remote, local), [cli])
        try:
            return instance.run(args.run_seconds, once=args.once)
        except Exception as error:
            instance.stop("broker_infrastructure_error", repr(error))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
