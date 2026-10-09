"""One-use external UTC deadline enforcement; never selects model processes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import signal
import time

ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NAME = 'originx-confirmatory-multigpu-20261009-v2'
END = datetime.fromisoformat('2026-10-09T22:40:12+00:00')

def require(value, message):
    if not value:
        raise RuntimeError(message)

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def option(command, name):
    require(command.count(name) == 1, 'Argument missing or duplicated: '+name)
    index = command.index(name)
    require(index+1 < len(command), 'Missing argument value')
    return command[index+1]

def scoped_role(owner, config, config_sha, root=ROOT):
    require(type(owner.get('pid')) is int and owner['pid'] > 1
            and type(owner.get('process_start_ticks')) is int
            and isinstance(owner.get('command'), list)
            and all(isinstance(x, str) for x in owner['command']), 'Malformed owner')
    command = owner['command']
    require(owner.get('cwd') == str(root), 'Owner cwd differs')
    require(option(command, '--config') == str(config), 'Config argv differs')
    campaign = str(root/'continue_multigpu_v2.py')
    runner = str(root/'originx_confirmatory_20261009/runner.py')
    if campaign in command:
        require(command.count(campaign) == 1 and command[command.index(campaign)+1] == 'campaign', 'Wrong campaign role')
        require(option(command, '--config-sha256') == config_sha, 'Campaign SHA argv differs')
        return 'campaign'
    require(runner in command and command.count(runner) == 1
            and command[command.index(runner)+1] == 'episode', 'Not an episode worker')
    require(option(command, '--config-sha') == config_sha, 'Worker SHA argv differs')
    return 'episode'

def snapshot(pid):
    proc = Path('/proc')/str(pid)
    try:
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return dict(pid=pid, process_start_ticks=int(fields[19]),
                    command=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                    cwd=str((proc/'cwd').resolve()))
    except (FileNotFoundError, ProcessLookupError):
        return None

def bound(owner, observe=snapshot):
    live = observe(owner['pid'])
    if live is None:
        return False
    require(all(live.get(k) == owner[k] for k in ('pid', 'process_start_ticks', 'command', 'cwd')),
            'Live process differs from recorded owner; no signal')
    return True

def enforce(output, config_sha, parent_utc, root=ROOT, observe=snapshot,
            open_fd=None, send=None, close=None, sleep=time.sleep):
    require(datetime.fromisoformat(parent_utc.replace('Z', '+00:00')) >= END,
            'Parent UTC is before fixed deadline')
    config = output/'config.json'
    require(output == root/'results'/NAME and sha(config) == config_sha
            and read(config)['output'] == str(output), 'Config/output binding differs')
    result = dict(schema='originx_external_deadline_v2', config_sha256=config_sha,
                  parent_utc=parent_utc, deadline_utc=END.isoformat(), checked=[], signals=[],
                  errors=[], models_excluded=True, foreign_processes_untouched=True)
    def write_new(name, value):
        with (output/name).open('x') as f:
            json.dump(value, f, indent=2)
    write_new('external-deadline-intent.json', result)
    if (output/'campaign-finished.json').exists():
        result['already_terminal'] = True
        write_new('external-deadline-receipt.json', result)
        return result
    handles = []
    open_fd = open_fd or os.pidfd_open
    send = send or signal.pidfd_send_signal
    close = close or os.close
    def event(value):
        with (output/'external-deadline-events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n'); f.flush(); os.fsync(f.fileno())
    def dispatch(owner, fd, sig):
        if not bound(owner, observe):
            return False
        event(dict(phase='signal_intent', pid=owner['pid'], signal=int(sig), pidfd=True))
        send(fd, sig)
        value=dict(pid=owner['pid'], signal=int(sig), pidfd=True)
        result['signals'].append(value); event(dict(phase='signal_sent', **value))
        return True
    try:
        seen = set()
        def register(path):
            try:
                owner = read(path); role = scoped_role(owner, config, config_sha, root)
                key = (owner['pid'], owner['process_start_ticks'])
                if key in seen:
                    return
                seen.add(key)
                if role == 'episode':
                    claim = read(output/'claims'/path.name)
                    require(claim['config_sha256'] == config_sha and claim['command'] == owner['command'], 'Claim owner/config differs')
                if not bound(owner, observe):
                    result['checked'].append(dict(path=str(path), inactive=True)); return
                fd = open_fd(owner['pid'])
                try:
                    require(bound(owner, observe), 'Owner exited after pidfd open')
                except BaseException:
                    close(fd); raise
                handles.append((owner, role, fd))
                result['checked'].append(dict(path=str(path), role=role, identity_verified=True))
            except Exception as exc:
                result['errors'].append(dict(path=str(path), error=repr(exc), no_signal=True))
        register(output/'campaign.owner.json')
        # Allow a Popen already in progress to publish its owner (registration bound: 10s).
        for owner, role, fd in handles:
            require(role == 'campaign', 'Campaign owner has wrong role')
            try:
                if dispatch(owner, fd, signal.SIGTERM):
                    sleep(12)
            except Exception as exc:
                result['errors'].append(dict(pid=owner['pid'], error=repr(exc), stage='campaign_term'))
        for path in sorted((output/'processes').glob('*.json')):
            register(path)
        for owner, role, fd in handles:
            if role != 'episode':
                continue
            try:
                dispatch(owner, fd, signal.SIGTERM)
            except Exception as exc:
                result['errors'].append(dict(pid=owner['pid'], error=repr(exc), stage='term'))
        if any(role == 'episode' for _, role, _ in handles):
            sleep(15)
        for owner, role, fd in handles:
            if role != 'episode':
                continue
            try:
                dispatch(owner, fd, signal.SIGKILL)
            except Exception as exc:
                result['errors'].append(dict(pid=owner['pid'], error=repr(exc), stage='kill'))
    finally:
        for _, _, fd in handles:
            close(fd)
        write_new('external-deadline-receipt.json', result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--parent-utc', required=True)
    args=parser.parse_args()
    print(json.dumps(enforce(ROOT/'results'/NAME, args.config_sha256, args.parent_utc)))

if __name__ == '__main__':
    main()
