"""Start this one continuation after exact prior owners have drained."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
HERE = ROOT/'astra_rescue_continuation_20261009'
PRIOR = ROOT/'results/astra-native-reset-full-20261009-v1'
OLD_SERVICES = ROOT/'results/astra-rescue-services-20261009-v1'
SERVICES = ROOT/'results/astra-rescue-services-20261009-v2'
OUTPUT = ROOT/'results/astra-native-reset-continuation-20261009-v2'
PYTHON = ROOT/'envs/training/bin/python'

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

def identity(pid):
    proc=Path('/proc')/str(pid)
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    if stat[0] in ('Z','X'):
        raise ProcessLookupError('Exited')
    return dict(pid=pid,process_start_ticks=int(stat[19]),
                command=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str((proc/'cwd').resolve()))

def live(owner):
    try:
        actual=identity(owner['pid'])
    except (FileNotFoundError,ProcessLookupError):
        return False
    if actual != {k:owner[k] for k in actual}:
        raise RuntimeError('Prior PID identity changed; preserve all processes')
    return True

def main():
    import fcntl
    with (HERE/'bootstrap.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'bootstrap-owner.json').exists():
            raise RuntimeError('Previous bootstrap owner exists; duplicate launch forbidden')
        write(HERE/'bootstrap-owner.json',identity(os.getpid()))
        inventory=read(HERE/'deployment-inventory.json')
        for name,expected in inventory['files'].items():
            assert Path(name).name==name
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
        end=time.monotonic()+3600
        while not (PRIOR/'campaign-finished.json').exists():
            if time.monotonic()>end:
                raise TimeoutError('Prior campaign did not terminate; nothing launched')
            time.sleep(10)
        assert read(PRIOR/'completion.json')['all_children_drained'] is True
        owners=list((PRIOR/'processes').glob('*.json'))
        owners += [PRIOR/'campaign-owner.json',PRIOR/'supervisor-owner.json']
        owners += list((OLD_SERVICES/'services').glob('*/owner.json'))
        # Finish marker can precede process exit by milliseconds.
        while any(live(read(p)) for p in owners if p.exists()):
            if time.monotonic()>end:
                raise TimeoutError('Prior owned processes remain active; nothing launched')
            time.sleep(5)
        if SERVICES.exists() or OUTPUT.exists():
            raise RuntimeError('Continuation output already exists; duplicate launch forbidden')
        SERVICES.mkdir()
        env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'remote_processor_candidate_v1/deps'),
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (SERVICES/'launcher.log').open('xb') as log:
            launcher=subprocess.Popen([str(PYTHON),'-u',str(HERE/'launch_models.py'),
                    '--output',str(SERVICES),'--slots-per-gpu','6'],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(SERVICES/'launcher-owner.json',identity(launcher.pid))
        OUTPUT.mkdir()
        with (OUTPUT/'campaign.log').open('xb') as log:
            child=subprocess.Popen([str(PYTHON),'-u','-m',
                'astra_rescue_continuation_20261009.continuation_campaign'],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(HERE/'dispatch.json',dict(launcher=identity(launcher.pid),campaign=identity(child.pid),
               source_inventory_sha256=hashlib.sha256((HERE/'deployment-inventory.json').read_bytes()).hexdigest(),
               original_denominator=1004,continuation_cases=968,prior_owners_inactive=True))
        print(json.dumps(read(HERE/'dispatch.json')),flush=True)

if __name__=='__main__':
    main()
