"""Owner-scoped GPU lease: two hours for development, five days once scored."""
import fcntl,time,traceback
from pathlib import Path
from . import runner as r,services

def main():
    out=services.SERVICE_OUTPUT;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'lease.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.write(out/'lease.owner.json',r.identity(),exclusive=True)
    start=time.monotonic();deadline=start+7200;full_seen=False
    full=r.ROOT/'results/originx-confirmatory-20261009-v1'
    while time.monotonic()<deadline:
        if not full_seen and (full/'campaign.owner.json').exists():
            owner=r.read(full/'campaign.owner.json')
            if r.active(owner):
                full_seen=True;deadline=time.monotonic()+432000+3600
        if (full/'cleanup.json').exists() and r.read(full/'cleanup.json').get('all_owned_services_stopped'):return
        time.sleep(30)
    r.write(out/'lease-expired.json',dict(monotonic_elapsed=time.monotonic()-start,full_seen=full_seen),exclusive=True)
    # Request orderly draining only from exact, owned runner identities.
    for path in (r.ROOT/'results').glob('originx-confirmatory-*/rollout.owner.json'):
        owner=r.read(path)
        if str(Path(__file__).parent/'runner.py') in owner.get('command',[]) and r.active(owner):
            r.signal_owned(owner,15)
    end=time.monotonic()+2700
    while time.monotonic()<end:
        try:
            result=services.cleanup();r.write(out/'lease-cleanup.json',result,exclusive=True);return
        except Exception as error:
            r.write(out/'lease-cleanup-wait.json',dict(error=str(error),unix=time.time()))
            time.sleep(60)
    r.write(out/'lease-cleanup-error.json',dict(error='Owned services still connected or identity could not be verified; preserved',unix=time.time()),exclusive=True)
if __name__=='__main__':main()
