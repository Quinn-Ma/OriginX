"""Exclusive sprint director: engineering admission, XAB, B/V, owned cleanup."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from campaign import ROOT,SERVICE,tracker
from episode import read,write,sha,require,identity,environment,active,signal_owned

def run(a):
    out=HERE/'director';out.mkdir(exist_ok=False);write(out/'owner.json',identity(),exclusive=True)
    T=tracker();children=[];begin=time.monotonic();hard=begin+a.hard_seconds;dispatch=begin+a.dispatch_seconds
    stop=[];signal.signal(signal.SIGTERM,lambda *_:stop.append('requested_stop'))
    def spawn(label,command,env=None):
        log=(out/(label+'.log')).open('x');p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        t=T(p,command,ROOT);children.append((label,t,log));write(out/(label+'.pending.json'),t.receipt(),exclusive=True)
        write(out/(label+'.owner.json'),t.await_identity(),exclusive=True);return t
    def wait(t,limit):
        while t.child.poll() is None:
            if time.monotonic()>min(limit,hard) or stop:t.drain(grace=10,kill_timeout=5);break
            time.sleep(1)
        t.close();require(t.child.returncode==0,'Child failed: '+repr(t.command))
    python=str(ROOT/'envs/training/bin/python');cp=HERE/'campaign.py'
    def prepare(name,dev=False,bv=False):
        output=ROOT/'results'/name;cmd=[python,'-B',str(cp),'prepare','--output',str(output)]
        if dev:cmd.append('--development')
        if bv:cmd.append('--bv')
        wait(spawn(name+'-prepare',cmd),time.monotonic()+180)
        c=read(output/'config.json');cmd=[c['python'],'-B',str(cp),'preflight','--config',str(output/'config.json')]
        wait(spawn(name+'-preflight',cmd,environment(c)),time.monotonic()+180);return output,c
    def campaign(label,output,c,pilot=False):
        remain=max(60,min(1200 if pilot else 100000,hard-time.monotonic()-90))
        ds=min(remain,120 if pilot else max(0,dispatch-time.monotonic()))
        cmd=[python,'-B','-u',str(cp),'run','--config',str(output/'config.json'),'--hard-seconds',str(remain),'--dispatch-seconds',str(ds),'--authoritative-utc',a.authoritative_utc]
        return spawn(label,cmd)
    def aggregate(label,output,c,pilot=False):
        cmd=[c['python'],'-B',str(HERE/'aggregate.py'),'--config',str(output/'config.json'),'--output',str(output/'analysis')]
        if pilot:cmd.append('--pilot')
        wait(spawn(label,cmd,environment(c)),time.monotonic()+180)
    failure=None
    try:
        while not (SERVICE/'ready.json').exists():
            require(not (SERVICE/'first-error.json').exists(),'Service admission failed')
            require(time.monotonic()<min(begin+1800,dispatch) and not stop,'Service admission deadline')
            time.sleep(3)
        pilot,pc=prepare('originx-sprint-xab-engineering-20261011-v1',dev=True)
        wait(campaign('engineering',pilot,pc,True),time.monotonic()+1230);aggregate('engineering-admission',pilot,pc,True)
        core,c=prepare('originx-sprint-xab-20261011-v1')
        core_t=campaign('core',core,c)
        bv,bc=prepare('originx-sprint-bv-20261011-v1',bv=True)
        bv_t=campaign('bv',bv,bc)
        write(out/'formal-launched.json',dict(core=str(core),bv=str(bv),core_owner=core_t.owner,bv_owner=bv_t.owner,manifest_sha256=c['manifest_sha256']),exclusive=True)
        wait(core_t,hard);wait(bv_t,hard)
        aggregate('core-aggregate',core,c);aggregate('bv-aggregate',bv,bc)
    except BaseException:
        failure=traceback.format_exc();write(out/'first-error.json',dict(error=failure),exclusive=True)
    finally:
        configs=[]
        for name in ['originx-sprint-xab-engineering-20261011-v1','originx-sprint-xab-20261011-v1','originx-sprint-bv-20261011-v1']:
            path=ROOT/'results'/name
            if (path/'config.json').exists():configs.append(path)
        # Failure may occur after a sibling scheduler started. Drain dispatch,
        # then end exact owned workers before ending their scheduler parent.
        for label,t,log in children:
            if label in ('engineering','core','bv') and t.child.poll() is None:t.send(signal.SIGTERM)
        time.sleep(1)
        processes=[read(p) for d in configs for p in (d/'processes').glob('*.json')]
        for owner in processes:
            if active(owner):signal_owned(owner,signal.SIGTERM)
        end=time.monotonic()+10
        while any(active(o) for o in processes) and time.monotonic()<end:time.sleep(.2)
        for owner in processes:
            if active(owner):signal_owned(owner,signal.SIGKILL)
        for label,t,log in reversed(children):
            if t.child.poll() is None:t.drain(grace=15,kill_timeout=10)
            else:t.close()
            log.close();write(out/(label+'.exit.json'),t.receipt(),exclusive=True)
        # The service parent owns the models. Request a drain only after all
        # direct workers' schedulers have reaped their child workers.
        processes=[read(p) for d in configs for p in (d/'processes').glob('*.json')]
        inactive=all(not active(o) for o in processes)
        write(out/'worker-cleanup.json',dict(all_workers_inactive=inactive,owners=processes),exclusive=True)
        if inactive and (SERVICE/'services-ready.json').exists() and not (SERVICE/'release-request.json').exists():
            write(SERVICE/'release-request.json',dict(schema='originx_multigpu_service_release_v1',services_ready_sha256=sha(SERVICE/'services-ready.json'),all_assigned_workers_stopped=True),exclusive=True)
        end=time.monotonic()+120
        while not (SERVICE/'finished.json').exists() and time.monotonic()<end:time.sleep(2)
        profiles=read(SERVICE/'services-ready.json')['models'] if (SERVICE/'services-ready.json').exists() else []
        models=[dict(owner=m['owner'],port=m['port'],still_active=active(m['owner'])) for m in profiles]
        write(out/'transfer-cleanup.json',dict(all_workers_inactive=inactive,models=models,all_models_inactive=all(not m['still_active'] for m in models),
              service_finished=read(SERVICE/'finished.json') if (SERVICE/'finished.json').exists() else None,
              service_cleanup_sha256=sha(SERVICE/'cleanup.json') if (SERVICE/'cleanup.json').exists() else None),exclusive=True)
        write(out/'completion.json',dict(failed=failure is not None,elapsed_seconds=time.monotonic()-begin,all_direct_children_reaped=all(t.reaped for _,t,_ in children)),exclusive=True)
    return 0 if failure is None else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hard-seconds',type=float,required=True);p.add_argument('--dispatch-seconds',type=float,required=True);p.add_argument('--authoritative-utc',required=True)
    sys.exit(run(p.parse_args()))
