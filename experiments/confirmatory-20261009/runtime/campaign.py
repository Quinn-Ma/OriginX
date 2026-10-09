"""Bounded, owner-recorded development/confirmation campaign supervisor."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import runner as r,services
ROOT=r.ROOT;HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--duration-seconds',type=int,default=432000)
    args=p.parse_args();out=ROOT/'results'/('originx-confirmatory-development-20261009-v2' if args.development else 'originx-confirmatory-20261009-v1')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.require(not (out/'campaign.owner.json').exists(),'Prior campaign exists, inspect rather than duplicate')
    r.write(out/'campaign.owner.json',dict(r.identity(),development=args.development,unix=time.time()),exclusive=True)
    start=time.monotonic();active=None;failure=None;phase='starting'
    def state(name,**extra):
        nonlocal phase
        phase=name;r.write(out/'campaign-status.json',dict(phase=phase,development=args.development,elapsed_seconds=time.monotonic()-start,unix=time.time(),**extra))
    def command(label,cmd,env=None,timeout=3600):
        nonlocal active
        state(label)
        with (out/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(r.identity(child.pid),command=cmd,cwd=str(ROOT))
            r.write(out/(label+'.owner.json'),active,exclusive=True)
            try:rc=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                r.signal_owned(active,15)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:r.signal_owned(active,9);child.wait(timeout=10)
                raise
            finally:active=None
        r.require(rc==0,f'{label} exited {rc}; inspect preserved log')
    try:
        if not args.development:
            gate=r.read(ROOT/'results/originx-confirmatory-development-20261009-v2/admission-review.json')
            r.require(gate.get('passed') is True and gate.get('no_confirmatory_outcomes_seen') is True,'Development admission required')
        state('waiting_services')
        ready=services.SERVICE_OUTPUT/'services-ready.json';deadline=time.monotonic()+1800
        while not ready.exists():
            r.require(time.monotonic()<deadline,'Service readiness timeout');time.sleep(5)
        models=r.read(ready)['models'];r.require(all(services.active(m['owner']) for m in models),'Ready services no longer alive')
        env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'remote_processor_candidate_v1/deps'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        probe_output=services.SERVICE_OUTPUT/'probes'/('development-v2' if args.development else 'confirmatory-v1')
        command('socket-probe',[str(ROOT/'envs/training/bin/python'),'-u','-m','originx_confirmatory_20261009.probe','--services',str(ready),'--output',str(probe_output)],env,1200)
        cmd=[sys.executable,'-u',str(HERE/'runner.py'),'prepare','--services-ready',str(ready),'--output',str(out),'--duration-seconds',str(args.duration_seconds)]
        if args.development:cmd.append('--development')
        command('prepare',cmd,env,300)
        config=r.read(out/'config.json');simulation=r.environment(config)
        command('preflight',[config['python'],'-u',str(HERE/'runner.py'),'preflight','--config',str(out/'config.json')],simulation,300)
        state('waiting_broker');deadline=time.monotonic()+1800
        while not (out/'broker-ready.json').exists():
            r.require(time.monotonic()<deadline,'Broker readiness timeout');time.sleep(5)
        command('rollout',[config['python'],'-u',str(HERE/'runner.py'),'run','--config',str(out/'config.json')],simulation,args.duration_seconds+300)
        command('aggregate',[sys.executable,'-u','-m','originx_confirmatory_20261009.aggregate','--config',str(out/'config.json')],env,300)
        state('completed',result=str(out/'analysis/report.json'))
    except BaseException:
        failure=traceback.format_exc();r.write(out/'campaign-error.json',dict(phase=phase,error=failure,unix=time.time()),exclusive=True)
        if active and r.active(active):r.signal_owned(active,15)
        state('failed',error=failure)
    finally:
        # Scored run always releases this namespace. Development keeps the
        # admitted services briefly for the explicit root review/expansion,
        # but a watchdog bounds that lease if the interactive agent disappears.
        if failure or not args.development:
            try:r.write(out/'cleanup.json',services.cleanup(),exclusive=True)
            except BaseException:r.write(out/'cleanup-error.json',dict(error=traceback.format_exc()),exclusive=True)
        r.write(out/'campaign-finished.json',dict(development=args.development,phase=phase,failed=failure is not None,
                unix=time.time(),elapsed_seconds=time.monotonic()-start),exclusive=True)
    return 1 if failure else 0
if __name__=='__main__':raise SystemExit(main())
