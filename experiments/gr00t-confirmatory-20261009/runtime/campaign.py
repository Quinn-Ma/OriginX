"""Independent bounded GR00T campaign; admits existing parity, never loads GPU.

The caller supplies pinned source/service/roster references. Development runs
four C/V outcomes; full runs all 5,000 only after an independent dev review.
The generated runtime admission binds evidence, not a user-permission claim.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from . import runner as r


def verify_parity(path,digest,model):
    from .parity_probe import RNG_KEYS,STATE_KEYS,SEEDS,SEQUENCE
    r.require(r.sha(path)==digest,'Socket parity hash changed')
    result=r.read(path)
    r.require(result.get('schema')=='originx_gr00t_socket_parity_v1' and result.get('passed') is True
        and result.get('clients')==6 and result.get('hold_seconds',0)>=360
        and 0<result.get('interval_seconds',999)<=120 and result.get('no_reconnect') is True
        and result.get('one_reset') is True,'Six-client 360-second parity admission not passed')
    r.require(result.get('profile')==model['server_manifest'] and result.get('profile_sha256')==model['server_manifest_sha256'],
        'Parity profile differs from rollout service')
    reference_path=Path(result['reference']);r.require(r.sha(reference_path)==result['reference_sha256'],'Direct parity hash changed')
    reference=r.read(reference_path)
    r.require(reference.get('schema')=='originx_gr00t_direct_parity_v1' and reference.get('passed') is True
        and reference.get('ambient_rng_unchanged') is True and reference.get('stream_count')==6
        and reference.get('seeds')==list(SEEDS) and reference.get('sequence')==list(SEQUENCE), 'Invalid direct policy reference')
    r.require(reference['identity']['asset_source_identity_sha256']==model['hello']['asset_source_identity_sha256']
        and reference['model_config_runtime_sha256']==model['hello']['model_config_runtime_sha256'],
        'Direct/service numerical identity changed')
    fixtures_path=Path(reference['fixtures']);r.require(r.sha(fixtures_path)==reference['fixtures_sha256'],'Fixture manifest changed')
    fixtures=r.read(fixtures_path)
    r.require(fixtures.get('synthetic') is False and len(fixtures.get('fixtures',[]))==2,'Use two actual native development observations')
    for fixture in fixtures['fixtures']:
        r.require(r.sha(fixture['path'])==fixture['sha256'] and fixture.get('environment_steps')==0
            and fixture.get('policy_queries')==0 and fixture.get('gym_resets')==1,'Native fixture proof differs')
    capture_path=fixtures_path.parent/'capture-receipt.json'
    r.require(r.sha(capture_path)==fixtures['capture_receipt_sha256'] and r.read(capture_path).get('passed') is True,
        'Native capture/guard receipt not verified')
    connections=set();client_checks=[]
    for index in range(6):
        client_path=Path(path).parent/f'client-{index}.json';client=r.read(client_path)
        trace=reference['streams'][index]['trace'];queries=client['queries']
        r.require(client.get('passed') is True and client.get('resets')==1 and client.get('reconnects')==0
            and client.get('hold_seconds_actual',0)>=360 and client.get('seed')==SEEDS[index]
            and len(queries)==3 and client.get('keepalives'),'Client parity incomplete')
        connections.add(client['connection_id'])
        r.require(client['reset_ack']['seed']==SEEDS[index] and client['reset_ack']['requests_since_reset']==0,
            'Client RNG seed acknowledgment changed')
        for qi,(query,ref) in enumerate(zip(queries,trace)):
            rng=query['rng']
            r.require(query['sequence_index']==qi and query.get('exact') is True and query['action_sha256']==ref['action_sha256'],
                'Direct/socket action hash differs')
            r.require(rng['seed']==SEEDS[index] and rng['requests_since_reset']==qi+1
                and rng['connection_id']==client['connection_id'] and rng['server_instance']==model['hello']['server_instance']
                and all(rng[k]==ref['rng'][k] for k in RNG_KEYS),'Direct/socket RNG identity differs')
        r.require(queries[2].get('after_hold') is True,'Final forward is not after socket hold')
        for hold in client['keepalives']:
            r.require(hold.get('unchanged') is True and all(hold['rng'][k]==queries[1]['rng'][k] for k in STATE_KEYS),
                'Keepalive modified connection or RNG state')
        client_checks.append(dict(path=str(client_path),sha256=r.sha(client_path),passed=True))
    r.require(len(connections)==6,'Six distinct persistent clients required')
    return dict(path=str(Path(path).resolve()),sha256=digest,direct_reference_sha256=result['reference_sha256'],
        native_fixtures_sha256=reference['fixtures_sha256'],clients=client_checks,passed=True)


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',action='store_true');p.add_argument('--duration-seconds',type=int)
    p.add_argument('--config',type=Path);p.add_argument('--config-sha')
    p.add_argument('--reference-config',type=Path);p.add_argument('--reference-config-sha256')
    p.add_argument('--reference-manifest',type=Path);p.add_argument('--reference-manifest-sha256')
    p.add_argument('--service-manifest',type=Path);p.add_argument('--service-manifest-sha256')
    p.add_argument('--source-freeze',type=Path);p.add_argument('--source-freeze-sha256')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--shared-budget-evidence',type=Path);p.add_argument('--shared-budget-evidence-sha256')
    p.add_argument('--parity-result',type=Path,required=True);p.add_argument('--parity-result-sha256',required=True)
    p.add_argument('--development-review',type=Path);p.add_argument('--development-review-sha256')
    args=p.parse_args()
    output=r.ROOT/'results'/r.OUTPUT_NAMES[args.development]
    control=r.HERE/'campaigns'/output.name;control.mkdir(parents=True,exist_ok=True)
    lock=(control/'campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.require(not (control/'owner.json').exists(),'Previous independent campaign owner exists; inspect instead of restarting')
    owner=dict(r.identity(),namespace=r.NAMESPACE,development=args.development,unix=time.time())
    r.write(control/'owner.json',owner,True)
    start=time.monotonic();active=None;phase='starting';failure=None;config=None;rollout_returncode=None;drain=False
    def state(name,**more):
        nonlocal phase
        phase=name;value=dict(schema='originx_gr00t_campaign_status_v1',phase=phase,
            development=args.development,elapsed_seconds=time.monotonic()-start,unix=time.time(),**more)
        r.write(control/'status.json',value)
        if (output/'config.json').exists():r.write(output/'campaign-status.json',value)
    def request_drain(*_):
        nonlocal drain
        drain=True
        if (output/'config.json').exists() and not (output/'drain.request.json').exists():
            r.write(output/'drain.request.json',dict(reason='campaign_signal',owner=owner),True)
    signal.signal(signal.SIGTERM,request_drain);signal.signal(signal.SIGINT,request_drain)
    def command(label,cmd,env,timeout,allow_failure=False):
        nonlocal active
        state(label)
        with (control/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(cmd,cwd=r.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active=r.TrackedChild(child,cmd,r.ROOT)
            try:
                child_owner=active.await_identity();r.write(control/(label+'.owner.json'),child_owner,True)
                code=child.wait(timeout=timeout)
            except BaseException:
                request_drain();active.drain(grace=45)
                raise
            finally:
                if child.poll() is not None:active.close()
                r.write(control/(label+'.spawn.json'),active.receipt(),True)
                if child.poll() is not None:active=None
        if not allow_failure:r.require(code==0,f'{label} exited {code}; original log retained')
        return code
    try:
        state('prepare')
        if args.config is None:
            args.output=output
            required=('reference_config','reference_config_sha256','reference_manifest','reference_manifest_sha256',
                      'service_manifest','service_manifest_sha256','source_freeze','source_freeze_sha256','checkpoint')
            r.require(all(getattr(args,k) is not None for k in required),'All source/config/service preparation pins are required')
            r.prepare(args);args.config=output/'config.json';args.config_sha=r.sha(args.config)
        else:
            r.require(args.config.resolve()==output/'config.json' and r.sha(args.config)==args.config_sha,'Existing config path/hash differs')
        config=r.read(args.config);r.check_sources(config)
        r.require(config['development'] is args.development,'Campaign cohort differs from prepared config')
        r.write(output/'campaign.owner.json',owner,True)
        numerical=verify_parity(args.parity_result,args.parity_result_sha256,config['models'][0])
        admission=dict(schema='originx_gr00t_runtime_admission_v1',passed=True,development=args.development,
            config_sha256=args.config_sha,manifest_sha256=config['manifest_sha256'],socket_parity=numerical,
            profiles=[dict(path=m['server_manifest'],sha256=m['server_manifest_sha256']) for m in config['models']],
            no_confirmatory_outcomes_seen=True)
        if not args.development:
            r.require(args.development_review is not None and args.development_review_sha256 is not None,'Full-cohort development review required')
            r.require(r.sha(args.development_review)==args.development_review_sha256,'Development review digest differs')
            dev=r.read(args.development_review)
            r.require(dev.get('passed') is True and dev.get('no_confirmatory_outcomes_seen') is True,'Development did not pass independent admission')
            admission['development_review']=dict(path=str(args.development_review.resolve()),sha256=args.development_review_sha256)
        r.write(output/'runtime-admission.json',admission,True)
        simulation=r.environment(config)
        if not (output/'preflight.json').exists():
            command('preflight',[config['python'],'-u','-m',r.NAMESPACE+'.runner','preflight',
                '--config',str(args.config),'--config-sha',args.config_sha],simulation,300)
        state('waiting_broker');limit=time.monotonic()+1800
        while not (output/'broker-ready.json').exists():
            r.require(not drain and time.monotonic()<limit,'Broker readiness timeout or requested drain');time.sleep(5)
        r.validate_broker_ready(config,args.config_sha,r.read(output/'broker-ready.json'))
        r.require(not drain,'Campaign was drained before dispatch')
        rollout_returncode=command('rollout',[config['python'],'-u','-m',r.NAMESPACE+'.runner','run',
            '--config',str(args.config),'--config-sha',args.config_sha],simulation,config['duration_seconds']+120,True)
        completion=r.read(output/'completion.json')
        r.require(completion.get('all_children_drained') is True,'Own workers have not fully drained')
        # Aggregate even after a clean stop/technical failure; all missing work
        # remains unknown on the immutable denominator.
        command('aggregate',[config['python'],'-u','-m',r.NAMESPACE+'.aggregate','--config',str(args.config)],simulation,600)
        state('completed' if rollout_returncode==0 else 'terminated_with_unknowns',result=str(output/'analysis/report.json'),rollout_returncode=rollout_returncode)
    except BaseException:
        failure=traceback.format_exc();r.write(control/'campaign-error.json',dict(phase=phase,error=failure),True)
        if (output/'config.json').exists():r.write(output/'campaign-error.json',dict(phase=phase,error=failure,unix=time.time()),True)
        if active:
            try:active.drain(grace=45);active=None
            except BaseException:r.write(control/'child-drain-error.json',dict(error=traceback.format_exc()),True)
        state('failed',error=failure)
        if config and (output/'completion.json').exists() and r.read(output/'completion.json').get('all_children_drained') is True:
            try:
                if not (output/'analysis/report.json').exists():
                    command('aggregate-after-failure',[config['python'],'-u','-m',r.NAMESPACE+'.aggregate',
                        '--config',str(args.config)],r.environment(config),600)
            except BaseException:r.write(control/'aggregate-error.json',dict(error=traceback.format_exc()),True)
            state('failed',error=failure)
    finally:
        # Development and full runs both release only this exact owned service.
        # A later full run requires a new profile and fresh numerical admission.
        if config:
            try:
                r.require(active is None or active.child.poll() is not None,
                    'Campaign child remains active; preserve service and block admission')
                if (output/'completion.json').exists():
                    r.require(r.read(output/'completion.json').get('all_children_drained') is True,
                        'Worker cleanup unconfirmed; preserve service and block admission')
                cleanup=r.cleanup_servers(config);r.write(output/'cleanup.json',cleanup,True)
                r.require(cleanup['all_owned_models_inactive'],'Owned model cleanup remains unconfirmed; admission blocked')
            except BaseException:
                cleanup_error=traceback.format_exc();r.write(control/'cleanup-error.json',dict(error=cleanup_error),True)
                failure=failure or cleanup_error;state('failed',error=failure)
        final=dict(schema='originx_gr00t_campaign_finished_v1',development=args.development,phase=phase,
            failed=failure is not None,rollout_returncode=rollout_returncode,unix=time.time(),elapsed_seconds=time.monotonic()-start)
        r.write(control/'finished.json',final,True)
        if (output/'config.json').exists():r.write(output/'campaign-finished.json',final,True)
        lock.close()
    return 1 if failure is not None or rollout_returncode not in (None,0) else 0


if __name__=='__main__':raise SystemExit(main())
