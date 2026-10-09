"""Three fixed engineering episodes through the unchanged production episode body.

No formal450 claims, score, retries, process termination, or model loading here.
Parent and each --one child use the simulator's CPU-only OSMesa environment.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace

from . import runner as r


def fixed_manifest():
    return dict(schema='lora_dev150_final_engineering_gate_v1',task='OpenOven',horizon=300,
        scope='Infrastructure only; independent engineering seeds; no dev450 assignment consumed',
        success_is_gate=False,max_episodes=3,max_control_steps=900,
        jobs=[dict(id=f'engineering-OpenOven-{arm}-{2040200900+i}',assignment_index=0,
            arm=arm,task='OpenOven',env_seed=2040200900+i,policy_seed=2040200900+i,
            split='pretrain',horizon=300) for i,arm in enumerate(r.ARMS)])


def inspect_manifest(path):
    value=r.read(path)
    r.require(value==fixed_manifest(),'Engineering manifest differs from fixed3 jobs')
    return value


def seed_audit(path,digest,manifest_sha):
    r.require(r.sha(path)==digest,'Engineering seed audit bytes changed')
    audit=r.read(path)
    r.require(audit.get('passed') is True and audit.get('manifest_sha256')==manifest_sha and
        audit.get('environment_inference_seed_range')==[2040200900,2040200902] and
        audit.get('collisions')==[] and bool(audit.get('scope')),'Engineering seeds need their own complete collision audit')


def source_proof():
    pins=r.read(r.HERE/'runner-source-pins.json')
    for name,digest in pins.items():r.require(r.sha(r.HERE/name)==digest,'Changed runner/helper: '+name)
    return dict(engineering_gate_sha256=r.sha(__file__),runner_source_pins_sha256=r.sha(r.HERE/'runner-source-pins.json'),
                runner_source_sha256=pins)


def one(args):
    owner=r.read(args.output/'owner.json');manifest=inspect_manifest(args.manifest)
    r.require(owner['bindings_sha256']==args.bindings_sha256==r.sha(args.bindings) and
        owner['gate_manifest_sha256']==r.sha(args.manifest) and owner['sources']==source_proof(),
        'Child launch binding/manifest/source differs')
    r.require(owner['seed_audit_sha256']==args.seed_audit_sha256 and
        owner['seed_audit_path']==str(args.seed_audit.resolve()),'Child seed audit authority differs')
    seed_audit(args.seed_audit,args.seed_audit_sha256,owner['gate_manifest_sha256'])
    job=next(job for job in manifest['jobs'] if job['arm']==args.one)
    r.require(r.read(args.output/'claims'/(job['id']+'.json'))['job']==job,'Unknown engineering claim')
    # Exclusive durable child entry prevents an accidental second --one call,
    # even if the original has not created its renderer or episode directory.
    r.new_json(args.output/('one-'+args.one+'-owner.json'),dict(r.identity(),job=job,engineering_only=True))
    episode_args=SimpleNamespace(bindings=args.bindings,bindings_sha=args.bindings_sha256,
        manifest=args.manifest,index=0,output=args.output)
    try:
        return r.episode(episode_args,engineering_job=job)
    except BaseException:
        r.new_json(args.output/'.sealed'/(job['id']+'-entry-error.json'),
            dict(status='infrastructure_unknown',job=job,traceback=traceback.format_exc(),no_retry=True))
        raise


def admit_terminal(args,job,returncode):
    path=args.output/'.sealed/episodes'/job['id']/'result.json'
    r.require(returncode==0 and path.is_file(),'Engineering child did not exit with a complete result')
    row=r.read(path)
    r.require(row.get('status')=='completed' and row.get('guard_passed') is True,'Engineering infrastructure is unknown')
    r.require(all(row.get(key)==job[key] for key in ('id','assignment_index','arm','task','env_seed','policy_seed')),
        'Engineering result identity differs')
    r.require(row.get('bindings_sha256')==args.bindings_sha256 and row.get('manifest_sha256')==r.sha(args.manifest),
        'Engineering result authority differs')
    proof=row['guard_close_proof'];guard_path=path.parent/'readback-guard/summary.json'
    r.require(Path(proof['path']).resolve()==guard_path.resolve() and r.sha(guard_path)==proof['sha256'] and
        proof.get('closed') is True and r.guard_complete(r.read(guard_path)),'Engineering readback guard did not close healthy')
    stats=row['stats'];ep=stats['episodes'][0]
    r.require(len(stats['episodes'])==1 and stats['horizon']==300 and 1<=ep['steps']<=300 and
        row['rng_after']['requests_since_reset']==(ep['steps']+15)//16,'Engineering horizon/query protocol differs')
    # Deliberately never inspect or expose success. A naturally completed
    # successful or unsuccessful task has identical infrastructure eligibility.
    return dict(arm=job['arm'],id=job['id'],status='completed',result_path=str(path),result_sha256=r.sha(path),
        guard_path=str(guard_path),guard_sha256=r.sha(guard_path),steps=ep['steps'],
        policy_queries=row['rng_after']['requests_since_reset'],success_used=False)


def execute(args):
    manifest=inspect_manifest(args.manifest);manifest_sha=r.sha(args.manifest)
    r.require(r.sha(args.bindings)==args.bindings_sha256,'Actual evaluation bindings changed')
    seed_audit(args.seed_audit,args.seed_audit_sha256,manifest_sha)
    sources=source_proof();binding=r.bindings(args.bindings,list(r.ARMS))
    r.empty_connections(binding,list(r.ARMS))
    args.output=args.output.resolve();root=Path(binding['root']).resolve()
    r.require(args.output.is_relative_to(root/'results/engineering') and
        not args.output.is_relative_to(Path(binding['cohort_root']).resolve()),'Use a new separate engineering output')
    r.require(not args.output.exists(),'Engineering output already exists; no retry or overwrite')
    owner=dict(r.identity(),output=str(args.output),bindings_path=str(args.bindings.resolve()),
        bindings_sha256=args.bindings_sha256,gate_manifest_path=str(args.manifest.resolve()),gate_manifest_sha256=manifest_sha,
        seed_audit_path=str(args.seed_audit.resolve()),seed_audit_sha256=args.seed_audit_sha256,
        sources=sources,engineering_only=True,success_used=False,formal450_claims_consumed=0,started_unix=time.time())
    # Persistent independent lease prevents the same gate being silently
    # rerun in another output. It never touches the formal per-arm leases.
    r.new_json(r.HERE/'engineering_gate.lease.json',owner)
    args.output.mkdir(parents=True,exist_ok=False,mode=0o700)
    for name in ('claims','processes','.sealed/logs','sessions'):(args.output/name).mkdir(parents=True,mode=0o700)
    r.new_json(args.output/'owner.json',owner)
    r.new_json(args.output/'gate_manifest.json',manifest)
    r.new_json(args.output/'provenance.json',owner)
    stop=[];active=[];records=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.append('parent_signal_natural_drain'))
    def persist():
        r.save(args.output/'status.json',dict(planned=3,terminal=len(records),active=len(active),
            completed=sum(row['status']=='completed' for row in records),unknown=sum(row['status']!='completed' for row in records),
            engineering_only=True,score_emitted=False,success_used=False,stop_reasons=stop))
    try:
        for job in manifest['jobs']:
            if stop:
                records.append(dict(arm=job['arm'],id=job['id'],status='infrastructure_unknown',reason='not_started_after_signal'));continue
            command=[binding['python'],'-u','-m','lora_dev150_final_v1.engineering_gate','--one',job['arm'],
                '--bindings',str(args.bindings.resolve()),'--bindings-sha256',args.bindings_sha256,
                '--manifest',str(args.manifest.resolve()),'--seed-audit',str(args.seed_audit.resolve()),
                '--seed-audit-sha256',args.seed_audit_sha256,'--output',str(args.output)]
            r.new_json(args.output/'claims'/(job['id']+'.json'),dict(job=job,bindings_sha256=args.bindings_sha256,
                gate_manifest_sha256=manifest_sha,parent=owner,command=command,engineering_only=True))
            log=(args.output/'.sealed/logs'/(job['id']+'.log')).open('x')
            try:
                child=subprocess.Popen(command,env=r.environment(binding),cwd=binding['root'],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:
                log.close();records.append(dict(arm=job['arm'],id=job['id'],status='infrastructure_unknown',reason='spawn_failed',traceback=traceback.format_exc()))
                stop.append('spawn_failed');continue
            active.append(dict(child=child,job=job,log=log))
            process=dict(pid=child.pid,process_start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]),
                command=command,cwd=binding['root'],job=job)
            r.new_json(args.output/'processes'/(job['id']+'.json'),process)
        persist()
        while active:
            for item in list(active):
                code=item['child'].poll()
                if code is None:continue
                item['log'].close();job=item['job']
                try:row=admit_terminal(args,job,code)
                except BaseException:
                    row=dict(arm=job['arm'],id=job['id'],status='infrastructure_unknown',returncode=code,traceback=traceback.format_exc(),no_retry=True)
                records.append(row);active.remove(item);persist()
            if active:time.sleep(.2)
        passed=len(records)==3 and all(row['status']=='completed' for row in records)
        report=dict(passed=passed,engineering_only=True,scope='Infrastructure only; no task-performance threshold',
            success_used=False,score_emitted=False,formal450_claims_consumed=0,no_retry=True,
            owner_sha256=r.sha(args.output/'owner.json'),bindings_sha256=args.bindings_sha256,gate_manifest_sha256=manifest_sha,
            seed_audit_sha256=args.seed_audit_sha256,sources=sources,records=records,completed_unix=time.time())
        r.new_json(args.output/'report.json',report)
        return 0 if passed else 1
    except BaseException:
        r.new_json(args.output/'supervisor-error.json',dict(status='infrastructure_unknown',traceback=traceback.format_exc(),
            healthy_children_left_running=[item['child'].pid for item in active if item['child'].poll() is None],no_retry=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',type=Path)
    parser.add_argument('--one',choices=r.ARMS)
    for name in ('bindings','manifest','output','seed-audit'):parser.add_argument('--'+name,type=Path)
    for name in ('bindings-sha256','seed-audit-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.write_manifest:
        r.new_json(args.write_manifest,fixed_manifest())
        print(json.dumps(dict(manifest=str(args.write_manifest),sha256=r.sha(args.write_manifest),GPU_or_sim_started=False)));return 0
    r.require(all(getattr(args,key) is not None for key in ('bindings','bindings_sha256','manifest','output','seed_audit','seed_audit_sha256')),
        'Bindings, fixed manifest, independent seed audit and new output must all be explicit')
    return one(args) if args.one else execute(args)


if __name__=='__main__':raise SystemExit(main())
