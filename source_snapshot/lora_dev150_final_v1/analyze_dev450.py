"""CPU-only, all-or-nothing analysis of the fixed completed base/L/A450 run.

No model import, RPC, simulator, retry, source mutation or partial-score output.
Public completeness and provenance pass before sealed JSON is decoded. Sealed
integrity passes for every assignment before any success count is accumulated.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import uuid

HERE = Path(__file__).resolve().parent
MANIFEST_SHA = '8643051dab874f68ca8763242265ad3016788c4da62af3f57060004fe5144fe1'
BASE_SHA = '21f3a9ddf040e8afd1e56268ee76263df61692bd1c568f76ae9a7f905e065816'
ARMS = ('base', 'L', 'A')
PUBLIC_KEYS = ('id','assignment_index','arm','task','env_seed','policy_seed','status',
               'bindings_sha256','manifest_sha256','endpoint_key','wall_seconds')
HELLO_KEYS = ('server_instance','protocol','exclusive_connection','shared_model','isolated_rng_stream',
    'serial_forward','rng_isolation','model_path','model_config_sha256','model_assets_sha256',
    'official_server_sha256','multiplex_sources_sha256','model_config_runtime_sha256',
    'model_execution_thread_name','model_execution_thread_ident','torch_version',
    'cuda_visible_device_count','full_request_only','inference_optimization')
EVENTS = ('reset_requested','reset_completed','initial_fingerprints_recorded',
          'assignment_selected','server_connected','rng_ack','first_infer')


class Refusal(Exception):
    """Only fixed validation labels and planned assignment IDs enter error output."""
    def __init__(self, code, assignment=None):
        self.code, self.assignment = code, assignment
        super().__init__(code)


def need(value, code, assignment=None):
    if not value: raise Refusal(code, assignment)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Evidence:
    def __init__(self): self.files = {}
    def pin(self, path, expected=None):
        p = Path(path).resolve(strict=True);digest = sha(p)
        need(expected is None or digest == expected, 'artifact_sha_changed')
        need(str(p) not in self.files or self.files[str(p)] == digest, 'artifact_changed_during_analysis')
        self.files[str(p)] = digest
        return p
    def read(self, path, expected=None):
        p = self.pin(path, expected)
        with p.open() as stream: return json.load(stream)
    def final_check(self):
        for path, digest in self.files.items(): need(sha(path) == digest, 'artifact_changed_before_publication')


def exited(owner):
    need(type(owner.get('pid')) is int and owner['pid'] > 1 and
         type(owner.get('process_start_ticks')) is int, 'invalid_process_identity')
    path = Path('/proc')/str(owner['pid'])/'stat'
    try: fields = path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError: return True
    return fields[0] == 'Z' or int(fields[19]) != owner['process_start_ticks']


def process_fields(owner):
    return {k:owner[k] for k in ('pid','process_start_ticks','command','cwd')}


def guarded(g, guard_source_sha):
    return (g.get('status') == 'closed' and g.get('failure') is None and
        set(g.get('gl_error_counts',{})) <= {'after_draw:1281'} and
        g.get('raw_frames_checked',0) > 0 and g.get('sentinel_checks',0) > 0 and
        g.get('source_sha256') == guard_source_sha and g.get('backend') == 'osmesa' and
        g.get('infrastructure_guard') is True)


def hello_matches(actual, expected, arm):
    return (all(actual.get(k) == expected.get(k) for k in HELLO_KEYS) and
            actual.get('lora_serving_identity') == expected.get('lora_serving_identity') and
            (arm == 'base' or actual.get('loader_kind') == expected.get('loader_kind')))


def wilson(successes, n):
    # Per-arm descriptive binomial intervals; not an arm-difference test/gate.
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.0,center-half),min(1.0,center+half)]


def endpoint_proofs(b, evidence, root):
    proofs={};loader=None;loaded_bindings={}
    for arm in ARMS:
        ids=b['arm_endpoints'][arm]
        need(1 <= len(ids) <= 8 and len(ids)==len(set(ids)), 'endpoint_mapping_invalid')
        for key in ids:
            need(key not in proofs, 'endpoint_shared_between_arms')
            model=b['models'][key];need(model['arm']==arm, 'endpoint_arm_mismatch')
            server=evidence.read(model['server_manifest'],model['server_manifest_sha256'])
            parity=evidence.read(model['parity_report'],model['parity_sha256'])
            h=server['hello'];need(server.get('ready') is True and len(server['servers'])==1, 'endpoint_not_admitted')
            need(canonical(h['model_assets_sha256'])==model['checkpoint_sha256']==BASE_SHA and
                 h['model_path']==str(Path(model['checkpoint_path']).resolve()), 'released_base_identity_changed')
            need(h.get('exclusive_connection') is False and h.get('isolated_rng_stream') is True and
                 h.get('serial_forward') is True and h.get('inference_optimization')=='none' and
                 h.get('cuda_visible_device_count')==1 and h.get('server_instance'), 'endpoint_protocol_changed')
            need(server['parity_sha256']==model['parity_sha256'] and parity.get('passed') is True and
                 len(parity.get('trace',[]))==6 and parity.get('ambient_rng_unchanged') is True and
                 all(t.get('full_action_exact') is True and t.get('all_rng_exact') is True for t in parity['trace']),
                 'endpoint_parity_invalid')
            need(parity['source_sha256']==server['source_sha256']==h['multiplex_sources_sha256'] and
                 parity['model']['model_assets_sha256']==h['model_assets_sha256'] and
                 parity['model']['official_server_sha256']==h['official_server_sha256'] and
                 parity['trace'][0]['actual_rng']['model_config_runtime_sha256']==h['model_config_runtime_sha256'],
                 'endpoint_parity_identity_changed')
            for name,digest in server['source_sha256'].items():
                need(Path(name).name==name, 'transport_source_path_invalid')
                evidence.pin(root/'multiplex_inference'/name,digest)
            if arm=='base': need(not h.get('lora_serving_identity'), 'base_contains_adapter')
            else:
                if loader is None:
                    spec=importlib.util.spec_from_file_location('_dev450_cpu_loader',HERE/'loader.py')
                    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
                # read_binding is explicitly CPU/file-only and verifies complete
                # final1613/80000 lineage/adapter/source pins, never loads torch.
                binding_key=(model['endpoint_binding'],model['endpoint_binding_sha256'])
                if binding_key not in loaded_bindings:
                    loaded_bindings[binding_key]=loader.read_binding(*binding_key)
                ep=loaded_bindings[binding_key]
                evidence.pin(model['endpoint_binding'],model['endpoint_binding_sha256'])
                for item in ep.values():
                    if isinstance(item,dict) and item.get('path') and item.get('sha256'):
                        evidence.pin(item['path'],item['sha256'])
                for path,digest in ep['source_sha256'].items(): evidence.pin(path,digest)
                ident=h['lora_serving_identity']
                need(ident['arm']==arm and ident['binding_sha256']==model['endpoint_binding_sha256'] and
                     ident['adapter_file_sha256']==ep['adapter']['sha256'] and
                     ident['endpoint_updates']==1613 and ident['endpoint_samples']==80000 and
                     ident['inference_arithmetic']==ep['inference_arithmetic'] and
                     ident==parity['model']['lora_serving_identity'], 'adapter_or_arm_identity_changed')
            proofs[key]={'model':model,'hello':h}
    need(set(proofs)==set(b['models']), 'unmapped_extra_endpoint')
    return proofs


def analyze(args, state, evidence):
    root=args.root.resolve(strict=True);run=args.run_output.resolve(strict=True)
    manifest=evidence.read(args.manifest,MANIFEST_SHA);jobs=manifest['jobs'];tasks=manifest['tasks']
    strata={t['task']:t['stratum'] for t in tasks};task_names=set(strata)
    need(len(tasks)==30 and len(task_names)==30 and Counter(strata.values())=={'atomic':10,'composite':20}, 'task_strata_changed')
    need(len(jobs)==450 and manifest['dispatch_order']==list(range(450)) and
         len({j['id'] for j in jobs})==450 and Counter(j['arm'] for j in jobs)==dict.fromkeys(ARMS,150), 'fixed_schedule_changed')
    need(Counter((j['arm'],j['task']) for j in jobs)=={(a,t):5 for a in ARMS for t in task_names}, 'task_denominators_changed')
    need(all(j['assignment_index']==i and j['split']=='pretrain' and
        j['env_seed']==j['policy_seed']==2040200000+i for i,j in enumerate(jobs)), 'assignment_or_seeds_changed')
    b=evidence.read(args.bindings);binding_sha=sha(args.bindings)
    need(b['schema']=='lora_dev150_final_bindings_v1' and b.get('approved_for_evaluation') is True and
         Path(b['root']).resolve()==root and run.parent==Path(b['cohort_root']).resolve(), 'run_binding_scope_changed')
    decision=evidence.read(args.decision_protocol,b['decision_protocol']['sha256'])
    need(Path(b['decision_protocol']['path']).resolve()==args.decision_protocol.resolve() and
         decision['kind']=='final1613_dev150_decision_v1' and decision.get('frozen_before_new_outcomes') is True and
         decision['manifest_sha256']==MANIFEST_SHA and decision['primary_candidate']=='A' and
         decision['primary_dev_gate']=={'net_success_gain_over_base_at_least':8,
         'composite_success_count_difference_at_least':0,'all_450_completed':True,'infrastructure_unknowns':0},
         'prospective_A_only_decision_changed')
    pins=evidence.read(HERE/'runner-source-pins.json',b['runner_pins_sha256'])
    need({'runner.py','loader.py','client.py','server.py','__init__.py','candidate.py','readback_guard.py',
          'context_lifetime.py','dev_randomized_broad_alpha.py','dev_repeat_audit.py','dev_paired.py','dev_extend.py'} <= set(pins), 'source_inventory_incomplete')
    for name,digest in pins.items():
        need(Path(name).name==name, 'unsafe_runner_source_path');evidence.pin(HERE/name,digest)
    for path,digest in b['source_sha256'].items(): evidence.pin(path,digest)
    seed=evidence.read(b['seed_audit']['path'],b['seed_audit']['sha256'])
    need(seed.get('passed') is True and seed['manifest_sha256']==MANIFEST_SHA and
         seed['environment_inference_seed_range']==[2040200000,2040200449] and
         seed.get('collisions')==[] and not seed.get('errors') and seed.get('scope'), 'seed_audit_invalid')

    state['phase']='public_completeness_and_provenance'
    launch=evidence.read(run/'launch.json');summary=evidence.read(run/'summary.json')
    completion=evidence.read(run/'completion.json');public=evidence.read(run/'records.json')
    need(summary=={'complete':True,'planned':450,'attempted':450,'completed':450,'infrastructure_unknown':0,
        'not_attempted':0,'per_arm':{a:{'planned':150,'completed':150} for a in ARMS},
        'score_emitted':False,'outcomes_sealed':True,'no_replacement':True}, 'public_run_incomplete_or_unknown')
    need(completion.get('selected_assignments_complete') is True and completion.get('active_children')==0 and
         completion.get('score_emitted') is False and completion.get('dispatch_order')==list(range(450)) and
         completion.get('stop_reasons')==[], 'terminal_completion_incomplete')
    # This runner has summary/completion instead of a separate status.json.
    if (run/'status.json').exists():
        status=evidence.read(run/'status.json')
        need(status.get('stage',status.get('status'))=='complete', 'additional_status_not_complete')
    authority={'output':str(run),'bindings_sha256':binding_sha,'manifest_sha256':MANIFEST_SHA,
               'source_pins_sha256':sha(HERE/'runner-source-pins.json'),'dispatch_order':list(range(450))}
    need(all(launch.get(k)==v for k,v in authority.items()) and set(launch['arms'])==set(ARMS) and
         launch['source_sha256']==pins and launch['models']==b['models'], 'launch_authority_changed')
    need(exited(launch), 'runner_still_active')
    sessions=[]
    for path in sorted((run/'sessions').glob('*.json')):
        session=evidence.read(path)
        need(all(session.get(k)==v for k,v in authority.items()) and session['arms']==launch['arms'] and
             exited(session), 'session_authority_or_terminal_state_changed')
        sessions.append(session)
    need(sessions, 'missing_execution_session')
    session_ids={canonical(process_fields(s)) for s in sessions}
    for arm in ARMS:
        lease=evidence.read(run.parent/(arm+'.lease.json'))
        need(lease.get('arm')==arm and all(lease.get(k)==v for k,v in authority.items()) and
             lease['arms']==launch['arms'], 'arm_lease_changed')
    need(len(public)==450 and len({r['id'] for r in public})==450 and
         {r['id'] for r in public}=={j['id'] for j in jobs} and
         all(r.get('status')=='completed' for r in public), 'public_assignment_incomplete_or_unknown')
    records={r['id']:r for r in public};expected_ids={j['id'] for j in jobs}
    need({p.stem for p in (run/'claims').glob('*.json')}==expected_ids and
         {p.stem for p in (run/'processes').glob('*.json')}==expected_ids and
         {p.name for p in (run/'.sealed/episodes').iterdir()}==expected_ids, 'claim_process_or_episode_inventory_changed')
    proofs=endpoint_proofs(b,evidence,root);ordinals=Counter();prepared=[];process_ids=set()
    for job in jobs:
        jid=job['id'];record=records[jid];arm=job['arm'];ids=b['arm_endpoints'][arm]
        endpoint=ids[ordinals[arm]%len(ids)];ordinals[arm]+=1
        need(all(record.get(k)==job[k] for k in ('id','assignment_index','arm','task','env_seed','policy_seed')) and
             record.get('bindings_sha256')==binding_sha and record.get('manifest_sha256')==MANIFEST_SHA and
             record.get('endpoint_key')==endpoint, 'public_record_authority_changed',jid)
        claim=evidence.read(run/'claims'/(jid+'.json'));process=evidence.read(run/'processes'/(jid+'.json'))
        need(claim['job']==job and claim['bindings_sha256']==binding_sha and
             all(claim['parent'].get(k)==v for k,v in authority.items()) and
             canonical(process_fields(claim['parent'])) in session_ids, 'claim_or_parent_identity_changed',jid)
        command=[b['python'],'-u',str(HERE/'runner.py'),'episode','--bindings',str(args.bindings.resolve()),
            '--bindings-sha',binding_sha,'--manifest',str(args.manifest.resolve()),'--output',str(run),
            '--index',str(job['assignment_index'])]
        need(claim['command']==command and process['command']==command and process['cwd']==str(root) and
             process['id']==jid and process['index']==job['assignment_index'] and exited(process),
             'episode_process_not_terminal_or_wrong_identity',jid)
        process_id=(process['pid'],process['process_start_ticks'])
        need(process_id not in process_ids, 'duplicate_episode_process_identity',jid);process_ids.add(process_id)
        directory=run/'.sealed/episodes'/jid;owner=evidence.read(directory/'episode-owner.json')
        need(process_fields(owner)==process_fields(process) and exited(owner), 'child_owner_mismatch',jid)
        path=directory/'result.json'
        need(Path(record['sealed_result_path']).resolve()==path.resolve(), 'sealed_result_path_changed',jid)
        evidence.pin(path,record['sealed_result_sha256'])  # Bytes only; no outcome JSON decode yet.
        guard_path=directory/'readback-guard/summary.json';guard=evidence.read(guard_path)
        need(guarded(guard,pins['readback_guard.py']), 'renderer_guard_not_healthy_closed',jid)
        prepared.append((job,record,path,guard_path,endpoint))
    state['public_completeness_verified']=True
    state['phase']='all_sealed_integrity_before_counting';state['sealed_json_decoding_started']=True
    validated=[]
    for job,public,path,guard_path,endpoint in prepared:
        jid=job['id'];arm=job['arm'];row=evidence.read(path,public['sealed_result_sha256'])
        regenerated={k:row[k] for k in PUBLIC_KEYS if k in row}
        regenerated.update(sealed_result_path=str(path),sealed_result_sha256=public['sealed_result_sha256'])
        need(regenerated==public and row['status']=='completed' and row.get('guard_passed') is True,
             'sealed_record_disagrees_with_public',jid)
        proof=row['guard_close_proof']
        need(Path(proof['path']).resolve()==guard_path.resolve() and proof.get('closed') is True and
             proof['sha256']==evidence.files[str(guard_path.resolve())], 'guard_closure_receipt_changed',jid)
        expected=proofs[endpoint]['hello'];hello=row['server_identity'];ack=row['rng_ack'];after=row['rng_after']
        need(hello_matches(hello,expected,arm) and hello_matches(ack,expected,arm) and
             hello_matches(after,expected,arm), 'RPC_arm_adapter_or_server_identity_changed',jid)
        need(hello.get('connection_id') is not None and all(x.get('connection_id')==hello['connection_id'] for x in (ack,after)) and
             ack.get('seed')==after.get('seed')==job['policy_seed'] and ack.get('requests_since_reset')==0 and
             all(x.get('torch_cpu_rng_sha256') and x.get('torch_cuda_rng_sha256') for x in (ack,after)),
             'RPC_rng_connection_or_seed_changed',jid)
        events=row['events'];need(tuple(e['event'] for e in events)==EVENTS and
             [e['sequence'] for e in events]==list(range(len(EVENTS))) and
             events[3]['model']==events[4]['model']==arm and events[4]['hello']==hello and
             events[5]['acknowledgment']==ack, 'natural_reset_assignment_or_RPC_event_order_changed',jid)
        stats=row['stats'];need(len(stats['episodes'])==1 and stats['horizon']==job['horizon'], 'official_trial_shape_changed',jid)
        ep=stats['episodes'][0]
        need(stats.get('env_name')==job['task'] and stats.get('split')=='pretrain' and
             stats.get('num_episodes')==1, 'official_task_or_split_changed',jid)
        need(ep['seed']==job['env_seed'] and ep['episode']==ep['global_episode_index']==0 and
             type(ep['steps']) is int and 1<=ep['steps']<=job['horizon'] and
             after['requests_since_reset']==(ep['steps']+15)//16, 'official_horizon_or_query_count_changed',jid)
        need(row['context_lifetime']['source_sha256']==pins['context_lifetime.py'], 'context_fix_source_changed',jid)
        need(set(row['local_rng']['before'])=={'python','numpy','torch_cpu'} and
             row['local_rng']['before']==row['local_rng']['after'], 'local_rng_restore_changed',jid)
        # Outcome values are validated here but never accumulated/exposed until
        # EVERY assignment's integrity checks have succeeded.
        need(type(row.get('success')) is bool and type(ep.get('success')) is bool and
             row['success']==ep['success'] and stats.get('successes')==int(ep['success']) and
             stats.get('success_rate')==float(ep['success']) and (ep['success'] or ep['steps']==job['horizon']),
             'official_terminal_outcome_invalid',jid)
        validated.append((job,row))
    evidence.pin(__file__);evidence.final_check()
    state['phase']='all_integrity_verified_aggregate';state['all450_integrity_verified']=True
    counts={arm:{group:{'successes':0,'episodes':0} for group in ('total','atomic','composite')} for arm in ARMS}
    per_task={t:{a:{'successes':0,'episodes':0} for a in ARMS} for t in sorted(task_names)}
    for job,row in validated:
        arm=job['arm'];task=job['task'];success=int(row['success'])
        for group in ('total',strata[task]):counts[arm][group]['episodes']+=1;counts[arm][group]['successes']+=success
        per_task[task][arm]['episodes']+=1;per_task[task][arm]['successes']+=success
    for arm in ARMS:
        need([counts[arm][k]['episodes'] for k in ('total','atomic','composite')]==[150,50,100], 'aggregate_denominators_invalid')
        for group,row in counts[arm].items():
            row.update(success_rate=row['successes']/row['episodes'],wilson95_descriptive=wilson(row['successes'],row['episodes']))
    differences={a:{group:{'net_successes':counts[a][group]['successes']-counts['base'][group]['successes'],
        'absolute_percentage_points':100*(counts[a][group]['success_rate']-counts['base'][group]['success_rate'])}
        for group in ('total','atomic','composite')} for a in ('L','A')}
    gate={'primary_candidate':'A','diagnostic_only_arm':'L','all450_complete_zero_unknown':True,
          'net_success_gain_at_least8':differences['A']['total']['net_successes']>=8,
          'composite_difference_at_least0':differences['A']['composite']['net_successes']>=0}
    gate['eligible_for_independent_confirm600']=gate['net_success_gain_at_least8'] and gate['composite_difference_at_least0']
    report={'kind':'fixed450_final1613_development_analysis_v1','status':'complete','observed_unix':time.time(),
        'scope':'30 fixed seen-pretrain tasks, independent arm resets; development results only',
        'formal_leaderboard_score':None,'historical_baseline_comparison':None,'paired_inference_used':False,
        'counts':counts,'per_task':per_task,'differences_vs_contemporary_base':differences,'A_promotion_gate':gate,
        'next_step':decision['next_if_pass'] if gate['eligible_for_independent_confirm600'] else decision['next_if_not_pass'],
        'interval_note':'Wilson intervals are per-arm descriptive binomial intervals under independent episode assumptions; fixed task heterogeneity and one training seed limit interpretation. They are not paired intervals, arm-difference significance tests or promotion gates.',
        'official2500_started_or_authorized_by_report':False,
        'completion_evidence':'450 terminal public records; pinned runner accepts completed children only after zero exit; recorded original PID/start identities no longer active. Separate exit-code receipts are not emitted by this runner.',
        'analysis_source_sha256':sha(__file__),'manifest_sha256':MANIFEST_SHA,'bindings_sha256':binding_sha,
        'decision_protocol_sha256':sha(args.decision_protocol),'verified_files':[{'path':p,'sha256':s} for p,s in sorted(evidence.files.items())]}
    lines=['Fixed dev150 per arm: 450 completed, 0 unknown. Independent resets; no leaderboard score.']
    for arm in ARMS:
        c=counts[arm];lines.append(f"{arm}: {c['total']['successes']}/150 ({100*c['total']['success_rate']:.2f}%); atomic {c['atomic']['successes']}/50; composite {c['composite']['successes']}/100.")
    for arm in ('L','A'):
        d=differences[arm]['total'];lines.append(f"{arm} minus contemporary base: {d['net_successes']:+d} successes, {d['absolute_percentage_points']:+.2f} percentage points.")
    lines.append('A eligible for independent confirm600 per arm: '+str(gate['eligible_for_independent_confirm600'])+'. L remains diagnostic only.')
    lines.append(report['next_step']);lines.append('No paired statistical claim; no comparison against historical57.92%; no formal2500 score.')
    return report,'\n'.join(lines)+'\n'


def publish(output, report, text):
    output=output.resolve();need(not output.exists(),'analysis_output_already_exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.parent/('.'+output.name+'.tmp-'+uuid.uuid4().hex);staging.mkdir(mode=0o700)
    with (staging/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    with (staging/'report.txt').open('x') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    need(not output.exists(),'analysis_output_appeared_during_publication');staging.rename(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','run-output','manifest','bindings','decision-protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();state={'phase':'input_validation','public_completeness_verified':False,
        'sealed_json_decoding_started':False,'all450_integrity_verified':False};evidence=Evidence()
    need(not args.output.exists(),'analysis_output_already_exists')
    try:
        report,text=analyze(args,state,evidence)
    except Exception as error:
        # No raw traceback or data values: even a late integrity error cannot
        # expose successes from an otherwise partially validated cohort.
        report={'kind':'fixed450_analysis_refused_v1','status':'validation_refused','state':state,
            'error_code':error.code if isinstance(error,Refusal) else 'unexpected_'+type(error).__name__,
            'assignment':error.assignment if isinstance(error,Refusal) else None,
            'partial_success_counts_emitted':False,'promotion_decision_emitted':False,
            'analysis_source_sha256':sha(__file__)}
        text='Analysis refused: '+report['error_code']+'. No partial success counts or promotion decision emitted.\n'
        publish(args.output,report,text);print(json.dumps({'status':'validation_refused','output':str(args.output),'error_code':report['error_code']}));return 2
    publish(args.output,report,text)
    print(json.dumps({'status':'complete','output':str(args.output),'report_sha256':sha(args.output/'report.json')}));return 0


if __name__=='__main__':raise SystemExit(main())
