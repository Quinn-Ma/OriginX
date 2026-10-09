"""Join three preserved batches; each previously claimed arm is used exactly once.

The original and failed multigpu-v1 outcomes remain unchanged. Only future v2
receipts can use the independently pinned strict transport-recovery adapter.
The fixed cohort is still the same 50 minimum-seed cases and 350 planned arms.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
OLD_NAME='originx-confirmatory-20261009-v1'
V1_NAME='originx-confirmatory-multigpu-20261009-v1'
NEW_NAME='originx-confirmatory-multigpu-20261009-v2'
JOIN_V1_SHA='e5ef204cd61f19726d26afbb4a1424fe876f27d85dac15fc775a7a61d31f6151'
CONTINUATION_SHA='bdd8e17ac5bb62cdf61264843ce06d2453476a77b8d77aa33cea2e6d69dd2d9d'
TRANSPORT_SHA='c4245b40033e128ee3398f5ffcbc4d6e7a0e81448330f596eddcb2c32f651541'


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def pinned_module(path,name,digest):
    require(sha(path)==digest,'Analysis dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def merge_three(join,manifest,selection1,selection2,records,claims):
    """Claim precedence is irrevocable, irrespective of success or unknown."""
    old,v1,new=claims
    require(not set(old).intersection(v1) and not set(old).intersection(new) and not set(v1).intersection(new),
        'Repeated assignment across the three batches')
    combined1,_,_,_=join.merge_records(manifest,selection1,records[0],records[1],old,v1)
    inherited=dict(old);inherited.update(v1)
    combined,reduced,retained,provenance=join.merge_records(manifest,selection2,combined1,records[2],inherited,new)
    for row in provenance:
        key=row['episode_id']
        row['source_batch']=OLD_NAME if key in old else V1_NAME if key in v1 else NEW_NAME
        row['claimed']=key in old or key in v1 or key in new
    return combined,reduced,retained,provenance


def run(args):
    sys.path.insert(0,str(ROOT))
    from originx_confirmatory_20261009 import aggregate,analysis
    join=pinned_module(ROOT/'aggregate_reduced_v1.py','originx_verified_join_v1',JOIN_V1_SHA)
    continuation=pinned_module(ROOT/'continue_multigpu_v2.py','originx_verified_continuation_v2',CONTINUATION_SHA)
    transport=pinned_module(ROOT/'aggregate_transport_v2.py','originx_verified_transport_v2',TRANSPORT_SHA)
    aggregate=transport.install(aggregate)
    outputs=[ROOT/'results'/name for name in (OLD_NAME,V1_NAME,NEW_NAME)]
    destination=ROOT/'results/originx-three-hour-amendment-20261009-v1/analysis-reduced-v2'
    require(not destination.exists(),'Three-batch final join already exists; never overwrite')
    configs=[read(output/'config.json') for output in outputs]
    manifest=read(configs[0]['manifest']);manifest_bytes=Path(configs[0]['manifest']).read_bytes()
    jobs={j['id']:j for j in manifest['jobs']}
    for output,config in zip(outputs,configs):
        require(Path(config['output'])==output and Path(config['manifest']).read_bytes()==manifest_bytes
            and sha(config['manifest'])==config['manifest_sha256']==configs[0]['manifest_sha256'],
            'Batch output or full original manifest differs')
    selections=[]
    for config in configs[1:]:
        reference=config['continuation_selection']
        continuation.checked_file(reference['path'],reference['sha256'],ROOT)
        selection=read(reference['path']);continuation.validate_selection(selection,manifest)
        selections.append(selection)
    cohort=continuation.reduced_cohort(manifest)
    for selection,config in zip(selections,configs[1:]):
        require(selection['reduced_cohort']==cohort and config['fixed_reduced_cohort']==cohort,
            '50-case cohort changed between continuations')
        for path,digest in selection['original_claims_sha256'].items():
            require(sha(path)==digest,'Prior durable claim changed')
    require(selections[1].get('prior_batch_count')==2
        and selections[1].get('multigpu_v1_config_sha256')==sha(outputs[1]/'config.json')
        and selections[1]['original_config_sha256']==sha(outputs[0]/'config.json'),
        'v2 selection does not bind both actual prior configurations')
    claims=[join.claimed(output,sha(output/'config.json'),jobs) for output in outputs]
    expected=continuation.combine_claim_selections(manifest,claims[0],sha(outputs[0]/'config.json'),
        claims[1],sha(outputs[1]/'config.json'))
    for key in ('selected_ids','excluded_claimed_ids','target_case_ids','original_claimed_ids','multigpu_v1_claimed_ids'):
        require(selections[1].get(key)==expected[key],'v2 prior-claim partition differs: '+key)
    pins={};proofs=[];records=[];usage_rows=[]
    for label,output,config,batch_claims in zip(('old','v1','new'),outputs,configs,claims):
        pins.update(join.verify_terminal_batch(output,config,continuation))
        cleanup=read(output/'cleanup.json')
        require(cleanup.get('all_owned_services_stopped') is True and cleanup.get('still_active_owned_pids')==[],
            'Model cleanup receipt not complete')
        pins[str(output/'cleanup.json')]=sha(output/'cleanup.json')
        state=getattr(args,label+'_broker_state');terminal=getattr(args,label+'_broker_terminal')
        proofs.append(join.verify_sealed_broker(terminal,getattr(args,label+'_broker_terminal_sha256'),state,
            output,sha(output/'config.json'),config['manifest_sha256'],continuation))
        analysis_dir=output/'analysis-final'
        if label=='old':require(analysis_dir.exists(),'Original final normalization must be preserved')
        if not analysis_dir.exists():
            aggregate.aggregate(output/'config.json',broker_state=state,output_directory=analysis_dir,
                bootstrap_samples=args.bootstrap_samples)
        records.append(join.verified_normalization(analysis_dir,output/'config.json',manifest,batch_claims,aggregate))
        usage_rows.append(aggregate.collect_usage(output,{j['request_id']:j for j in manifest['jobs']},
            sha(output/'config.json'),config['manifest_sha256'],state))
        for path in analysis_dir.glob('*.json'):pins[str(path)]=sha(path)
        pins[str(output/'config.json')]=sha(output/'config.json')
    combined,reduced,retained,provenance=merge_three(join,manifest,selections[0],selections[1],records,claims)
    normalized_manifests=[read(output/'analysis-final/normalized_manifest.json') for output in outputs]
    require(normalized_manifests[0]==normalized_manifests[1]==normalized_manifests[2],
        'Normalized assignment authority differs')
    target=set(cohort['case_ids']);reduced_manifest=[r for r in normalized_manifests[0] if r['case_id'] in target]
    report=analysis.analyze(reduced_manifest,[r for r in reduced if r['record_present']],N=50,
        bootstrap_samples=args.bootstrap_samples)
    usage=join.merge_usage(usage_rows)
    report.update(schema='originx_user_amended_50case_analysis_v2',scored_confirmation=False,
        original_2500_case_confirmation_complete=False,user_reduced_cohort=cohort,
        original_registered_unique_cases=2500,original_registered_arm_outcomes=17500,
        amended_unique_cases=50,amended_arm_outcomes=350,authority_valid=True,prior_batches_preserved=3,
        historical_1496_of_2500_and_56_of_1004_unchanged=True,
        outside_cohort_prior_arm_records_retained=len(retained),
        outside_cohort_evidence_excluded_from_amended_estimand_not_deleted=True,
        normalized_amended_valid_arm_records=sum(r['valid'] for r in reduced),
        amended_claimed_arm_outcomes=len(set().union(*[set(c) for c in claims]).intersection(r['episode_id'] for r in reduced)),
        claimed_arms_by_batch={name:len(c) for name,c in zip((OLD_NAME,V1_NAME,NEW_NAME),claims)},
        cli_usage={key:value for key,value in usage.items() if key!='calls'},
        transport_recovery_scope='Strict completed reconnect receipts only in v2; original and failed v1 receipts/outcomes unchanged',
        interpretation='Post-start user schedule amendment and disclosed infrastructure continuation. Fixed 50-case experiment; all earlier claims retained and never replayed. Not the original 2500-case confirmation or a new benchmark score.',
        no_official_rank_claim=True,generated_unix=time.time())
    for config in configs[1:]:
        path=Path(config['continuation_selection']['path']);pins[str(path)]=sha(path)
    for path in (Path(__file__).resolve(),ROOT/'continue_multigpu_v2.py',ROOT/'aggregate_reduced_v1.py',ROOT/'aggregate_transport_v2.py'):
        pins[str(path)]=sha(path)
    destination.mkdir(parents=True)
    for name,value in [('normalized_manifest',reduced_manifest),('normalized_records',reduced),
        ('all_original_assignment_records',combined),('outside_cohort_prior_records',retained),
        ('record_provenance',provenance),('cli_usage',usage),('report',report)]:
        join.write_new(destination/(name+'.json'),value)
    join.write_new(destination/'authority.json',dict(schema='originx_amendment_three_batch_join_authority_v2',
        passed=True,evidence_sha256=pins,broker_terminal_proofs=proofs,claims_disjoint=True,
        fixed_cohort_count=50,planned_arm_outcomes=350,full_manifest_identical=True,
        every_claimed_raw_record_renormalized=True,no_models_or_cli_invoked=True,
        original_and_v1_claims_never_replayed=True))
    return dict(output=str(destination),N=50,planned_arm_outcomes=350,
        valid_arm_records=report['normalized_amended_valid_arm_records'],
        cli_invocations=usage['unique_observed_cli_invocations'],
        complete_input_output_token_accounting=usage['complete_input_output_token_accounting'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old','v1','new'):
        parser.add_argument('--'+name+'-broker-state',type=Path,required=True)
        parser.add_argument('--'+name+'-broker-terminal',type=Path,required=True)
        parser.add_argument('--'+name+'-broker-terminal-sha256',required=True)
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    print(json.dumps(run(parser.parse_args())))


if __name__=='__main__':main()
