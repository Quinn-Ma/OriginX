import copy
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


join=load('aggregate_reduced_v1');v2=load('aggregate_reduced_v2');c=load('continue_multigpu_v2')


def fixture():
    jobs=[dict(id=str(i),request_id='r'+str(i),case_id='case',task='task',policy_id='B' if i<5 else 'base',
        arm=('C','R','G','L','V','C','V')[i],env_seed=1,horizon=100) for i in range(7)]
    manifest=dict(jobs=jobs,cases=[dict(case_id='case',task='task',env_seed=1)],case_count=1)
    cohort=dict(case_ids=['case'])
    old={jobs[0]['id']:dict(job=jobs[0],config_sha256='old')}
    first={jobs[1]['id']:dict(job=jobs[1],config_sha256='first')}
    new={jobs[2]['id']:dict(job=jobs[2],config_sha256='new')}
    s1=c.build_selection(manifest,old,'old',cohort)
    union={key:dict(value,config_sha256='union') for claims in (old,first) for key,value in claims.items()}
    s2=c.build_selection(manifest,union,'union',cohort)
    sets=[]
    for label,claims in [('old',old),('first',first),('new',new)]:
        sets.append([dict(episode_id=j['id'],case_id=j['case_id'],task_name=j['task'],policy_id=j['policy_id'],
            arm=j['arm'],seed=j['env_seed'],horizon=j['horizon'],trace=[],record_present=j['id'] in claims,
            success=False if j['id'] in claims else None,valid=j['id'] in claims,label=label) for j in jobs])
    return manifest,s1,s2,sets,[old,first,new]


def test_three_batch_join_preserves_prior_negative_unknown_and_never_uses_better_later_result():
    manifest,s1,s2,sets,claims=fixture()
    sets[1][1].update(valid=False,success=None,adapter_error='original_v1_transport_failure')
    combined,reduced,retained,provenance=v2.merge_three(join,manifest,s1,s2,sets,claims)
    assert len(combined)==len(reduced)==7 and len(retained)==0
    assert reduced[0]['label']=='old' and reduced[0]['success'] is False
    assert reduced[1]['label']=='first' and reduced[1]['valid'] is False
    assert reduced[1]['adapter_error']=='original_v1_transport_failure'
    assert reduced[2]['label']=='new'
    assert [p['source_batch'] for p in provenance[:3]]==[v2.OLD_NAME,v2.V1_NAME,v2.NEW_NAME]


def test_three_batch_duplicate_claim_is_rejected_even_if_success_is_better():
    manifest,s1,s2,sets,claims=fixture()
    claims[2].update(claims[0]);sets[2][0].update(record_present=True,success=True)
    with pytest.raises(RuntimeError,match='Repeated'):v2.merge_three(join,manifest,s1,s2,sets,claims)


def test_three_batch_union_partition_tampering_is_rejected():
    manifest,s1,s2,sets,claims=fixture()
    s2['excluded_claimed_ids']=s2['excluded_claimed_ids'][:-1]
    with pytest.raises(RuntimeError,match='claim set'):v2.merge_three(join,manifest,s1,s2,sets,claims)
