"""Capture two disclosed development reset observations, zero policy/steps.

Run explicitly in the existing CPU simulation environment. This does not load
a model, create an inference client, read scored outcomes, or alter reset code.
"""
import argparse
import os
from pathlib import Path
import random
import sys
import time
import traceback
from . import runner as r
from .adapter import CAMERAS,STATE_DIMS,LANGUAGE,build_observation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--reference-config-sha256',required=True)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--reference-manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    r.require(r.sha(args.reference_config)==args.reference_config_sha256,'Native config pin changed')
    r.require(r.sha(args.reference_manifest)==args.reference_manifest_sha256,'Fresh roster pin changed')
    config=r.read(args.reference_config)
    for name,digest in config['source_sha256'].items():r.require(r.sha(name)==digest,'Pinned simulator source changed: '+name)
    r.require(r.sha(Path(config['repo'])/'eval_robocasa365/entry.py')==r.ENTRY_SHA,'Official entry source differs')
    r.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONHASHSEED')=='0',
              'Use the pinned CPU-only native simulation environment')
    out=args.output.resolve();base=r.ROOT/r.NAMESPACE
    r.require(out.is_relative_to(base/'fixtures') and not out.exists(),'Fresh fixture directory in independent namespace required')
    manifest=r.make_manifest(r.read(args.reference_manifest),development=True)
    out.mkdir(parents=True,exist_ok=False)
    r.write(out/'owner.json',dict(r.identity(),namespace=r.NAMESPACE,purpose='two native reset observations; zero steps/inference'),True)
    r.setup_path()
    import importlib.metadata
    import numpy as np
    import torch
    import gymnasium as gym
    import robocasa
    from context_lifetime import install
    from readback_guard import install_guard
    from official_b2500_v1.runner import guard_complete
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    r.require({name:importlib.metadata.version(name) for name in config['packages']}==config['packages'],
              'Simulation packages differ')
    before=r.native_reset_proof(config);lifetime=install()
    guard=install_guard(out/'readback-guard',library_path=config['osmesa_library'])
    records=[];error=None;started=time.monotonic()
    try:
        for i,case in enumerate(manifest['cases']):
            r.require(get_task_horizon(case['task'])==case['horizon'],'Official development horizon changed')
            random.seed(case['seed']);np.random.seed(case['seed'])
            env=None
            try:
                env=gym.make('robocasa/'+case['task'],split='pretrain',seed=case['seed'])
                observation,_=env.reset(seed=case['seed'])
                guard.check()
                instruction=observation[LANGUAGE]
                build_observation(observation,instruction)
                # Export public inputs only. No physics/qpos/qvel/object poses,
                # segmentation, success flags, or post-reset task outcomes.
                native={key:np.array(observation[key],copy=True) for key in (*CAMERAS,*STATE_DIMS)}
                native[LANGUAGE]=np.asarray([instruction])
                path=out/f'fixture-{i}.npz'
                with path.open('xb') as stream:np.savez_compressed(stream,**native)
                r.require(r.native_reset_proof(config)==before,'Reset implementation changed during capture')
                records.append(dict(path=str(path),sha256=r.sha(path),instruction=instruction,
                    task=case['task'],seed=case['seed'],development_case_id=case['case_id'],
                    synthetic=False,gym_resets=1,environment_steps=0,policy_queries=0,
                    observation_source='same returned native Gym.reset observation',public_fields_only=True))
            finally:
                if env is not None:env.close()
            guard.check()
        from robosuite.utils import binding_utils
        r.require(not getattr(binding_utils,'_osmesa_lifetime_failure',None),'Renderer context close failure')
        r.require(not torch.cuda.is_initialized(),'Fixture process unexpectedly initialized CUDA')
    except BaseException:
        error=traceback.format_exc();r.write(out/'error.json',dict(error=error),True)
    finally:
        try:guard.close()
        except BaseException:
            if error is None:error=traceback.format_exc();r.write(out/'close-error.json',dict(error=error),True)
    healthy=guard_complete(r.read(out/'readback-guard/summary.json'))
    r.write(out/'capture-receipt.json',dict(schema='originx_gr00t_native_fixture_capture_v1',passed=error is None and healthy and len(records)==2,
        native_reset=before,readback_guard_sha256=r.sha(out/'readback-guard/summary.json'),context_lifetime=lifetime,
        reference_config_sha256=args.reference_config_sha256,reference_manifest_sha256=args.reference_manifest_sha256,
        source_sha256={str(Path(__file__).resolve()):r.sha(__file__),str(Path(r.__file__).resolve()):r.sha(r.__file__)},
        synthetic=False,development_ids_disclosed=True,environment_steps=0,policy_queries=0,
        no_scored_outcomes_read=True,elapsed_seconds=time.monotonic()-started,error=error),True)
    r.require(error is None and healthy and len(records)==2,'Native fixture capture failed; evidence retained')
    r.write(out/'fixtures.json',dict(schema='originx_gr00t_parity_fixtures_v1',synthetic=False,
        purpose='Real native development reset inputs for infrastructure parity; not task-performance evidence',
        fixtures=records,capture_receipt_sha256=r.sha(out/'capture-receipt.json'),no_hidden_state_exported=True),True)
    print(str(out/'fixtures.json'));print(r.sha(out/'fixtures.json'))


if __name__=='__main__':main()
