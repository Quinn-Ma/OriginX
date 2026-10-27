"""One worker, exact frozen CUDA services; synthetic plumbing, never a rollout.

Runs only after the root agent has released the GPU from renderer timing tests.
Does not change model source, prompts, precision, or generation limits.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys, threading, time, urllib.request
from contextlib import contextmanager
import numpy as np

ROOT = Path('/ephemeral/qinzhen/originx-harness-v2')
sys.path.insert(0, str(ROOT))
from originx_harness.transport import PolicyClient, dumps, loads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--policy-port', type=int, default=20186)
    ap.add_argument('--planner-port', type=int, default=20180)
    ap.add_argument('--barrier-dir')
    ap.add_argument('--replica-id', type=int, default=0)
    ap.add_argument('--replica-count', type=int, default=1)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    processes, logs, samples, results = [], [], [], {}
    stop = threading.Event()
    client = None
    started = time.monotonic()

    def barrier(phase):
        if not a.barrier_dir: return
        folder = Path(a.barrier_dir)
        folder.mkdir(exist_ok=True, parents=True)
        (folder/f'{phase}-{a.replica_id}').write_text(str(os.getpid()))
        deadline = time.monotonic()+180
        while len(list(folder.glob(f'{phase}-*'))) != a.replica_count:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{phase} replica barrier expired')
            time.sleep(.1)

    @contextmanager
    def inference_slot():
        if not a.barrier_dir:
            yield
            return
        import fcntl
        with (Path(a.barrier_dir)/'interleaved-inference.lock').open('a') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def propose(seed):
        with inference_slot(): return client.propose(seed)

    def launch(role, script, args):
        env = dict(os.environ, HF_HOME=f'/ephemeral/qinzhen/originx-harness-v2/work/hkust/hf-cache-{role}',
                   HF_MODULES_CACHE=f'/ephemeral/qinzhen/originx-harness-v2/work/hkust/hf-modules-{role}',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                   CUDA_VISIBLE_DEVICES='7', CUDA_DEVICE_ORDER='PCI_BUS_ID', TOKENIZERS_PARALLELISM='false')
        log = (out / f'{role}.log').open('w')
        logs.append(log)
        proc = subprocess.Popen([f'/dev/shm/qinzhen/originx-envs/{role}/bin/python', '-u',
                                 str(ROOT / script), *args], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(proc)
        return proc

    def monitor():
        while not stop.is_set():
            try:
                line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '-i', '7',
                                                '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
                used, free, util = (int(x.strip()) for x in line.split(','))
                samples.append({'seconds': time.monotonic()-started, 'used_mib': used,
                                'free_mib': free, 'utilization_percent': util})
            except Exception as exc:
                samples.append({'error': repr(exc)})
            stop.wait(.2)

    def post(payload):
        req = urllib.request.Request(f'http://127.0.0.1:{a.planner_port}/infer',
                                     data=dumps(payload), headers={'Content-Type': 'application/json'})
        with inference_slot():
            with urllib.request.urlopen(req, timeout=600) as reply:
                return loads(reply.read())

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        launch('policy', 'scripts/serve_id60_hierarchical_policy.py',
               ['--config', '/ephemeral/qinzhen/originx-harness-v2/work/hkust/backend-xr1-hkust.json', '--port', str(a.policy_port),
                '--cuda-audit-log', str(out / 'policy-cuda.jsonl')])
        planner = launch('planner', 'scripts/serve_id60_hierarchical_verifier.py',
                         ['--weights', '/dev/shm/qinzhen/originx-models/planner', '--port', str(a.planner_port),
                          '--identity-output', str(out / 'planner-identity.json'), '--duration-seconds', '900'])
        deadline = time.monotonic() + 180
        while True:
            try:
                client = PolicyClient({'port': a.policy_port, 'timeout': 600})
                break
            except ConnectionRefusedError:
                if time.monotonic() > deadline: raise
                time.sleep(.5)
        results['policy_identity'] = client.identity()
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{a.planner_port}/identity', timeout=2) as reply:
                    results['planner_identity'] = loads(reply.read())
                break
            except Exception:
                if planner.poll() is not None or time.monotonic() > deadline: raise
                time.sleep(.5)
        assert results['policy_identity']['checkpoint_sha256'] == 'ccaa65176cf4294b7c0c618e1cc3be7b65617359630c35d3a18dd75c406fc184'
        assert results['planner_identity']['checkpoint_sha256'] == '6efa4249417c81fb61d50abe3d27e7bec93db5766656ca0c1f83da22551512d6'
        barrier('resident')
        results['all_replicas_resident_epoch'] = time.time()
        state = np.zeros(16, np.float32)
        state[[6, 13]] = 1
        state[14:] = [.04, -.04]
        rgb = np.full((3, 256, 256, 3), 96, np.uint8)
        instruction = 'Pick up the cup and place it on the counter.'
        raw = {'rgb': rgb, 'proprio': state, 'instruction': instruction, 'step': 0,
               'episode_id': 'synthetic-worker-mechanical-smoke'}
        client.reset(raw['episode_id'], instruction)
        client.observe(raw)
        t = time.monotonic(); first = propose(7)
        results['first_propose_seconds'] = time.monotonic()-t
        results['actions_shape'] = list(first.shape)
        results['finite'] = bool(np.isfinite(first).all())
        corrected = client.finalize_actions(first, first.copy())
        results['zero_delta_exact'] = bool(np.array_equal(first.view(np.uint32), corrected.view(np.uint32)))
        client.reset(raw['episode_id'], instruction); client.observe(raw)
        second = propose(7)
        results['reset_seed_repeat_exact'] = bool(np.array_equal(first.view(np.uint32), second.view(np.uint32)))
        for step in range(1, 49):
            raw['step'] = step
            client.observe(raw)
            if step % 16 == 0:
                value = propose(7 + step)
                assert value.shape == (16, 12) and np.isfinite(value).all()
        cameras = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        candidates = [{'id': 'O_smoke', 'kind': 'object', 'label': 'cup', 'parent_id': None,
                       'visible': True, 'views': {c: {'bbox': [.25, .25, .75, .75]} for c in cameras},
                       'metadata': {}}]
        plan = post({'rgb': rgb, 'mode': 'plan', 'instruction': instruction,
                     'candidates': candidates, 'camera_names': cameras})
        (out/'planner-plan-response.json').write_text(json.dumps(plan, indent=2))
        step = post({'rgb': rgb, 'reference_rgb': rgb.copy(), 'mode': 'step',
                     'instruction': instruction, 'candidates': candidates, 'camera_names': cameras,
                     'stage': {'operation': 'grasp', 'instruction': 'Pick up the cup.',
                               'completion_predicate': 'The cup is visibly held by the gripper.'},
                     'current_stage_index': 0, 'stage_index': 0, 'bound_target_id': 'O_smoke'})
        (out/'planner-step-response.json').write_text(json.dumps(step, indent=2))
        results['planner_generation_audit'] = [plan.get('cuda_execution_audit'), step.get('cuda_execution_audit')]
        results['status'] = 'passed' if all(results[k] for k in ['finite','zero_delta_exact','reset_seed_repeat_exact']) else 'failed'
        if plan.get('error') or step.get('error'):
            results.update(status='failed', planner_error=[plan.get('error'), step.get('error')])
        barrier('inferred')
    except BaseException as exc:
        results.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if client is not None:
            try: client.close()
            except Exception as exc: results['client_cleanup_error'] = repr(exc)
        for proc in processes:
            if proc.poll() is None: proc.terminate()
        for proc in processes:
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
        stop.set(); watcher.join(timeout=6)
        for log in logs: log.close()
        results.update(schema='originx-cloud-worker-mechanical-smoke-v1',
                       elapsed_seconds=time.monotonic()-started, synthetic_observation=True,
                       policy_score=None, real_model=True, benchmark_episodes=0,
                       limitation='Synthetic 3-image plan and 6-full+3-crop grasp verify; not a worst-case placement/context bound.',
                       samples=samples, max_global_used_mib=max((s.get('used_mib', 0) for s in samples), default=0))
        results['replica_count'] = a.replica_count
        results['cross_replica_inference_schedule'] = 'interleaved_under_mechanical_probe_only_lock' if a.barrier_dir else 'single_worker'
        (out/'receipt.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({k:v for k,v in results.items() if k not in ('samples','policy_identity','planner_identity','planner_generation_audit')}, indent=2))


if __name__ == '__main__': main()
