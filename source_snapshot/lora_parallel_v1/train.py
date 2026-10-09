"""Explicit migration to ordered CPU prefetch; fixed legacy numerical recipe.

No GPU/model/process actions on import. SIGUSR1 requests a checkpointed exit at
an optimizer boundary. New execution identity and parent SHA are never hidden
inside the legacy numerical/source identity.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_pinned(path, expected):
    if sha(path) != expected:
        raise ValueError('Pinned file changed: ' + str(path))
    return json.loads(Path(path).read_text())


def validate_sources(root, sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Empty source inventory')
    for name, digest in sources.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Changed/outside-root source: ' + name)


def execution_identity(binding, numerical_identity, sources):
    return dict(numerical_identity, execution={
        'kind': 'ordered_cpu_prefetch_v1',
        'source_sha256': sources,
        'source_inventory_sha256': binding['execution_sources_sha256'],
        'parent_checkpoint_sha256': binding['parent_checkpoint_sha256'],
        'parent_run_identity_sha256': binding['parent_run_identity_sha256'],
        'workers': binding.get('workers', 8), 'prefetch': binding.get('prefetch', 16),
        'checkpoint_interval': 250, 'numerical_recipe_changed': False,
    })


class BoundaryStop:
    def __init__(self):
        self.requested = False
        self.signals = 0
    def handler(self, signum, frame):
        self.requested = True
        self.signals += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--resume', help='Only a checkpoint from this new output namespace')
    parser.add_argument('--stop-after-updates', type=int, help='Engineering benchmark only: save/pause after this many new updates')
    args = parser.parse_args()
    if args.stop_after_updates is not None and not 1 <= args.stop_after_updates <= 10000:
        raise ValueError('Engineering stop count must be1..10000')
    binding = json.loads(Path(args.binding).read_text())
    root = Path(binding['root']).resolve()
    config = read_pinned(root / 'lora_retention_run_v1/training_config.json', binding['config_sha256'])
    numerical_sources = read_pinned(root / 'lora_retention_run_v1/runtime_sources.json', config['runtime_sources_sha256'])
    validate_sources(root, numerical_sources)
    actual_sources = read_pinned(root / binding['execution_sources'], binding['execution_sources_sha256'])
    validate_sources(root, actual_sources)
    for required in ('lora_parallel_v1/train.py', 'lora_parallel_v1/provider.py',
                     'cpu_prefetch_mechanical_v1/run.py', 'cpu_prefetch_mechanical_v1/core.py',
                     'cpu_prefetch_mechanical_v1/external-pins.json'):
        if required not in actual_sources:
            raise ValueError('Missing new execution source: ' + required)
    if config['updates'] != 10000 or config['batch'] != 8 or config['checkpoint_every'] != 250:
        raise ValueError('The fixed10000/batch8/checkpoint250 numerical recipe changed')
    arm = binding['arm']
    if arm not in ('L', 'A') or binding['physical_gpu'] not in (0, 1, 5, 6):
        raise ValueError('Unknown arm or excluded physical GPU')
    if binding.get('workers', 8) != 8 or binding.get('prefetch', 16) < 8:
        raise ValueError('This execution revision requires8workers and at least8prefetch')
    parent_checkpoint = Path(binding['parent_checkpoint']).resolve()
    if sha(parent_checkpoint) != binding['parent_checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    parent_identity = read_pinned(binding['parent_run_identity'], binding['parent_run_identity_sha256'])
    out = Path(binding['output']).resolve()
    if out == parent_checkpoint.parent or out.is_relative_to(parent_checkpoint.parent):
        raise ValueError('Migration requires a separate new output namespace')
    os.environ.update(CUDA_VISIBLE_DEVICES=binding['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.resume:
        if not out.is_dir() or (out / 'completion.json').exists():
            raise ValueError('Resume requires an unfinished new namespace')
        resume_path = Path(args.resume).resolve()
        if resume_path.parent != out:
            raise ValueError('Own resume checkpoint must belong to the new namespace')
    else:
        out.mkdir(parents=True, exist_ok=False)
    run_lock = (out / '.training.lock').open('a')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = BoundaryStop()
    signal.signal(signal.SIGUSR1, stop.handler)
    attempt = str(time.time_ns())
    owner = {'pid': os.getpid(), 'gpu_uuid': binding['gpu_uuid'], 'physical_gpu': binding['physical_gpu'],
             'start_ticks': Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19],
             'started_unix': time.time(), 'binding_sha256': sha(args.binding),
             'resume': args.resume, 'parent_checkpoint_sha256': binding['parent_checkpoint_sha256'],
             'engineering_stop_after_updates': args.stop_after_updates}
    write_json(out / ('owner-' + attempt + '.json'), owner)
    write_json(out / 'owner.json', owner)
    provider = None
    try:
        import random
        import numpy as np
        import torch
        from transformers import AutoModel, AutoProcessor
        from training_bridge.train_single import checkpoint_identity, runtime_identity
        from human_mg_training_v1.bridge import slot_draws, describe
        from online_flow_training_gate_v1.check import gpu_inventory
        from route_reassessment_20261004.action_lora.action_lora import inject_action_lora, save_adapter, _tensor_digest
        from route_reassessment_20261004.action_lora.lora_flow_bridge import LoRAFlowBridge
        from route_reassessment_20261004.action_lora.training_checkpoint import save_training_checkpoint, load_training_checkpoint, _digest, _rng
        from .provider import make_provider
        write_json(out / ('gpu-' + attempt + '.json'), gpu_inventory(binding['physical_gpu'], binding['gpu_uuid']))
        smoke = read_pinned(root / config['mechanical_report'], config['mechanical_report_sha256'])
        if not smoke['passed']:
            raise ValueError('Original real-model mechanical check did not pass')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        write_json(out / 'status.json', {'stage': 'verifying_source_and_data', 'arm': arm, 'time': time.time()})
        assets = checkpoint_identity(config['model'])
        if assets != json.loads((root / config['base_assets']).read_text()):
            raise ValueError('Original released checkpoint assets changed')
        base_id = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
        processor = AutoProcessor.from_pretrained(config['model'], trust_remote_code=True, use_fast=False, local_files_only=True)
        provider = make_provider(arm, root / config['data_binding'], config['data_binding_sha256'], processor,
                                 'cuda:0', expected_config_sha256=config['sampling_config_sha256'],
                                 processor_path=config['model'], workers=8, prefetch=binding.get('prefetch', 16))
        write_json(out / 'status.json', {'stage': 'loading_model', 'arm': arm, 'time': time.time()})
        model = AutoModel.from_pretrained(config['model'], trust_remote_code=True, attn_implementation='flash_attention_2',
                                         dtype=torch.bfloat16, local_files_only=True).to('cuda:0').to(torch.bfloat16).eval()
        handle = inject_action_lora(model, base_identity=base_id, **config['lora'])
        bridge = LoRAFlowBridge(handle, retention_coefficient=config['retention_coefficients'][arm])
        parameters = list(bridge.trainable_parameters())
        if sum(p.numel() for p in parameters) != 3538944:
            raise ValueError('Wrong trainable parameter count')
        opt = config['optimizer']
        optimizer = torch.optim.AdamW(parameters, lr=opt['lr'], betas=tuple(opt['betas']),
                                      eps=opt['eps'], weight_decay=opt['weight_decay'], foreach=False)
        random.seed(config['train_seed']); np.random.seed(config['train_seed']); torch.manual_seed(config['train_seed'])
        numerical_identity = {'config_sha256': binding['config_sha256'], 'source_sha256': numerical_sources,
                              'arm': arm, 'base_identity': base_id, 'base_tensor_sha256': handle.metadata['base_state_sha256'],
                              'sampling_config_sha256': config['sampling_config_sha256'], 'runtime': runtime_identity()}
        if parent_identity != numerical_identity:
            raise ValueError('Parent run identity differs from independently reconstructed legacy identity')
        identity = execution_identity(binding, numerical_identity, actual_sources)
        identity['execution']['engineering_stop_after_updates'] = args.stop_after_updates
        common = dict(handle=handle, optimizer=optimizer, provider=provider,
                      device='cuda:0', accumulate=config['batch'], max_grad_norm=config['max_grad_norm'])
        if args.resume:
            if json.loads((out / 'run_identity.json').read_text()) != identity:
                raise ValueError('New execution identity changed on resume')
            latest = json.loads((out / 'latest_checkpoint.json').read_text())
            if Path(latest['path']).resolve() != resume_path or sha(resume_path) != latest['file_sha256']:
                raise ValueError('Resume must use the latest pinned checkpoint from this namespace')
            step = load_training_checkpoint(resume_path, identity=identity, **common)
        else:
            # Strict old identity is used only for this first parent load.
            step = load_training_checkpoint(parent_checkpoint, identity=numerical_identity, **common)
            write_json(out / 'numerical_identity.json', numerical_identity)
            write_json(out / 'run_identity.json', identity)
            write_json(out / 'migration.json', {'kind': 'explicit_execution_migration_v1', 'arm': arm,
                'parent_checkpoint': str(parent_checkpoint), 'parent_checkpoint_sha256': binding['parent_checkpoint_sha256'],
                'parent_identity_sha256': binding['parent_run_identity_sha256'], 'restored_step': step,
                'restored_samples': provider.cursor, 'numerical_identity_verified': True,
                'execution_sources_sha256': actual_sources, 'execution_identity_sha256': _digest(identity),
                'original_weights_untouched': True, 'new_namespace': str(out), 'time': time.time()})
        if not 0 <= step <= 10000 or provider.cursor != step * config['batch']:
            raise ValueError('Recovered cursor/update mismatch')
        bridge._assert_trainable_contract()
        common['identity'] = identity
        start_step = step
        engineering_end = start_step + args.stop_after_updates if args.stop_after_updates is not None else None
        if engineering_end is not None and engineering_end > 10000:
            raise ValueError('Engineering benchmark exceeds fixed training endpoint')
        torch.cuda.reset_peak_memory_stats()
        recent = collections.deque(maxlen=30)
        recent_wait = collections.deque(maxlen=30)
        started = time.monotonic()

        def save_boundary():
            begin = time.monotonic()
            path = out / f'checkpoint-{step:08d}.pt'
            if path.exists():
                latest = json.loads((out / 'latest_checkpoint.json').read_text())
                if latest['step'] != step or Path(latest['path']).resolve() != path or sha(path) != latest['file_sha256']:
                    raise ValueError('Existing boundary checkpoint lacks an exact matching receipt')
                return latest
            saved = save_training_checkpoint(path, step=step, **common)
            saved.update(file_sha256=sha(path), execution_identity_sha256=_digest(identity),
                         parent_checkpoint_sha256=binding['parent_checkpoint_sha256'],
                         execution_source_inventory_sha256=binding['execution_sources_sha256'],
                         save_wall_seconds=time.monotonic() - begin)
            write_json(out / f'checkpoint-{step:08d}.receipt.json', saved)
            write_json(out / 'latest_checkpoint.json', saved)
            return saved

        def stop_at_boundary(reason='SIGUSR1'):
            saved = save_boundary()
            paused = {'stage': 'engineering_pause' if args.stop_after_updates is not None else 'paused_at_optimizer_boundary',
                      'arm': arm, 'step': step, 'start_step': start_step, 'new_updates': step - start_step,
                      'samples': provider.cursor, 'checkpoint': saved, 'reason': reason,
                      'engineering_only': args.stop_after_updates is not None, 'candidate_selected': False,
                      'signals_received': stop.signals, 'time': time.time(), 'complete': False}
            write_json(out / ('pause-' + attempt + '.json'), paused)
            write_json(out / 'status.json', paused)
            print(json.dumps(paused), flush=True)

        if stop.requested:
            stop_at_boundary()
            return
        logname = 'metrics-' + attempt + '.jsonl'
        with (out / logname).open('x') as stream:
            while step < config['updates']:
                wall = time.monotonic()
                bridge.train(); optimizer.zero_grad(set_to_none=True)
                flow, retention, losses, ids = [], [], [], []
                timing = collections.defaultdict(float)
                cuda_segments = {'forward': [], 'backward': [], 'optimizer': []}
                for micro in range(config['batch']):
                    tick = time.monotonic()
                    inputs, actions, valid, sample_id = provider.next_batch()
                    timing['provider_wait_and_transfer_wall_seconds'] += time.monotonic() - tick
                    for key, value in (getattr(provider, 'last_prefetch_timing', None) or {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            timing['prefetch_' + key] += value
                    if sample_id['slot_index'] != step * config['batch'] + micro:
                        raise ValueError('Unexpected training slot')
                    tick = time.monotonic()
                    noise, times = slot_draws(inputs['action_mask'], sample_id['slot_seed'])
                    timing['noise_schedule_wall_seconds'] += time.monotonic() - tick
                    f0, f1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    f0.record()
                    result = bridge(inputs, actions, valid, noise=noise, times=times)
                    f1.record(); cuda_segments['forward'].append((f0, f1))
                    loss = result['loss']
                    if not torch.isfinite(loss).item(): raise FloatingPointError('Nonfinite loss')
                    b0, b1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    b0.record(); (loss / config['batch']).backward(); b1.record()
                    cuda_segments['backward'].append((b0, b1))
                    tick = time.monotonic()
                    losses.append(loss.detach().item()); flow.append(result['flow_loss'].detach().item())
                    retention.append(result['retention_loss'].detach().item())
                    sample_id['tensor_sha256'] = {k: describe(v)['sha256'] for k, v in
                        [('actions', actions), ('valid', valid), ('noise', noise), ('times', times)]}
                    timing['metrics_hash_and_sync_wall_seconds'] += time.monotonic() - tick
                    ids.append(sample_id)
                o0, o1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                o0.record()
                norm = torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                o1.record(); cuda_segments['optimizer'].append((o0, o1))
                step += 1; torch.cuda.synchronize()
                for name, segments in cuda_segments.items():
                    timing[name + '_cuda_seconds'] = sum(a.elapsed_time(b) for a, b in segments) / 1000
                seconds = time.monotonic() - wall
                recent.append(seconds); recent_wait.append(timing['provider_wait_and_transfer_wall_seconds'])
                row = {'step': step, 'arm': arm, 'loss': sum(losses) / len(losses),
                       'flow_loss': sum(flow) / len(flow), 'retention_loss': sum(retention) / len(retention),
                       'gradient_norm': norm.item(), 'wall_seconds': seconds, 'pipeline_timing': dict(timing),
                       'completed_unix': time.time(), 'samples': ids,
                       'engineering_only': args.stop_after_updates is not None}
                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                status = {k: v for k, v in row.items() if k != 'samples'}
                status.update(stage='engineering_benchmark' if args.stop_after_updates is not None else 'training',
                              samples=provider.cursor, planned_steps=10000, engineering_end_step=engineering_end,
                              prefetch=provider.prefetch_status(),
                              recent_seconds_per_step=sum(recent)/len(recent),
                              recent_provider_wait_fraction=sum(recent_wait)/sum(recent),
                              estimated_remaining_seconds=(10000-step)*sum(recent)/len(recent))
                write_json(out / 'status.json', status)
                if step <= start_step + 3 or step % 25 == 0: print(json.dumps(status), flush=True)
                if step % 250 == 0 or step == 10000: save_boundary()
                if stop.requested:
                    stop_at_boundary()
                    return
                if engineering_end is not None and step >= engineering_end:
                    stop_at_boundary('fixed_engineering_update_budget')
                    return
            if provider.cursor != 80000 or provider.source_counts != {'human': 80000, 'mg': 0}:
                raise ValueError('Wrong terminal coverage')
            adapter = save_adapter(handle, out / 'adapter-step-00010000.pt')
            completion = {'completed': True, 'arm': arm, 'updates': step, 'samples': provider.cursor,
                          'adapter': adapter, 'adapter_file_sha256': sha(adapter['path']),
                          'identity_sha256': sha(out / 'run_identity.json'),
                          'parent_checkpoint_sha256': binding['parent_checkpoint_sha256'],
                          'migration_sha256': sha(out / 'migration.json'), 'start_step': start_step,
                          'metrics_files': {p.name: sha(p) for p in out.glob('metrics*.jsonl')},
                          'elapsed_seconds': time.monotonic()-started,
                          'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30, 'completed_unix': time.time()}
            write_json(out / 'completion.json', completion)
            write_json(out / 'status.json', dict(completion, stage='complete'))
    except BaseException as error:
        failure = {'error': repr(error), 'traceback': traceback.format_exc(), 'time': time.time(),
                   'no_uncontrolled_retry': True}
        write_json(out / ('failure-' + attempt + '.json'), failure)
        write_json(out / 'status.json', dict(failure, stage='failed', arm=arm))
        raise
    finally:
        if provider is not None and hasattr(provider, 'close'):
            provider.close()


if __name__ == '__main__':
    main()
