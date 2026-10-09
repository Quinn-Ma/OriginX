"""Prepared bounded pilot. Default mode only checks files and writes a preflight.

Run from a parent containing the unchanged training_bridge and this package.
GPU modes are explicit: lambda0-gate (2 original-control updates), then train
(200 total anchored updates). No training, CUDA access or model load on import.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from .contract import (PLAN_SHA256, anchored_manifest, canonical_sha, read, sha,
                       verify_bridge_sources, verify_inputs)

HERE = Path(__file__).resolve().parent


def code_identity():
    return {path.name: sha(path) for path in sorted(HERE.glob('*.py'))}


def prepare(args):
    import training_bridge
    plan, manifest, control, trace, steps = verify_inputs(args.plan, args.manifest, args.control_run)
    code = Path(training_bridge.__file__).parent
    verify_bridge_sources(control, code)
    contract = dict(method_plan_sha256=PLAN_SHA256, objective='flow_plus_frozen_base_velocity_anchor',
                    anchor_coefficient=1.0 if args.mode == 'train' else 0.0,
                    mode=args.mode, pilot_code=code_identity(),
                    control_trace_sha256=plan['data']['control_sample_trace_sha256'],
                    control_identity_sha256=sha(args.control_run / 'run_identity.json'),
                    noise_schedule='original BF16 randn_like(mask), then BF16 uniform rand; seed193 after construction',
                    teacher_initialization='independent FP32 action copies from released base before any student resume')
    return plan, manifest, control, trace, steps, contract


def build_provider(readers, processor, device, seed, trace):
    from training_bridge.train_single import SamplePool
    class VerifiedSamplePool(SamplePool):
        def next_batch(self):
            result = super().next_batch()
            if self.draws > len(trace) or result[3] != trace[self.draws - 1]:
                raise ValueError('Actual decoded sample differs from frozen control trace')
            return result
    return VerifiedSamplePool(readers, processor, device, seed)


def assert_tree_equal(actual, expected, name='root'):
    import numpy as np
    import torch
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, msg=name)
    elif isinstance(expected, np.ndarray):
        if not np.array_equal(actual, expected):
            raise AssertionError(name)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(name + ': keys')
        for key in expected:
            assert_tree_equal(actual[key], expected[key], name + '/' + str(key))
    elif isinstance(expected, (list, tuple)):
        if type(actual) != type(expected) or len(actual) != len(expected):
            raise AssertionError(name + ': sequence')
        for index, value in enumerate(expected):
            assert_tree_equal(actual[index], value, name + '/' + str(index))
    elif actual != expected:
        raise AssertionError(name)


def reconcile_metrics(output, completed_step):
    """Archive an interrupted attempt's tail before replay from its checkpoint."""
    path = output / 'metrics.jsonl'
    if not path.exists():
        if completed_step:
            raise ValueError('Checkpoint has no corresponding metric history')
        return
    lines = path.read_bytes().splitlines(keepends=True)
    if not completed_step:
        if any(line.strip() for line in lines):
            raise ValueError('Uncheckpointed prior attempt requires a new output directory')
        return
    prefix = []
    for index in range(completed_step):
        if index >= len(lines) or json.loads(lines[index])['step'] != index + 1:
            raise ValueError('Metric prefix does not match the completed checkpoint')
        prefix.append(lines[index])
    tail = b''.join(lines[completed_step:])
    if tail:
        # Preserve even a partially written final JSON line as original evidence.
        archive = output / f'metrics-abandoned-after-{completed_step:08d}-{time.time_ns()}.jsonl'
        with archive.open('xb') as stream:
            stream.write(tail); stream.flush(); os.fsync(stream.fileno())
        temporary = output / 'metrics-resume.tmp'
        with temporary.open('xb') as stream:
            stream.write(b''.join(prefix)); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)


def run_gpu(args, prepared):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor
    from training_bridge import train_single as original
    from training_bridge.bridge import TRAINABLE_MODULES
    from training_bridge.lerobot_reader import LeRobotPretrainReader
    from training_bridge.provenance import provenance_from_acquisition
    from training_bridge.train_core import SingleDeviceTrainer, rng_state
    from .bridge import TeacherAnchoredFlowBridge

    plan, source, control, trace, control_steps, contract = prepared
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('GPU execution requires an explicitly scheduled CUDA device')
    torch.cuda.set_device(device)
    runtime = original.runtime_identity()
    if runtime != control['identity']['runtime']:
        raise ValueError('Original control runtime differs')
    if args.mode == 'train':
        if args.lambda0_gate is None:
            raise ValueError('First complete the separate exact lambda0-gate')
        gate = read(args.lambda0_gate)
        if (gate.get('status') != 'passed' or gate.get('method_plan_sha256') != PLAN_SHA256
                or gate.get('pilot_code') != code_identity() or gate.get('runtime') != runtime
                or gate.get('comparison') != 'exact_loss_heads_optimizer_provider_rng_after2updates'):
            raise ValueError('Lambda0 real-model/runtime equivalence evidence differs')
        contract['lambda0_gate_sha256'] = sha(args.lambda0_gate)
    checkpoints = sorted(args.output.glob('checkpoint-step-*.pt'))
    if args.resume is None and checkpoints:
        raise FileExistsError('Use the latest explicit resume or a new output directory')
    if args.resume is None:
        reconcile_metrics(args.output, 0)
    if args.resume is not None and (args.mode != 'train' or not checkpoints or args.resume.resolve() != checkpoints[-1].resolve()):
        raise ValueError('Only latest own candidate checkpoint may be resumed')
    readers = []
    for item in source['datasets']:
        if item['role'] != 'train':
            raise ValueError('Training data role differs')
        provenance = provenance_from_acquisition(item['root'], item['acquisition'], source['registry'], source['box_links'])
        if item['task'] != provenance['task'] or item['archive_sha256'] != provenance['archive_sha256']:
            raise ValueError('Data acquisition differs')
        readers.append(LeRobotPretrainReader(item['root'], provenance, source['registry'], source['box_links'], episode_ids=item['episode_ids']))
    inventories = [reader.inventory() for reader in readers]
    if inventories != control['identity']['data']:
        raise ValueError('Human300 inventory differs from control')
    if original.checkpoint_identity(source['base_model']) != control['identity']['base_checkpoint']:
        raise ValueError('Released base assets differ')
    manifest = anchored_manifest(source, contract)
    identity = dict(control['identity'], manifest_sha256=canonical_sha(manifest), runtime=runtime)
    job = dict(identity=identity, manifest=manifest)
    job_path = args.output / 'run_identity.json'
    if job_path.exists():
        if read(job_path) != job:
            raise ValueError('Candidate recipe/code/control/gate identity changed')
    else:
        if args.resume is not None:
            raise ValueError('Resume requires original run_identity.json')
        job_path.write_text(json.dumps(job, indent=2) + '\n')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(source['base_model'], trust_remote_code=True, use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(source['base_model'], trust_remote_code=True, attn_implementation='flash_attention_2',
                                      dtype=torch.bfloat16, local_files_only=True).to(device).to(torch.bfloat16)
    bridge = TeacherAnchoredFlowBridge(model, anchor_coefficient=contract['anchor_coefficient'])
    cfg = source['optimizer']
    optimizer = torch.optim.AdamW(bridge.trainable_parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                                  eps=cfg['eps'], weight_decay=cfg['weight_decay'], foreach=False)
    # Preserve the original control's exact post-construction seed order.
    random.seed(source['seed']); np.random.seed(source['seed']); torch.manual_seed(source['seed'])
    provider = build_provider(readers, processor, device, source['seed'], trace)
    trainer = SingleDeviceTrainer(bridge, optimizer, provider, identity=identity, device=device,
                                  accumulate=source['gradient_accumulation'], max_grad_norm=source['max_grad_norm'])
    if args.resume is not None:
        trainer.load(args.resume)  # Teacher still contains released-base values.
        if args.resume.name != f'checkpoint-step-{trainer.global_step:08d}.pt' or not 1 <= trainer.global_step <= 200:
            raise ValueError('Resume filename/step violates the frozen pilot')
        reconcile_metrics(args.output, trainer.global_step)
    limit = 2 if args.mode == 'lambda0-gate' else 200
    estimated_bytes = int(sum(p.numel() for p in bridge.trainable_parameters()) * 12 * 1.1)
    events = []
    with (args.output / 'metrics.jsonl').open('a') as log:
        while trainer.global_step < limit:
            if shutil.disk_usage(args.output).free < estimated_bytes + 50 * 2**30:
                raise OSError('Required checkpoint disk reserve unavailable')
            started = time.monotonic()
            event = trainer.step()
            bridge.teacher.assert_frozen()
            events.append(copy.deepcopy(event))
            event.update(wall_seconds=time.monotonic() - started, lr=cfg['lr'])
            log.write(json.dumps(event) + '\n'); log.flush()
            print(json.dumps({k:v for k,v in event.items() if k != 'samples'}), flush=True)
            if trainer.global_step % 50 == 0 or trainer.global_step == limit:
                trainer.save(args.output / f'checkpoint-step-{trainer.global_step:08d}.pt')
    if args.mode == 'lambda0-gate':
        reference_path = args.control_run / 'checkpoint-step-00000002.pt'
        # Only the trusted original locally-created training checkpoint is read.
        reference = torch.load(reference_path, map_location='cpu', weights_only=False, mmap=True)
        if reference['identity'] != control['identity'] or reference['global_step'] != 2:
            raise ValueError('Original step2 reference differs')
        for actual, expected in zip(events, control_steps[:2]):
            for key in ['step', 'loss', 'gradient_norm_before_clip', 'samples']:
                assert_tree_equal(actual[key], expected[key], key)
        heads = {name: getattr(model, name).state_dict() for name in TRAINABLE_MODULES}
        for label, actual in [('heads', heads), ('optimizer', optimizer.state_dict()),
                              ('provider', provider.state_dict()), ('rng', rng_state(device))]:
            assert_tree_equal(actual, reference[label], label)
        gate = dict(status='passed', method_plan_sha256=PLAN_SHA256, pilot_code=code_identity(), runtime=runtime,
                    comparison='exact_loss_heads_optimizer_provider_rng_after2updates',
                    reference_checkpoint_sha256=sha(reference_path),
                    limitation='Real teacher forward is excluded atlambda0; teacher runtime guards remain active during candidate training.')
        (args.output / 'lambda0-gate.json').write_text(json.dumps(gate, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'lambda0-gate', 'train'], default='preflight')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--control-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--lambda0-gate', type=Path)
    args = parser.parse_args()
    prepared = prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'preflight':
            report = dict(status='prepared_no_training', contract=prepared[-1], windows=len(prepared[3]),
                          limitation='Metadata and sample-sequence checks only; exact real-model gate is still required.')
            with (args.output / 'preflight.json').open('x') as stream:
                json.dump(report, stream, indent=2); stream.write('\n')
        else:
            run_gpu(args, prepared)


if __name__ == '__main__':
    main()
