"""CPU-only binding of the naturally completed 1613-update endpoint."""
import argparse
import json
import os
from pathlib import Path
from .loader import SERVING_SOURCES, sha, require, read_binding


def bind_endpoint(*, root, arm, training_output, output):
    root = Path(root).resolve(); out = Path(training_output).resolve(); output = Path(output).resolve()
    require(arm in ('L', 'A'), 'Unknown arm')
    require(out == root / 'training/lora-eight-gpu-v1' / arm, 'Use original audited final training directory')
    require(not output.exists(), 'Use a new endpoint binding')
    def pin(path):
        p = Path(path).resolve()
        return {'path': str(p), 'sha256': sha(p)}
    config_path = root / 'lora_retention_run_v1/training_config.json'
    config = json.loads(config_path.read_text())
    identity = json.loads((out / 'run_identity.json').read_text())
    complete = json.loads((out / 'completion.json').read_text())
    audit_path = root / 'lora_acceleration_ops_v1/eight-gpu-completion-audit.json'
    audit = json.loads(audit_path.read_text()); report = audit['arms'][arm]
    parent = root / 'training/lora-batch64-v2' / arm
    original = root / 'training/lora-retention-v1' / arm
    b = {
        'schema': 'lora_dev150_final_endpoint_v1', 'ready': True, 'arm': arm, 'root': str(root),
        'endpoint_updates': 1613, 'endpoint_samples': 80000, 'global_batch': 128,
        'base_model': config['model'], 'xr1_repo': str(root / 'code/Xiaomi-Robotics-1'),
        'base_identity': identity['base_identity'], 'base_tensor_sha256': identity['base_tensor_sha256'],
        'training_config_sha256': sha(config_path), 'training_config': pin(config_path),
        'batch128_config': pin(root / 'lora_parallel8_v1/training_config128.json'),
        'batch64_config': pin(root / 'lora_parallel64_v1/training_config64.json'),
        'lora': config['lora'], 'adapter': pin(complete['adapter']['path']),
        'checkpoint': pin(report['checkpoint']['path']), 'completion_audit': pin(audit_path),
        'completion': pin(out / 'completion.json'), 'run_identity': pin(out / 'run_identity.json'),
        'migration': pin(out / 'migration.json'), 'parent64_identity': pin(parent / 'run_identity.json'),
        'parent64_migration': pin(parent / 'migration.json'), 'original_identity': pin(original / 'run_identity.json'),
        'source_sha256': {str(root / name): sha(root / name) for name in SERVING_SOURCES},
        'inference_arithmetic': 'full_forward_cuda_bfloat16_autocast_fp32_adapter_master',
        'ready_scope': 'Exact final1613/80000 artifact and source binding; actual model/base loading, socket parity and renderer readiness remain separate checks.',
    }
    candidate = output.with_suffix(output.suffix + '.candidate')
    require(not candidate.exists(), 'Candidate binding already exists')
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open('x') as stream:
        stream.write(json.dumps(b, indent=2) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    read_binding(candidate, sha(candidate))
    # Hard-link publication fails rather than overwriting an existing endpoint.
    os.link(candidate, output); candidate.unlink()
    return {'binding': str(output), 'sha256': sha(output), 'GPU_used': False,
            'endpoint_updates': 1613, 'endpoint_samples': 80000,
            'parity_passed_claim': False, 'base_weights_loaded': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=('L', 'A'), required=True)
    p.add_argument('--training-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    print(json.dumps(bind_endpoint(**vars(p.parse_args()))))


if __name__ == '__main__':
    main()
