"""Bind the actual 1613-update / 80000-window endpoint; never merge LoRA."""
import hashlib
import json
from pathlib import Path


SERVING_SOURCES = (
    'lora_dev150_final_v1/loader.py', 'lora_dev150_final_v1/bind_endpoint.py',
    'lora_dev150_final_v1/server.py', 'lora_dev150_final_v1/client.py',
    'route_reassessment_20261004/action_lora/action_lora.py',
    'multiplex_inference/core.py', 'multiplex_inference/server.py',
    'multiplex_inference/client.py', 'multiplex_inference/parity.py',
    'training_bridge/train_single.py',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def pinned(item):
    require(isinstance(item, dict) and item.get('path') and item.get('sha256'), 'Unbound source artifact')
    require(sha(item['path']) == item['sha256'], 'Changed artifact: ' + item['path'])
    return json.loads(Path(item['path']).read_text())


def read_binding(path, expected_sha):
    """CPU/file-only recheck. The completed audit is evidence, not a GPU parity claim."""
    require(sha(path) == expected_sha, 'Endpoint binding changed')
    b = json.loads(Path(path).read_text())
    require(b.get('schema') == 'lora_dev150_final_endpoint_v1' and b.get('ready') is True,
            'Actual final endpoint must be bound before loading')
    arm = b.get('arm'); require(arm in ('L', 'A'), 'Only completed L/A endpoints supported')
    require((b.get('endpoint_updates'), b.get('endpoint_samples'), b.get('global_batch')) == (1613, 80000, 128),
            'Not the actual completed endpoint')
    root = Path(b['root']).resolve()
    audit = pinned(b['completion_audit'])
    require(audit.get('kind') == 'completed_eight_gpu_artifact_audit_v1'
            and audit.get('passed') is True and audit.get('training_only') is True
            and audit.get('evaluation_performed') is False and audit.get('success_rate_computed') is False,
            'Need complete training artifact audit, not a model-success claim')
    audited = {}
    for item in audit['verified_files']:
        name = str(Path(item['path']).resolve())
        require(name not in audited or audited[name] == item['sha256'], 'Conflicting audit file records')
        audited[name] = item['sha256']
    checked = {}

    def artifact(item, *, must_be_audited=True):
        require(isinstance(item, dict) and item.get('path') and item.get('sha256'), 'Missing artifact pin')
        p = Path(item['path']).resolve(); name = str(p)
        if must_be_audited:
            require(audited.get(name) == item['sha256'], 'Artifact is not the audited bytes: ' + name)
        if name not in checked:
            checked[name] = sha(p)
        require(checked[name] == item['sha256'], 'Artifact changed since completion audit: ' + name)
        return p

    def record(key):
        return json.loads(artifact(b[key]).read_text())

    complete = record('completion'); identity = record('run_identity'); migration = record('migration')
    config = record('training_config'); config128 = record('batch128_config')
    config64 = record('batch64_config'); parent = record('parent64_identity')
    parent_migration = record('parent64_migration'); original = record('original_identity')
    report = audit['arms'][arm]
    require(report.get('passed') is True and report['arm'] == arm
            and report['final_optimizer_step'] == 1613 and report['final_samples'] == 80000
            and report['adapter_tensor_count'] == 144 and report['adapter_parameters'] == 3538944
            and report['adapter_equals_checkpoint'] is True and report['all_AdamW_parameter_steps'] == 1613
            and report['source_counts'] == {'human': 80000, 'mg': 0}, 'Arm audit is incomplete')
    require(complete.get('completed') is True and complete['arm'] == arm
            and (complete['updates'], complete['samples'], complete['global_batch'], complete['target_samples'])
            == (1613, 80000, 128, 80000), 'Completion is not the fixed final endpoint')
    require(complete['identity_sha256'] == b['run_identity']['sha256']
            and complete['migration_sha256'] == b['migration']['sha256']
            and identity['config_sha256'] == b['batch128_config']['sha256']
            and b['training_config_sha256'] == b['training_config']['sha256']
            == original['config_sha256'], 'Configuration/completion identity mismatch')
    require(parent['config_sha256'] == b['batch64_config']['sha256']
            and parent == identity['parent_execution_identity']
            and original == parent['parent_numerical_identity'] == identity['parent_numerical_identity'],
            'Original numerical identity or parent execution changed')
    for ident in (identity, parent, original):
        require(ident['arm'] == arm and ident['base_identity'] == b['base_identity']
                and ident['base_tensor_sha256'] == b['base_tensor_sha256'], 'Wrong arm/base in lineage')
        require(isinstance(ident['source_sha256'], dict) and ident['source_sha256'], 'Missing training source pins')
        for name, digest in ident['source_sha256'].items():
            p = root / name
            require(not Path(name).is_absolute() and p.resolve().is_relative_to(root), 'Invalid relative training source')
            artifact({'path': str(p), 'sha256': digest})
    require(config['model'] == b['base_model'] and config['lora'] == b['lora']
            and b['lora']['rank'] == 16 and b['lora']['alpha'] == 16 and type(b['lora']['seed']) is int,
            'Wrong original base or LoRA configuration')
    require(config['retention_coefficients'][arm] == report['retention_coefficient']
            and original['sampling_config_sha256'] == config['sampling_config_sha256']
            and config128['sampling_plan_sha256'] == original['source_sha256']['lora_retention_run_v1/sample_plan.jsonl']
            and config128['learning_rate'] == config['optimizer']['lr'], 'Numerical recipe changed')
    require(config128['global_batch'] == 128 and config128['target_samples'] == 80000
            and config128['gpu_replicas_per_arm'] == 4 and config128['microbatch_per_device'] == 32
            and config64['global_batch'] == 64 and config64['target_samples'] == 80000, 'Wrong execution configurations')
    ex = identity['execution']; pex = parent['execution']
    require(ex['kind'] == 'four_replica_global128_v1' and ex['global_batch'] == 128
            and ex['microbatch_per_device'] == 32 and ex['target_samples'] == 80000
            and ex['per_sample_equal_weight'] is True and ex['optimizer_recipe_unchanged'] is True
            and ex['sampling_unchanged'] is True and ex['stop_after_updates'] is None,
            'Endpoint was not the completed prescribed execution')
    require(pex['kind'] == 'two_replica_global64_v1' and pex['global_batch'] == 64, 'Wrong parent execution')
    require(migration['strict_old_identity_verified'] is True and migration['parent_global_batch'] == 64
            and (migration['restored_step'], migration['restored_samples']) == (1100, 14400)
            and migration['execution'] == ex and parent_migration['strict_old_identity_verified'] is True
            and (parent_migration['restored_step'], parent_migration['restored_samples']) == (1000, 8000)
            and parent_migration['execution'] == pex, 'Invalid migration boundaries')
    require(ex['parent_run_identity_sha256'] == b['parent64_identity']['sha256']
            and pex['parent_run_identity_sha256'] == b['original_identity']['sha256'], 'Parent identity SHA mismatch')
    lineage = report['lineage']
    require(len(lineage) == 3 and lineage[0]['kind'] == 'original_batch8'
            and (lineage[0]['steps'], lineage[0]['samples']) == (1000, 8000)
            and lineage[1]['kind'] == 'batch64'
            and (lineage[1]['additional_steps'], lineage[1]['additional_samples']) == (100, 6400)
            and lineage[2] == {'kind': 'batch128_plus_final64', 'additional_steps': 513, 'additional_samples': 65600},
            'Wrong audited optimizer/sample lineage')
    for item, mig, execution in ((lineage[1]['checkpoint'], migration, ex),
                                 (lineage[0]['checkpoint'], parent_migration, pex)):
        p = artifact(item)
        require(p == Path(mig['parent_checkpoint']).resolve()
                and item['sha256'] == mig['parent_checkpoint_sha256'] == execution['parent_checkpoint_sha256'],
                'Parent checkpoint changed')
    require(complete['parent_checkpoint_sha256'] == ex['parent_checkpoint_sha256'], 'Completion parent differs')
    for key in ('adapter', 'checkpoint'):
        require(Path(b[key]['path']).resolve() == Path(report[key]['path']).resolve()
                and b[key]['sha256'] == report[key]['sha256'], 'Final artifact differs from arm audit: ' + key)
        artifact(b[key])
    require(Path(b['adapter']['path']).resolve() == Path(complete['adapter']['path']).resolve()
            and b['adapter']['sha256'] == complete['adapter_file_sha256']
            and complete['adapter']['base_state_sha256'] == b['base_tensor_sha256'], 'Adapter does not match completion')
    evidence = report['base_evidence']
    require(evidence['base_identity'] == b['base_identity'] and evidence['base_state_sha256'] == b['base_tensor_sha256']
            and evidence['artifact_metadata_agrees'] is True and evidence['save_code_checks_full_original_tensor_fingerprint'] is True,
            'Missing frozen-base evidence')
    m = report['metrics']
    expected_metrics = {'optimizer_updates': 513, 'full128_updates': 512, 'terminal64_updates': 1,
                        'first_step': 1101, 'last_step': 1613, 'first_slot': 14400,
                        'last_slot': 79999, 'samples_each_once': 65600}
    require(all(m[k] == v for k, v in expected_metrics.items()) and m['every_step_RNG_unchanged'] is True
            and m['final_checkpoint_master_rng_matches_terminal_update'] is True, 'Incomplete endpoint coverage audit')
    require(b['inference_arithmetic'] == 'full_forward_cuda_bfloat16_autocast_fp32_adapter_master',
            'Unsupported inference arithmetic')
    expected_sources = {str((root / name).resolve()) for name in SERVING_SOURCES}
    require(set(b['source_sha256']) == expected_sources, 'Missing/extra serving source pins')
    for name, digest in b['source_sha256'].items():
        artifact({'path': name, 'sha256': digest}, must_be_audited=False)
    require(Path(b['xr1_repo']).resolve() == root / 'code/Xiaomi-Robotics-1', 'Wrong released implementation directory')
    return b


class AutocastInferenceView:
    """Same loaded model with explicit BF16 autocast; FP32 adapter masters retained."""
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        require(model.device.type == 'cuda', 'This serving precision policy requires CUDA')
    @property
    def device(self): return self.model.device
    @property
    def dtype(self): return self.model.dtype
    @property
    def config(self): return self.model.config
    def __call__(self, **inputs):
        with self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            return self.model(**inputs)


def load_bound_model(binding_path, binding_sha256, *, device='cuda:0'):
    """Potentially expensive: called only by explicit GPU service/parity command."""
    b = read_binding(binding_path, binding_sha256)
    import torch
    from transformers import AutoModel
    from training_bridge.train_single import checkpoint_identity as training_identity
    from multiplex_inference.server import checkpoint_identity as serving_identity
    from route_reassessment_20261004.action_lora.action_lora import inject_action_lora, load_adapter
    assets = training_identity(b['base_model'])
    # Exactly the json spelling used in the actual training driver.
    base_id = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    require(base_id == b['base_identity'], 'Released base assets differ from training')
    wire_identity, stamps = serving_identity(b['base_model'])
    model = AutoModel.from_pretrained(b['base_model'], trust_remote_code=True,
                attn_implementation='flash_attention_2', dtype=torch.bfloat16,
                local_files_only=True).to(device).to(torch.bfloat16).eval()
    handle = inject_action_lora(model, base_identity=base_id, **b['lora'])
    require(handle.metadata['base_state_sha256'] == b['base_tensor_sha256'], 'Actual loaded base tensors differ')
    loaded = load_adapter(handle, b['adapter']['path'])
    require(loaded['loaded_tensors'] == 144, 'Require all144 action-LoRA tensors')
    b_tensors = [p for name, p in model.named_parameters() if name in handle.adapter_names and name.endswith('.lora_B')]
    require(len(b_tensors) == 72, 'Require all72 LoRA B tensors')
    b_nonzero = int(torch.stack([p.detach().ne(0).any() for p in b_tensors]).sum().item())
    require(b_nonzero > 0, 'Completed adapter has no nonzero LoRA B tensor')
    model.eval()
    # Never model.to(BF16) after injection: this would truncate FP32 adapters.
    handle.assert_integrity()
    require(stamps == {p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in stamps},
            'Base files changed during model load')
    require(sha(b['adapter']['path']) == b['adapter']['sha256'], 'Adapter changed during load')
    lora_identity = {'arm':b['arm'], 'binding_sha256':binding_sha256,
                    'adapter_file_sha256':b['adapter']['sha256'],
                    'adapter_payload_sha256':loaded['payload_sha256'],
                    'base_identity':base_id,'base_tensor_sha256':b['base_tensor_sha256'],
                    'endpoint_updates':1613,'endpoint_samples':80000,'global_batch':128,
                    'completion_audit_sha256':b['completion_audit']['sha256'],
                    'lora_B_nonzero_tensors':b_nonzero,'lora_B_tensor_count':72,'lora':b['lora'],
                    'inference_arithmetic':b['inference_arithmetic'],
                    'source_sha256':b['source_sha256']}
    wire_identity.update(lora_serving_identity=lora_identity,
                         official_server_sha256=sha(Path(b['xr1_repo'])/'deploy/server.py'),
                         loader_kind='explicit_original_base_then_action_lora_inject_load')
    return AutocastInferenceView(model, torch), handle, wire_identity, b
