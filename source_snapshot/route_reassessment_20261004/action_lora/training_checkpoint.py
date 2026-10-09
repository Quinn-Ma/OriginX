"""Adapter-only full AdamW training checkpoints; never load base state_dict.

Provider contract: state_dict/load_state_dict round-trip primitive containers
and torch tensors. Call only at a completed update after zero_grad(set_to_none=True).
Single-process, single-device. Source/config identities are mandatory SHA256 pins.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

try:
    from .action_lora import _tensor_digest
except ImportError:
    from action_lora import _tensor_digest


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    if value is None or type(value) in (bool, str, int, float):
        return value
    raise ValueError(f"Unsupported checkpoint value: {type(value).__name__}")


def _digest(value):
    result = hashlib.sha256()
    def walk(item):
        if isinstance(item, torch.Tensor):
            result.update(b'tensor' + _tensor_digest([('', item)]).encode())
        elif isinstance(item, dict):
            result.update(b'dict[')
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                walk(key); walk(item[key])
            result.update(b']')
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode() + b'[')
            for child in item: walk(child)
            result.update(b']')
        elif item is None or type(item) in (bool, str, int, float):
            result.update(type(item).__name__.encode() + json.dumps(item, allow_nan=False).encode() + b';')
        else:
            raise ValueError('Unsupported payload type')
    walk(value)
    return result.hexdigest()


def _identity(identity):
    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if not isinstance(identity, dict) or not sha(identity.get('config_sha256')):
        raise ValueError('identity.config_sha256 is required')
    sources = identity.get('source_sha256')
    if not isinstance(sources, dict) or not sources or not all(isinstance(k, str) and sha(v) for k, v in sources.items()):
        raise ValueError('identity.source_sha256 must contain exact source pins')
    return _clone(identity)


def _optimizer_recipe(handle, optimizer):
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError('This checkpoint version supports plain torch.optim.AdamW only')
    names = {id(value): name for name, value in handle.model.named_parameters() if name in handle.adapter_names}
    groups, observed = [], []
    for group in optimizer.param_groups:
        ids = [id(value) for value in group['params']]
        if any(item not in names for item in ids):
            raise ValueError('Optimizer contains original or unknown parameters')
        observed += ids
        groups.append({key: ([names[item] for item in ids] if key == 'params' else _clone(value))
                       for key, value in group.items()})
    if len(observed) != len(set(observed)) or set(observed) != set(names):
        raise ValueError('Optimizer must contain every adapter exactly once')
    return {'class': 'torch.optim.AdamW', 'groups': groups}


def _rng(device):
    device = torch.device(device)
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': {'name': state[0], 'keys': torch.from_numpy(state[1].astype(np.int64)),
                      'position': state[2], 'has_gauss': state[3], 'cached_gaussian': state[4]},
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'device_type': device.type}


def _numpy_state(state):
    return (state['name'], state['keys'].cpu().numpy().astype(np.uint32),
            state['position'], state['has_gauss'], state['cached_gaussian'])


def _validate_rng(state, device):
    if set(state) != {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'device_type'}:
        raise ValueError('Invalid RNG schema')
    device = torch.device(device)
    if state['device_type'] != device.type or device.type not in ('cpu', 'cuda'):
        raise ValueError('Checkpoint RNG device type differs')
    random.Random().setstate(state['python'])
    np.random.RandomState(0).set_state(_numpy_state(state['numpy']))
    torch.Generator(device='cpu').set_state(state['torch_cpu'])
    if device.type == 'cuda':
        torch.Generator(device=device).set_state(state['torch_cuda'])
    elif state['torch_cuda'] is not None:
        raise ValueError('Unexpected CUDA RNG in CPU checkpoint')


def _restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(_numpy_state(state['numpy']))
    torch.set_rng_state(state['torch_cpu'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state(state['torch_cuda'], device)


def _metadata(handle, optimizer, provider, identity, device, accumulate, max_grad_norm):
    handle.assert_integrity()
    if type(accumulate) is not int or accumulate < 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError('Invalid accumulation/clip configuration')
    if any(p.grad is not None for p in handle.model.parameters()):
        raise ValueError('Checkpoint only after complete update and zero_grad(set_to_none=True)')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU/CUDA single-device checkpoints are supported')
    if device.type == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    if any(p.device != device for p in handle.model.parameters() if p.requires_grad):
        raise ValueError('Adapter/device mismatch')
    return {'format': 'xr1-lora-training-v1', 'adapter': _clone(handle.metadata),
            'identity': _identity(identity), 'optimizer_recipe': _optimizer_recipe(handle, optimizer),
            'provider_class': f'{type(provider).__module__}.{type(provider).__qualname__}',
            'device_type': device.type, 'accumulate': accumulate, 'max_grad_norm': float(max_grad_norm)}


def _validate_optimizer(saved, handle, optimizer, step):
    current = optimizer.state_dict()
    if set(saved) != {'state', 'param_groups'} or saved['param_groups'] != current['param_groups']:
        raise ValueError('AdamW parameter groups/config differ')
    ids = [item for group in saved['param_groups'] for item in group['params']]
    parameters = [item for group in optimizer.param_groups for item in group['params']]
    if set(saved['state']) - set(ids) or (step > 0 and set(saved['state']) != set(ids)):
        raise ValueError('AdamW state parameter set differs')
    for group in saved['param_groups']:
        for item in group['params']:
            if item not in saved['state']: continue
            values = saved['state'][item]
            expected = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if group['amsgrad'] else set())
            if set(values) != expected:
                raise ValueError('Invalid AdamW state schema')
            parameter = parameters[ids.index(item)]
            for key, value in values.items():
                shape = () if key == 'step' else parameter.shape
                if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid AdamW state tensor')
                if key != 'step' and value.dtype != torch.float32:
                    raise ValueError('AdamW adapter moments must be FP32')
            if values['step'].item() != step:
                raise ValueError('AdamW per-parameter update count differs')


def save_training_checkpoint(path, *, handle, optimizer, provider, step, identity, device,
                             accumulate=1, max_grad_norm=1.0):
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    metadata = _metadata(handle, optimizer, provider, identity, device, accumulate, max_grad_norm)
    state = handle.model.state_dict()
    payload = {'metadata': metadata, 'step': step,
               'adapters': {name: state[name].detach().cpu().clone() for name in sorted(handle.adapter_names)},
               'optimizer': _clone(optimizer.state_dict()), 'provider': _clone(provider.state_dict()),
               'rng': _clone(_rng(device))}
    _validate_optimizer(payload['optimizer'], handle, optimizer, step)
    for value in payload['adapters'].values():
        if not torch.isfinite(value).all(): raise ValueError('Nonfinite adapter')
    envelope = {'payload': payload, 'sha256': _digest(payload)}
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.lora-checkpoint-', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open('wb') as stream:
            torch.save(envelope, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(path), 'step': step, 'payload_sha256': envelope['sha256']}


def load_training_checkpoint(path, *, handle, optimizer, provider, identity, device,
                             accumulate=1, max_grad_norm=1.0):
    expected = _metadata(handle, optimizer, provider, identity, device, accumulate, max_grad_norm)
    envelope = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(envelope, dict) or set(envelope) != {'payload', 'sha256'}:
        raise ValueError('Invalid training checkpoint envelope')
    payload = envelope['payload']
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'step', 'adapters', 'optimizer', 'provider', 'rng'}:
        raise ValueError('Invalid training checkpoint schema')
    if _digest(payload) != envelope['sha256']:
        raise ValueError('Training checkpoint content hash differs')
    if payload['metadata'] != expected:
        raise ValueError('Base/config/source/optimizer/provider identity differs')
    step = payload['step']
    if type(step) is not int or step < 0: raise ValueError('Invalid update count')
    state = handle.model.state_dict()
    if not isinstance(payload['adapters'], dict) or set(payload['adapters']) != handle.adapter_names:
        raise ValueError('Adapter names differ')
    for name, value in payload['adapters'].items():
        if (not isinstance(value, torch.Tensor) or value.shape != state[name].shape or
                value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('Adapter shape/dtype/value differs')
    _validate_optimizer(payload['optimizer'], handle, optimizer, step)
    _validate_rng(payload['rng'], device)
    # All immutable identities/content/tensor schemas validated before mutation.
    # Providers may own file handles and must not be deepcopied. If their own
    # load rejects cursor state, restore the previous provider state immediately.
    previous_provider = _clone(provider.state_dict())
    previous_rng = _rng(device)
    previous_optimizer = copy.deepcopy(optimizer.state_dict())
    previous_adapters = {name: state[name].detach().clone() for name in handle.adapter_names}
    try:
        provider.load_state_dict(payload['provider'])
        optimizer.load_state_dict(payload['optimizer'])
        with torch.no_grad():
            for name, value in payload['adapters'].items(): state[name].copy_(value)
        _restore_rng(payload['rng'], device)  # Restore last, after all loads.
    except Exception:
        provider.load_state_dict(previous_provider)
        optimizer.load_state_dict(previous_optimizer)
        with torch.no_grad():
            for name, value in previous_adapters.items(): state[name].copy_(value)
        _restore_rng(previous_rng, device)
        raise
    return step
