"""Strict frozen-A loading plus explicit branch sidecar; no HF full-model export."""
import hashlib
import json
from pathlib import Path

A_BINDING_SHA='fb47bbc98c4671f9afbed6d91eae0c3947badecf574070b5848df7686695d890'
A_ADAPTER_SHA='5892389863751f2245f83a6d3b96bf5725a052aa5b023230b093d8d8ece70ac4'


def load_frozen_a_bridge(binding_path, *, device, stage_count, supervised,
                         retention_coefficient=1.0,auxiliary_coefficient=.05,
                         confidence_threshold=1.0,hidden_mlp_dim=256,init_seed=2126104002):
    from lora_dev150_final_v1.loader import load_bound_model,require
    from .branch import StageBranch
    from .bridge import StageFlowBridge,StageInferenceView
    view,handle,wire,binding=load_bound_model(binding_path,A_BINDING_SHA,device=device)
    require(binding['arm']=='A' and binding['adapter']['sha256']==A_ADAPTER_SHA,'Only completed frozen A allowed')
    model=view.model
    branch=StageBranch(hidden_dim=model.config.vlm_config.text_config.hidden_size,
        dit_dim=model.config.dit_config.hidden_size,stage_count=stage_count,
        hidden_mlp_dim=hidden_mlp_dim,init_seed=init_seed).to(device=device)
    bridge=StageFlowBridge(model,branch,supervised=supervised,retention_coefficient=retention_coefficient,
        auxiliary_coefficient=auxiliary_coefficient,confidence_threshold=confidence_threshold)
    # Old handle.assert_integrity requires trainable LoRA and is intentionally
    # NOT called after freezing A. StageFlowBridge owns the stricter all-frozen guard.
    identity=dict(kind='continuous_frozen_A_anchor_v3',binding_sha256=A_BINDING_SHA,
        adapter_file_sha256=A_ADAPTER_SHA,base_identity=binding['base_identity'],
        base_tensor_sha256=binding['base_tensor_sha256'],wire_anchor=wire['lora_serving_identity'])
    bridge._verified_a_anchor=identity
    bridge._verified_a_binding=binding
    return bridge,StageInferenceView(bridge),identity,binding


def branch_identity(bridge,anchor,*,vocabulary_sha256,training_data_sha256):
    for v in (vocabulary_sha256,training_data_sha256):
        if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('Pin frozen vocabulary and training plan SHA256')
    return dict(kind='continuous_branch_sidecar_v3',anchor=anchor,settings=bridge.branch.settings,
        supervised=bridge.supervised,retention_coefficient=bridge.retention_coefficient,
        auxiliary_coefficient=bridge.auxiliary_coefficient,confidence_threshold=bridge.confidence_threshold,
        auxiliary_normalization='sum_admitted_CE_divided_by_all_windows',
        teacher='completed_frozen_A_with_LoRA_enabled',vocabulary_sha256=vocabulary_sha256,
        training_data_sha256=training_data_sha256)


def export_branch(path,bridge,identity):
    import os
    import torch
    bridge._assert_trainable_contract()
    path=Path(path)
    payload={'identity':identity,'branch':{n:v.detach().cpu().clone() for n,v in bridge.branch.state_dict().items()}}
    # An exclusive destination prevents overwriting either old A or a sidecar.
    # A successful read is verified against identity+all finite tensor keys.
    with path.open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_branch(path,bridge,identity,*,expected_sha256):
    import torch
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Branch sidecar bytes changed')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    expected=bridge.branch.state_dict()
    if set(payload)!={'identity','branch'} or payload['identity']!=identity or set(payload['branch'])!=set(expected):
        raise ValueError('Branch identity or tensor key mismatch')
    for name,tensor in payload['branch'].items():
        if tensor.shape!=expected[name].shape or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('Invalid branch tensor: '+name)
    bridge.branch.load_state_dict(payload['branch'],strict=True)
    bridge._assert_trainable_contract()
    return {'loaded_tensors':len(expected),'sha256':expected_sha256}


def clone_verified_a_bridge(source_bridge,anchor,binding,*,device,supervised):
    """Clone one verified A only, preserving mixed dtypes and independent storage.

    Every original parameter/buffer is compared on the source device, one tensor
    at a time. No repeated checkpoint hashing or 10GB comparison allocation.
    Fresh branch initialization must equal the not-yet-trained source branch.
    Returns the same four values as load_frozen_a_bridge; bridge.clone_proof is
    additional startup evidence. A failure occurs before any optimizer update.
    """
    import copy
    import torch
    from .branch import StageBranch,require
    from .bridge import StageFlowBridge,StageInferenceView
    source_bridge._assert_trainable_contract()
    require(getattr(source_bridge,'_verified_a_anchor',None)==anchor and
            getattr(source_bridge,'_verified_a_binding',None)==binding,
            'Clone source did not originate from strict completed-A loading')
    require(anchor['binding_sha256']==A_BINDING_SHA and anchor['adapter_file_sha256']==A_ADAPTER_SHA,
            'Only the verified completed A may be cloned')
    require(type(supervised) is bool,'Explicit C/U supervision required')
    source=source_bridge.model
    model=copy.deepcopy(source).to(device=device)  # NEVER cast the mixed A dtypes.
    require(not ({id(m) for m in source.modules()} & {id(m) for m in model.modules()}),
            'Cloned model shares module objects with its source')
    left,right=source.state_dict(),model.state_dict()
    require(list(left)==list(right),'Clone state names/order changed')
    count=0
    for name,value in left.items():
        other=right[name]
        require(other.shape==value.shape and other.dtype==value.dtype and other.device==torch.device(device),
                'Clone geometry/dtype/device changed: '+name)
        if value.numel():
            require((value.device,value.data_ptr())!=(other.device,other.data_ptr()),'Clone shares source storage: '+name)
        require(torch.equal(value,other.to(value.device)),'Clone changed a frozen A tensor: '+name)
        count+=value.numel()
    branch=StageBranch(**source_bridge.branch.settings).to(device=device)
    for name,value in source_bridge.branch.state_dict().items():
        require(torch.equal(value,branch.state_dict()[name].to(value.device)),
                'Clone initialization requires an untouched fresh source branch')
    bridge=StageFlowBridge(model,branch,supervised=supervised,
        retention_coefficient=source_bridge.retention_coefficient,
        auxiliary_coefficient=source_bridge.auxiliary_coefficient,
        confidence_threshold=source_bridge.confidence_threshold)
    bridge._verified_a_anchor=copy.deepcopy(anchor);bridge._verified_a_binding=copy.deepcopy(binding)
    bridge.clone_proof=dict(source_device=str(next(source.parameters()).device),target_device=str(torch.device(device)),
        tensor_count=len(left),tensor_elements=count,every_tensor_exact=True,mixed_dtypes_preserved=True,
        independent_modules_and_storage=True,fresh_branch_exact=True,anchor_adapter_sha256=A_ADAPTER_SHA)
    source_bridge._assert_trainable_contract();bridge._assert_trainable_contract()
    return bridge,StageInferenceView(bridge),bridge._verified_a_anchor,bridge._verified_a_binding
