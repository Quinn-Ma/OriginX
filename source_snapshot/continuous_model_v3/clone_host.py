"""CPU-staged exact clone, avoiding direct peer transfers on this host."""
from .loader import A_BINDING_SHA,A_ADAPTER_SHA
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
    model=copy.deepcopy(source).to(device='cpu').to(device=device)  # NEVER cast the mixed A dtypes.
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
        require(torch.equal(value.detach().cpu(),other.detach().cpu()),'Clone changed a frozen A tensor: '+name)
        count+=value.numel()
    branch=StageBranch(**source_bridge.branch.settings).to(device=device)
    for name,value in source_bridge.branch.state_dict().items():
        require(torch.equal(value.detach().cpu(),branch.state_dict()[name].detach().cpu()),
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
