"""Portable frozen B2000 loading, CPU artifact checks, and local RPC serving.

The original StageBranch / StageFlowBridge / StageInferenceView and action-LoRA
classes are imported unchanged from --runtime-root. No training lineage paths
are needed. This wrapper has CPU structural coverage, not new GPU parity proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

A_SHA='5892389863751f2245f83a6d3b96bf5725a052aa5b023230b093d8d8ece70ac4'
B_SHA='41988b92391953687b39a0bc5a35bce962fbfcd08fbf6e550108430850d9944f'
HF_REPO='XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365'
HF_REVISION='3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def configure_runtime(root):
    root=Path(root).resolve(strict=True)
    require((root/'continuous_model_v3/bridge.py').is_file(),'Missing unchanged runtime sources')
    inventory=Path(__file__).with_name('runtime-source-provenance.json')
    require(inventory.is_file(),'Missing runtime source provenance inventory')
    for item in json.loads(inventory.read_text()):
        rel=Path(item['original_relative_path'])
        require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe runtime source path')
        require(sha(root/rel)==item['sha256'],'Runtime source differs: '+str(rel))
    sys.path.insert(0,str(root))
    return root

def artifacts(adapter,branch):
    import torch
    require(sha(adapter)==A_SHA,'Expected exact completed A2000 adapter bytes')
    require(sha(branch)==B_SHA,'Expected exact completed B2000 branch bytes')
    a=torch.load(adapter,map_location='cpu',weights_only=True)
    b=torch.load(branch,map_location='cpu',weights_only=True)
    require(set(a)=={'metadata','tensors','sha256'},'Adapter schema differs')
    require(set(b)=={'identity','branch'},'Branch schema differs')
    meta=a['metadata'];ident=b['identity']
    require(meta['rank']==16 and meta['alpha']==16 and len(a['tensors'])==144,'Wrong A adapter geometry')
    require(ident['kind']=='continuous_branch_sidecar_v3' and ident['anchor']['adapter_file_sha256']==A_SHA,'Wrong B anchor')
    require(ident['anchor']['base_identity']==meta['base_identity'] and ident['anchor']['base_tensor_sha256']==meta['base_state_sha256'],'A/B base identity mismatch')
    require(ident['supervised'] is True and ident['retention_coefficient']==1.0 and ident['auxiliary_coefficient']==.05,'Wrong frozen B recipe')
    from route_reassessment_20261004.action_lora.action_lora import _payload_digest
    require(a['sha256']==_payload_digest(meta,a['tensors']),'Adapter internal payload checksum differs')
    expected={}
    for t in meta['targets']:
        expected[t['name']+'.lora_A']=(16,t['in_features'])
        expected[t['name']+'.lora_B']=(t['out_features'],16)
    require(set(expected)==set(a['tensors']),'Adapter target keys differ')
    for name,value in a['tensors'].items():
        require(tuple(value.shape)==expected[name] and value.dtype==torch.float32 and torch.isfinite(value).all(),'Invalid adapter '+name)
    from continuous_model_v3.branch import StageBranch
    branch_model=StageBranch(**ident['settings'])
    require(set(branch_model.state_dict())==set(b['branch']),'Branch tensor keys differ')
    for name,value in b['branch'].items():
        require(value.shape==branch_model.state_dict()[name].shape and value.dtype==torch.float32 and torch.isfinite(value).all(),'Invalid branch '+name)
    branch_model.load_state_dict(b['branch'],strict=True)
    require(any(v.ne(0).any() for k,v in b['branch'].items() if k.startswith('to_delta.')),'B modulation is all zero')
    return a,b,branch_model

def base_assets(source):
    """Same path-independent asset identity algorithm as training."""
    source=Path(source).resolve(strict=True);index=source/'model.safetensors.index.json'
    if index.exists():weights=set(json.loads(index.read_text())['weight_map'].values())
    else:weights={'model.safetensors'}
    metadata={p.name for p in source.iterdir() if p.is_file() and p.suffix in {'.py','.json','.jinja','.txt','.model'}}
    hashes={}
    for name in sorted(weights|metadata):
        require(Path(name).name==name,'Invalid model asset name')
        hashes[name]=sha(source/name)
    return hashes

def load_b2000(base_model,adapter,branch,*,runtime_root,device='cuda:0'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    configure_runtime(runtime_root)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    from transformers import AutoModel
    from route_reassessment_20261004.action_lora.action_lora import inject_action_lora,load_adapter
    from continuous_model_v3.bridge import StageFlowBridge,StageInferenceView
    a,b,branch_model=artifacts(adapter,branch);meta=a['metadata'];ident=b['identity']
    require(torch.device(device).type=='cuda','Published serving arithmetic requires CUDA BF16 autocast')
    assets=base_assets(base_model)
    base_id=hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()
    require(base_id==meta['base_identity'],'Base files differ from frozen training checkpoint')
    model=AutoModel.from_pretrained(str(base_model),trust_remote_code=True,attn_implementation='flash_attention_2',
                 dtype=torch.bfloat16,local_files_only=True).to(device).to(torch.bfloat16).eval()
    handle=inject_action_lora(model,base_identity=base_id,rank=meta['rank'],alpha=meta['alpha'],seed=meta['seed'])
    require(handle.metadata==meta,'Loaded base tensors/adapter configuration differ')
    loaded=load_adapter(handle,adapter);require(loaded['loaded_tensors']==144,'Missing adapter tensors')
    # Do not cast the full model after adapter injection: FP32 masters matter.
    handle.assert_integrity();branch_model=branch_model.to(device=device)
    bridge=StageFlowBridge(model,branch_model,supervised=ident['supervised'],
              retention_coefficient=ident['retention_coefficient'],auxiliary_coefficient=ident['auxiliary_coefficient'],
              confidence_threshold=ident['confidence_threshold'])
    bridge.eval();bridge._assert_trainable_contract()
    require(sha(adapter)==A_SHA and sha(branch)==B_SHA,'Artifacts changed while loading')
    return StageInferenceView(bridge)

def verify_cpu(adapter,branch):
    """Real frozen tensor validation plus synthetic original-hook structure check.

    This exercises real B weights with a toy base. It is intentionally not a
    claim of real-model action parity or rollout performance.
    """
    import torch
    from torch import nn
    from types import SimpleNamespace
    from continuous_model_v3.bridge import StageFlowBridge,StageInferenceView
    a,b,branch_model=artifacts(adapter,branch);settings=b['identity']['settings']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(711)
        ids=torch.tensor([[9,0,9,0,9,0,9,0,9,0,9,0,1]],dtype=torch.long)
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),video_grid_thw=torch.tensor([[2,2,2]]*3),
                    state=torch.zeros(1,4,60),action_mask=torch.cat([torch.ones(1,16,12),torch.zeros(1,16,48)],dim=-1))
        hidden=torch.randn(1,13,settings['hidden_dim'])
        class Context(nn.Module):
            def __init__(self):super().__init__();self.register_buffer('hidden',hidden)
            def forward(self,**kw):return SimpleNamespace(last_hidden_state=self.hidden)
        class FakeActor(nn.Module):
            def __init__(self):
                super().__init__();self.vlm=nn.Module();self.vlm.model=Context()
                self.t_projector=nn.Linear(1,6*settings['dit_dim'],bias=False)
                nn.init.zeros_(self.t_projector.weight)
                self.config=SimpleNamespace(state_dim=60,action_dim=60,state_length=4,
                     vlm_config=SimpleNamespace(video_token_id=9,vision_config=SimpleNamespace(spatial_merge_size=2)))
            def forward(self,**kw):
                self.vlm.model(**kw);action=torch.zeros(1,16,60)
                for _ in range(5):action=action+self.t_projector(torch.zeros(1,1,1)).reshape(1,-1)[:,:960].reshape(1,16,60)/5
                return SimpleNamespace(actions=action)
        base=FakeActor().eval();identity=b['identity']
        bridge=StageFlowBridge(base,branch_model,supervised=True,retention_coefficient=1.0,auxiliary_coefficient=.05,confidence_threshold=1.0).eval()
        view=StageInferenceView(bridge);before=torch.get_rng_state().clone();first=view(**inputs).actions
        require(torch.equal(before,torch.get_rng_state()),'Synthetic deterministic branch consumed RNG')
        second=view(**inputs).actions
        require(first.shape==(1,16,60) and torch.isfinite(first).all() and torch.equal(first,second),'Synthetic hook/action shape check failed')
        require(not base.t_projector._forward_hooks and not base.vlm.model._forward_hooks,'Temporary inference hooks leaked')
        try:view(**inputs,stage_target=torch.zeros(1,dtype=torch.long))
        except ValueError:pass
        else:raise ValueError('Oracle stage input was accepted')
    return dict(cpu_structural_validation_passed=True,adapter_sha256=A_SHA,branch_sha256=B_SHA,
                adapter_tensor_count=len(a['tensors']),adapter_parameters=sum(v.numel() for v in a['tensors'].values()),
                branch_tensor_count=len(b['branch']),branch_parameters=sum(v.numel() for v in b['branch'].values()),
                stage_settings=settings,synthetic_action_shape=[1,16,60],original_core_classes=True,
                one_vlm_five_euler_hook_contract=True,temporary_hooks_removed=True,oracle_inputs_rejected=True,
                gpu_action_parity_tested=False,full_model_rollout_tested=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','serve','infer-fixture'])
    p.add_argument('--runtime-root',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--branch',type=Path,required=True)
    p.add_argument('--base-model',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--port',type=int,default=18000)
    p.add_argument('--fixture',type=Path);p.add_argument('--seed',type=int,default=940001);p.add_argument('--output',type=Path)
    a=p.parse_args();configure_runtime(a.runtime_root)
    if a.mode=='verify':
        report=verify_cpu(a.adapter,a.branch)
        if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    require(a.base_model is not None,'--base-model required')
    model=load_b2000(a.base_model,a.adapter,a.branch,runtime_root=a.runtime_root,device=a.device)
    import torch
    from multiplex_inference.core import RngBank,SerialExecutor,official_forward
    if a.mode=='infer-fixture':
        from multiplex_inference.parity import load_request_fixture,tensor_hash
        require(a.fixture is not None and a.output is not None,'Need --fixture and --output')
        request=load_request_fixture(a.fixture,torch);bank=RngBank(torch);bank.seed(a.seed)
        action=official_forward(model,request,torch)
        torch.save({'actions':action,'seed':a.seed,'rng':bank.hashes(bank.capture())},a.output)
        print(json.dumps(dict(action_sha256=tensor_hash(action,torch),shape=list(action.shape),dtype=str(action.dtype))));return
    from multiplex_inference.server import checkpoint_identity,serve
    identity,_=checkpoint_identity(a.base_model)
    identity.update(loader_kind='portable_exact_frozen_B2000',adapter_sha256=A_SHA,branch_sha256=B_SHA,
                    base_hf_revision=HF_REVISION,stage_conditioning=True,original_euler_steps=5)
    executor=SerialExecutor(model,torch,identity,capacity=8)
    try:serve(executor,host='127.0.0.1',port=a.port,max_clients=8,stop=threading.Event(),ready_identity=executor.identity)
    finally:executor.close()

if __name__=='__main__':main()
