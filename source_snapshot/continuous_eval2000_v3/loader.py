"""Completed100-update C/U branch sidecars over the exact frozen A, never old-LoRA-only admission."""
import hashlib
import json
from pathlib import Path

A_BINDING_SHA='fb47bbc98c4671f9afbed6d91eae0c3947badecf574070b5848df7686695d890'
A_ADAPTER_SHA='5892389863751f2245f83a6d3b96bf5725a052aa5b023230b093d8d8ece70ac4'
RUNTIME_FILES=('continuous_eval2000_v3/__init__.py','continuous_eval2000_v3/bind_endpoint.py','continuous_eval2000_v3/loader.py','continuous_eval2000_v3/server.py','continuous_eval2000_v3/client.py','continuous_eval2000_v3/telemetry.py','continuous_eval2000_v3/parity.py',
 'continuous_model_v3/__init__.py','continuous_model_v3/loader.py','continuous_model_v3/bridge.py','continuous_model_v3/branch.py',
 'continuous_model_v3/observed.py','training_bridge/__init__.py','training_bridge/bridge.py','training_bridge/layout.py','training_bridge/train_core.py',
 'lora_parallel_v1/__init__.py','lora_parallel_v1/batched_bridge.py','teacher_anchor_pilot_v1/__init__.py','teacher_anchor_pilot_v1/bridge.py',
 'route_reassessment_20261004/action_lora/lora_flow_bridge.py','route_reassessment_20261004/action_lora/action_lora.py')


def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def pinned(item):
    require(isinstance(item,dict) and item.get('path') and item.get('sha256') and sha(item['path'])==item['sha256'],'Unbound/changed artifact')
    return json.loads(Path(item['path']).read_text())

def read_binding(path,expected_sha256):
    require(sha(path)==expected_sha256,'Stage endpoint binding bytes changed')
    b=json.loads(Path(path).read_text());arm=b.get('arm')
    require(b.get('schema')=='continuous_eval2000_v3_endpoint_v1' and b.get('ready') is True and arm in ('B',),'Real completed C/U stage endpoint required')
    require(b.get('endpoint_updates')==2000 and b.get('endpoint_samples')==256000,'Only the fixed100-update pilot endpoint is admitted')
    require(b['base_A_binding']['sha256']==A_BINDING_SHA,'Wrong A parent')
    base=pinned(b['base_A_binding']);require(base['arm']=='A' and base['adapter']['sha256']==A_ADAPTER_SHA,'Frozen A identity changed')
    complete=pinned(b['completion']);run=pinned(b['run_identity'])
    require(complete.get('completed') is True and complete.get('arm')==arm and (complete.get('updates'),complete.get('samples'))==(2000,256000),'Training has not completed the fixed endpoint')
    require(complete['branch_file_sha256']==b['branch']['sha256']==sha(b['branch']['path']) and complete['branch']==b['branch'] and
            complete['identity_sha256']==b['run_identity']['sha256'] and run['arm']==arm,'Completion, branch or training identity differs')
    ident=b['branch_identity'];require(ident==complete['branch_identity'] and ident['kind']=='continuous_branch_sidecar_v3','Wrong exported branch identity')
    require(ident['anchor']['binding_sha256']==A_BINDING_SHA and ident['anchor']['adapter_file_sha256']==A_ADAPTER_SHA and
            ident['anchor']['base_identity']==base['base_identity'] and ident['anchor']['base_tensor_sha256']==base['base_tensor_sha256'],'Sidecar changed frozen A')
    require(ident['supervised'] is (arm=='B') and ident['retention_coefficient']==1.0 and ident['auxiliary_coefficient']==.05 and
            ident['confidence_threshold']==1.0 and ident['teacher']=='completed_frozen_A_with_LoRA_enabled' and
            ident['auxiliary_normalization']=='sum_admitted_CE_divided_by_all_windows','C/U recipe or frozen-A teacher differs')
    for key in ('vocabulary_sha256','training_data_sha256'):
        value=ident[key];require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Missing train-only vocabulary/data pin')
    require(ident['settings']['hidden_dim']==2560 and ident['settings']['dit_dim']==1024 and ident['settings']['hidden_mlp_dim']==256 and
            type(ident['settings']['stage_count']) is int and ident['settings']['stage_count']>=2,'Wrong online branch geometry')
    require(b['inference_arithmetic']=='frozen_A_bf16_forward_fp32_stage_master_autocast' and b['online_stage_conditioning'] is True and b['no_label_actor_inputs'] is True,'Stage actor path is not explicit')
    root=Path(b['root']).resolve();required={str(root/f) for f in RUNTIME_FILES}
    require(required<=set(b['source_sha256']),'Missing runtime/model transitive source pins')
    for name,digest in b['source_sha256'].items():
        require(Path(name).is_absolute() and Path(name).resolve().is_relative_to(root) and sha(name)==digest,'Stage execution source changed: '+name)
    return b


def check_stage_hello(actual,expected):
    require(bool(expected.get('stage_serving_identity')) and actual.get('stage_serving_identity')==expected['stage_serving_identity'] and
            actual.get('loader_kind')==expected.get('loader_kind')=='completed_frozen_A_plus_online_stage_branch_v2','Stage endpoint identity differs')


class TrackedStageView:
    """The model agent's real one-VLM/five-Euler view; no extra forward or RNG."""
    def __init__(self,view,bridge):
        self.view,self.bridge=view,bridge;self.forward_calls=0
        self.versions=self.snapshot()
    def snapshot(self):return tuple((n,id(p),p._version,tuple(p.shape),p.dtype,p.device) for n,p in self.bridge.branch.named_parameters())
    @property
    def device(self):return self.view.device
    @property
    def dtype(self):return self.view.dtype
    @property
    def config(self):return self.view.config
    def __call__(self,**inputs):
        require(self.snapshot()==self.versions,'Stage weights changed before inference')
        captures=[]
        def observed(_module,_args,output):
            delta,logits=output
            captures.append((delta.detach().float().cpu(),logits.detach().float().cpu()))
        hook=self.bridge.branch.register_forward_hook(observed)
        try:result=self.view(**inputs)
        finally:hook.remove()
        require(self.snapshot()==self.versions,'Stage weights changed during inference')
        require(len(captures)==1,'Expected exactly one observed stage prediction')
        delta,logits=captures[0]
        require(len(logits)==1,'Evaluation telemetry requires batch1')
        probabilities=logits.softmax(dim=-1)[0]
        self.last_stage_trace=dict(logits=logits[0].tolist(),probabilities=probabilities.tolist(),
            stage_index=int(probabilities.argmax()),confidence=float(probabilities.max()),
            delta_l2=float(delta.norm()),delta_absmax=float(delta.abs().max()),
            observed_branch_calls=1,extra_model_calls=0,extra_random_draws=0)
        self.forward_calls+=1;return result


def load_bound_model(path,expected_sha256,*,device='cuda:0'):
    b=read_binding(path,expected_sha256)
    import torch
    import continuous_model_v3.loader as model_loader
    from multiplex_inference.server import checkpoint_identity
    root=Path(b['root']).resolve()
    require(Path(model_loader.__file__).resolve()==root/'continuous_model_v3/loader.py','Unexpected stage model module')
    ident=b['branch_identity'];settings=ident['settings']
    bridge,view,anchor,base=model_loader.load_frozen_a_bridge(b['base_A_binding']['path'],device=device,
        stage_count=settings['stage_count'],supervised=ident['supervised'],retention_coefficient=ident['retention_coefficient'],
        auxiliary_coefficient=ident['auxiliary_coefficient'],confidence_threshold=ident['confidence_threshold'],
        hidden_mlp_dim=settings['hidden_mlp_dim'],init_seed=settings['init_seed'])
    require(anchor==ident['anchor'],'Actual A anchor differs from sidecar')
    actual=model_loader.branch_identity(bridge,anchor,vocabulary_sha256=ident['vocabulary_sha256'],training_data_sha256=ident['training_data_sha256'])
    require(actual==ident,'Actual model branch identity differs')
    loaded=model_loader.load_branch(b['branch']['path'],bridge,ident,expected_sha256=b['branch']['sha256'])
    bridge.eval();bridge._assert_trainable_contract()
    nonzero=sum(int(p.detach().ne(0).any().item()) for n,p in bridge.branch.named_parameters() if n.startswith('to_delta.'))
    require(nonzero>0,'Trained online stage modulation is still identically zero')
    wire,_=checkpoint_identity(base['base_model'])
    specific=dict(arm=b['arm'],binding_sha256=expected_sha256,branch_file_sha256=b['branch']['sha256'],branch_identity=ident,
        loaded_branch_tensors=loaded['loaded_tensors'],completion_sha256=b['completion']['sha256'],run_identity_sha256=b['run_identity']['sha256'],
        endpoint_updates=2000,endpoint_samples=256000,anchor=anchor,source_sha256=b['source_sha256'],
        frozen_A_all_parameters=True,online_stage_conditioning=True,no_label_actor_inputs=True,to_delta_nonzero_tensors=nonzero,
        inference_arithmetic=b['inference_arithmetic'],original_vlm_calls=1,original_euler_steps=5)
    wire.update(stage_serving_identity=specific,lora_serving_identity=anchor['wire_anchor'],
        official_server_sha256=sha(Path(base['xr1_repo'])/'deploy/server.py'),loader_kind='completed_frozen_A_plus_online_stage_branch_v2')
    require(read_binding(path,expected_sha256)==b,'Endpoint changed during loading')
    return TrackedStageView(view,bridge),bridge,wire,dict(b,base_model=base['base_model'])
