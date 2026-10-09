"""Frozen completed-A actor/teacher plus an online stage-conditioned branch.

A temporary t_projector hook adds the predicted delta before the released view
into six adaLN channels. This acts on every real DiT layer. Hooks have bounded
scope, fail on reentrancy, and are removed even on exceptions. No A source edit.
"""
from contextlib import contextmanager,nullcontext
import math
import torch
from torch import nn
from training_bridge.bridge import FrozenVLMFlowBridge
from lora_parallel_v1.batched_bridge import equal_sample_masked_mse,masked_position_max
from teacher_anchor_pilot_v1.bridge import equal_rng,tensor_versions
from training_bridge.train_core import rng_state
from .branch import StageBranch,auxiliary_loss,require


@contextmanager
def modulation(model,delta):
    require(not getattr(model,'_stage_v2_modulation_active',False),'Stage modulation is non-reentrant')
    require(delta.ndim==3 and delta.shape[1]==6, 'Expected stage delta[B,6,D]')
    count=[0]
    def add(_module,_args,output):
        require(isinstance(output,torch.Tensor) and output.shape==(len(delta),1,delta.shape[1]*delta.shape[2]),
                'Released t_projector geometry changed')
        count[0]+=1
        return output+delta.reshape_as(output).to(output.dtype)
    model._stage_v2_modulation_active=True
    hook=model.t_projector.register_forward_hook(add)
    try:yield count
    finally:
        hook.remove();del model._stage_v2_modulation_active


class FrozenAGuard:
    def __init__(self,model):
        self.model=model
        model.requires_grad_(False);model.eval()
        self.versions=self.snapshot()
    def snapshot(self):
        return [(n,id(v),v._version,tuple(v.shape),v.dtype,v.device)
                for n,v in list(self.model.named_parameters())+list(self.model.named_buffers())]
    def check(self):
        require(self.snapshot()==self.versions,'Frozen A parameter/buffer changed')
        require(all(not p.requires_grad and p.grad is None for p in self.model.parameters()),'A must stay fully frozen including LoRA')
        require(all(not m.training for m in self.model.modules()),'Frozen A must stay in eval mode')
        require(all(getattr(m,'adapter_enabled',True) for m in self.model.modules()),'Frozen A LoRA must remain enabled')


class StageFlowBridge(FrozenVLMFlowBridge):
    def __init__(self, model, branch, *, supervised, retention_coefficient=1.0,
                 auxiliary_coefficient=1.0,confidence_threshold=1.0):
        nn.Module.__init__(self)  # Never call the old constructor that unfreezes A.
        require(isinstance(branch,StageBranch) and type(supervised) is bool,'Explicit shared branch and supervision mode required')
        require((model.config.state_dim,model.config.action_dim,model.config.state_length)==(60,60,4),'Released RoboCasa geometry required')
        for v in (retention_coefficient,auxiliary_coefficient):
            require(math.isfinite(v) and v>=0,'Finite nonnegative loss coefficient required')
        require(math.isfinite(confidence_threshold) and 0<=confidence_threshold<=1,'Invalid confidence threshold')
        self.model,self.branch=model,branch
        self.supervised=supervised;self.retention_coefficient=float(retention_coefficient)
        self.auxiliary_coefficient=float(auxiliary_coefficient);self.confidence_threshold=float(confidence_threshold)
        self.guard=FrozenAGuard(model);self._in_forward=False;self.teacher_calls=0
        self._assert_trainable_contract()

    def _assert_trainable_contract(self):
        self.guard.check()
        expected={'branch.'+n for n,_ in self.branch.named_parameters()}
        require({n for n,p in self.named_parameters() if p.requires_grad}==expected,'Only all new branch parameters may train')
        require(all(p.dtype==torch.float32 for p in self.branch.parameters()),'Branch requires FP32 masters')

    def train(self,mode=True):
        nn.Module.train(self,mode);self.model.eval();self._assert_trainable_contract();return self

    def trainable_parameters(self):
        self._assert_trainable_contract();return self.branch.parameters()

    def named_branch_parameters(self):return self.branch.named_parameters()

    def observed_delta(self,context,inputs):
        config=self.model.config.vlm_config
        return self.branch(context.last_hidden_state,inputs['input_ids'],inputs['attention_mask'],
            inputs['video_grid_thw'],inputs['state'],video_token_id=config.video_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size)

    def forward(self,inputs,actions,valid_steps,*,noise,times,stage_target=None,stage_confidence=None):
        require(not self._in_forward,'StageFlowBridge is non-reentrant')
        self._assert_trainable_contract();self._validate_batch(inputs,actions,valid_steps)
        self._in_forward=True
        try:
            autocast=torch.autocast('cuda',dtype=torch.bfloat16) if actions.is_cuda else nullcontext()
            with autocast:return self._stage_loss(inputs,actions,valid_steps,noise,times,stage_target,stage_confidence)
        finally:self._in_forward=False

    def _stage_loss(self,inputs,actions,valid,noise,times,stage_target,stage_confidence):
        model=self.model;B=len(actions);mask,state=inputs['action_mask'],inputs['state']
        require(noise is not None and times is not None,'Explicit independent per-slot noise/time required')
        require(noise.shape==mask.shape and noise.device==mask.device and torch.isfinite(noise).all(),'Bad flow noise')
        require(times.shape==(B,1,1) and times.device==mask.device and torch.isfinite(times).all()
                and ((times>=0)&(times<=1)).all(),'Bad flow time')
        vlm_inputs={k:v for k,v in inputs.items() if k not in {'state','action_mask','task_id'}}
        with torch.no_grad():context=model.vlm.model(**vlm_inputs,use_cache=True)
        delta,logits=self.observed_delta(context,inputs)
        length=1+state.shape[1]+mask.shape[1]
        positions=(torch.arange(length,device=mask.device).view(1,1,-1).repeat(3,B,1)
                   +masked_position_max(context.position_ids,inputs['attention_mask'])[...,None]+1)
        rotary=model.rotary_emb(mask,positions)
        cache=inputs['attention_mask'][:,None,:].expand(-1,length,-1)
        causal=torch.tril(torch.ones(B,length,length,device=mask.device))
        attention=torch.cat([cache,causal],dim=-1)[:,None].bool()
        target=actions.to(mask.dtype);noise=noise.to(mask.dtype);times=times.to(mask.dtype)
        with torch.no_grad():state_embed=model.state_projector(state)
        arguments=dict(noisy_action=(1-times)*noise+times*target,t=times,action_mask=mask,
            state_embed=state_embed,position_embeds=rotary,past_key_values=context.past_key_values,attn_mask=attention)
        with modulation(model,delta) as calls:student=model.dit_forward(**arguments)
        require(calls==[1] and student.shape==actions.shape,'Student must use stage modulation exactly once')
        loss_mask=mask.bool()&valid[...,None]
        flow=equal_sample_masked_mse(student.float()-(target.float()-noise.float()),loss_mask)
        if self.retention_coefficient:
            before_rng=rng_state(actions.device);versions=tensor_versions(arguments)
            # A's nonzero LoRA remains enabled. Only the new branch is absent.
            with torch.no_grad():teacher=model.dit_forward(**arguments)
            require(equal_rng(before_rng,rng_state(actions.device)) and tensor_versions(arguments)==versions,
                    'Frozen A teacher consumed RNG or mutated shared inputs/cache')
            require(not teacher.requires_grad and teacher.shape==student.shape,'Wrong teacher graph/geometry')
            retention=equal_sample_masked_mse(student.float()-teacher.float(),loss_mask);self.teacher_calls+=1
        else:retention=torch.zeros_like(flow)
        auxiliary,admitted=auxiliary_loss(logits,stage_target,stage_confidence,
            threshold=self.confidence_threshold,enabled=self.supervised)
        rows=flow+self.retention_coefficient*retention
        result=dict(loss=rows.mean()+self.auxiliary_coefficient*auxiliary,flow_loss=flow.mean(),retention_loss=retention.mean(),
            phase_aux_loss=auxiliary,phase_aux_rows=admitted.sum().detach(),phase_admitted=admitted,
            per_sample_loss=rows,per_sample_flow_loss=flow,per_sample_retention_loss=retention,
            active_values=loss_mask.sum().detach(),active_values_per_sample=loss_mask.flatten(1).sum(1).detach(),
            time_mean=times.float().mean().detach(),stage_logits=logits,stage_delta=delta)
        self._assert_trainable_contract();return result


class StageInferenceView:
    """Use the original full actor/Euler/RNG with one observed-context hook.

    Caller serializes requests; this view is intentionally non-reentrant. Neither
    true stage IDs nor confidence are accepted. A/processor control stays original.
    """
    def __init__(self,bridge):self.bridge=bridge;self.model=bridge.model;self._busy=False
    @property
    def device(self):return next(self.model.parameters()).device
    @property
    def dtype(self):return next(self.model.parameters()).dtype
    @property
    def config(self):return self.model.config
    def __call__(self,**inputs):
        require(not self._busy,'Stage inference is non-reentrant')
        require(not any(k in inputs for k in ('stage_target','stage_confidence','stage','phase_delta','phase_label')),
                'Labels/oracle stages are forbidden actor inputs')
        self.bridge._assert_trainable_contract();self._busy=True;scope=[None];contexts=[0];calls=[None]
        def capture(_module,_args,context):
            contexts[0]+=1;require(contexts[0]==1,'Only the original single VLM pass is allowed')
            delta,_=self.bridge.observed_delta(context,inputs)
            scope[0]=modulation(self.model,delta);calls[0]=scope[0].__enter__()
        hook=self.model.vlm.model.register_forward_hook(capture)
        try:
            autocast=torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else nullcontext()
            with torch.no_grad(),autocast:result=self.model(**inputs)
            require(contexts==[1] and calls[0]==[inputs.get('num_steps',5)],'Original VLM/Euler call count changed')
            return result
        finally:
            hook.remove()
            if scope[0] is not None:scope[0].__exit__(None,None,None)
            self._busy=False;self.bridge._assert_trainable_contract()
