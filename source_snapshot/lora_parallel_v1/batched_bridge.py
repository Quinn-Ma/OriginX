"""Equal-window batched flow loss over the frozen-base LoRA bridge.

Numerical recipe: each sample is normalized by its own valid action count,
then sample losses are averaged. Slot noise/time must be generated separately
and concatenated before calling this module. No model loading or optimizer here.
"""
import math

import torch

from route_reassessment_20261004.action_lora.lora_flow_bridge import LoRAFlowBridge
from route_reassessment_20261004.action_lora.action_lora import adapters_disabled
from training_bridge.train_core import rng_state
from teacher_anchor_pilot_v1.bridge import equal_rng, tensor_versions


def equal_sample_masked_mse(error, mask):
    if error.shape != mask.shape or error.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError('Expected matching [B,T,D] error and boolean mask')
    if not mask.flatten(1).any(dim=1).all():
        raise ValueError('Each sample needs active supervised values')
    # Match the original single-sample masked reduction, avoiding a global
    # masked mean that would overweight windows with more valid tail steps.
    return torch.stack([row[valid].float().square().mean() for row, valid in zip(error, mask)])


def masked_position_max(position_ids, attention_mask):
    if (position_ids.ndim != 3 or position_ids.shape[0] != 3 or
            tuple(position_ids.shape[1:]) != tuple(attention_mask.shape)):
        raise ValueError('Expected original3-axis [3,B,S] positions and[B,S] token mask')
    mask = attention_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError('Each sample needs at least one unpadded context token')
    minimum = torch.finfo(position_ids.dtype).min if position_ids.is_floating_point() else torch.iinfo(position_ids.dtype).min
    return position_ids.masked_fill(~mask[None], minimum).max(dim=-1).values


class BatchedLoRAFlowBridge(LoRAFlowBridge):
    def _loss(self, inputs, actions, valid_steps, noise, times):
        model = self.model
        batch_size = actions.shape[0]
        if batch_size > 1 and (noise is None or times is None):
            raise ValueError('Batched execution requires explicit independently drawn slot noise/time')
        vlm_inputs = {key: value for key, value in inputs.items() if key not in {'state','action_mask','task_id'}}
        with torch.no_grad():
            context = model.vlm.model(**vlm_inputs, use_cache=True)
        mask, state = inputs['action_mask'], inputs['state']
        query_length = 1 + state.shape[1] + mask.shape[1]
        max_positions = masked_position_max(context.position_ids, inputs['attention_mask'])
        positions = (torch.arange(query_length, device=mask.device).view(1,1,-1).repeat(3,batch_size,1)
                     + max_positions[...,None] + 1)
        rotary = model.rotary_emb(mask, positions)
        cache_mask = inputs['attention_mask'][:,None,:].expand(-1,query_length,-1)
        causal = torch.tril(torch.ones(batch_size,query_length,query_length,device=mask.device))
        attention = torch.cat([cache_mask,causal],dim=-1)[:,None].bool()
        target = actions.to(dtype=mask.dtype)
        if noise is None:
            noise = torch.randn_like(mask)
        else:
            if noise.shape != mask.shape or noise.device != mask.device or not torch.isfinite(noise).all():
                raise ValueError('Noise shape/device/value differs from the action mask')
            noise = noise.to(dtype=mask.dtype)
        if times is None:
            times = torch.rand((batch_size,1,1),device=mask.device,dtype=mask.dtype)
        else:
            if (times.shape != (batch_size,1,1) or times.device != mask.device or
                    not torch.isfinite(times).all() or (times<0).any() or (times>1).any()):
                raise ValueError('Times must be on-device finite[B,1,1] in[0,1]')
            times = times.to(dtype=mask.dtype)
        arguments = dict(noisy_action=(1-times)*noise+times*target,t=times,
                         action_mask=mask,state_embed=model.state_projector(state),
                         position_embeds=rotary,past_key_values=context.past_key_values,attn_mask=attention)
        student = model.dit_forward(**arguments)
        if student.shape != actions.shape:
            raise ValueError('Student returned wrong velocity geometry')
        loss_mask = mask.bool() & valid_steps[...,None]
        flow_rows = equal_sample_masked_mse(student.float()-(target.float()-noise.float()),loss_mask)
        if self.retention_coefficient:
            before_rng = rng_state(actions.device)
            before_versions = tensor_versions(arguments)
            with adapters_disabled(self.handle),torch.no_grad():
                teacher = model.dit_forward(**arguments)
            if not equal_rng(before_rng,rng_state(actions.device)):
                raise RuntimeError('Teacher consumed RNG')
            if tensor_versions(arguments) != before_versions:
                raise RuntimeError('Teacher mutated shared context/input')
            if teacher.requires_grad or teacher.grad_fn is not None or teacher.shape != student.shape:
                raise RuntimeError('Teacher graph/geometry invalid')
            self._assert_trainable_contract()
            self.teacher_calls += 1
            retention_rows = equal_sample_masked_mse(student.float()-teacher.float(),loss_mask)
        else:
            retention_rows = torch.zeros_like(flow_rows)
        rows = flow_rows + self.retention_coefficient*retention_rows
        return {'loss':rows.mean(),'flow_loss':flow_rows.mean(),'retention_loss':retention_rows.mean(),
                'per_sample_loss':rows,'per_sample_flow_loss':flow_rows,
                'per_sample_retention_loss':retention_rows,
                'active_values':loss_mask.sum().detach(),
                'active_values_per_sample':loss_mask.flatten(1).sum(1).detach(),
                'time_mean':times.float().mean().detach()}


def compute_microbatch_gradients(bridge,inputs,actions,valid,*,noise,times,loss_scale):
    """Backward only; caller owns zeroing, full-update clipping and optimization.

    For two independent four-window replicas of global batch8 use loss_scale=.5
    on each, then sum gradients before clipping/stepping. No hidden all-reduce.
    """
    if not math.isfinite(loss_scale) or not 0 < loss_scale <= 1:
        raise ValueError('loss_scale must be in(0,1]')
    result = bridge(inputs,actions,valid,noise=noise,times=times)
    if not torch.isfinite(result['loss']).item():
        raise FloatingPointError('Nonfinite batched flow/retention loss')
    (result['loss']*loss_scale).backward()
    return {'loss':result['loss'].detach().float().item(),
            'flow_loss':result['flow_loss'].detach().float().item(),
            'retention_loss':result['retention_loss'].detach().float().item(),
            'samples':actions.shape[0],
            'per_sample_loss':result['per_sample_loss'].detach().float().cpu().tolist(),
            'per_sample_flow_loss':result['per_sample_flow_loss'].detach().float().cpu().tolist(),
            'per_sample_retention_loss':result['per_sample_retention_loss'].detach().float().cpu().tolist(),
            'active_values':int(result['active_values'].item())}
