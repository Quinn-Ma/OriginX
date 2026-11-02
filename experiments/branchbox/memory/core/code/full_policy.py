# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
# Copyright 2025 2toINF (https://github.com/2toINF)
# Adapted from LeRobot/X-VLA modeling_xvla.py and soft_transformer.py.
# MemBodied / OpenPI architectural source: https://github.com/declare-lab/MemBodied
# Pinned commit: e268134a9e5c4b99a4607b92f466f25d784b3f26
# Reference: RMBench/policy/pi0/src/openpi/models/pi0_mem_vector.py
# OpenPI foundation: https://github.com/Physical-Intelligence/openpi
# MODIFIED for the BranchBox X-VLA port; this is not upstream source.
# Changes: PyTorch explicit carry; 24 action-expert memory reads; two appended state/memory tokens; widened domain action affine for first-frame anchor; native Florence7x7 clean patches pooled4x4; full fine tuning with sequence BPTT.
# See NOTICE.txt, FILE_ORIGINS.json, PUBLIC_SOURCE_MAP.json and third_party/.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. Complete texts accompany this release.

"""Differentiable MemBodied main-architecture port onto an existing X-VLA.

Load the trained X-VLA-P checkpoint first; then construct this wrapper. This
module never loads another checkpoint, changes installed code, or detaches the
ordinary VLA visual/language/action path. See PORT_MAPPING.md for adaptations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from full_memory import FullAssociativeMemory, FullMemorySpec, MemoryCarry


@dataclass
class ObservationContext:
    vlm_features: Tensor
    aux_visual_inputs: Tensor
    proprio: Tensor
    domain_id: Tensor
    patches: Tensor
    state_token: Tensor
    vision_encoding: Tensor


def timestep_embedding(t: Tensor, dim: int, max_period: int = 100) -> Tensor:
    half = dim // 2
    frequency = torch.exp(-math.log(max_period) * torch.arange(half, dtype=t.dtype, device=t.device) / half)
    args = t[:, None] * frequency[None]
    result = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(result, (0, dim % 2))


def widen_action_encoder(encoder: nn.Module, anchor_width: int) -> nn.Module:
    """Literal concat widening of native domain-specific affine weight.

    Copies all old coefficients/biases exactly, adds zero anchor coefficients.
    This is NOT a parallel residual adapter and all coefficients are trainable.
    """
    old_input, output = encoder.input_size, encoder.output_size
    domains = encoder.fc.num_embeddings
    widened = type(encoder)(old_input + anchor_width, output, num_domains=domains)
    widened = widened.to(device=encoder.fc.weight.device, dtype=encoder.fc.weight.dtype)
    with torch.no_grad():
        target = widened.fc.weight.view(domains, old_input + anchor_width, output)
        target.zero_()
        target[:, :old_input].copy_(encoder.fc.weight.view(domains, old_input, output))
        widened.bias.weight.copy_(encoder.bias.weight)
    return widened


@contextmanager
def temporary_eval(module: nn.Module):
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in modes:
            child.training = training


class FullMemBodiedPolicy(nn.Module):
    def __init__(self, base_model: nn.Module, spec: FullMemorySpec = FullMemorySpec(),
                 *, gradient_checkpointing: bool = False):
        super().__init__()
        self.base = base_model
        self.spec = spec
        expert = base_model.transformer
        if len(expert.blocks) != spec.layers or expert.hidden_size != spec.width:
            raise ValueError("Specification must cover ALL actual X-VLA expert layers")
        if expert.dim_action != 20:
            raise ValueError("This port preserves native X-VLA 20-D noisy action input")
        self.original_action_encoder_input = expert.action_encoder.input_size
        expected = 20 + expert.dim_time + spec.proprio_dim
        if self.original_action_encoder_input != expected:
            raise ValueError("Unexpected native proprio/time/action affine shape")
        self.memory = FullAssociativeMemory(spec).to(
            device=next(base_model.parameters()).device, dtype=torch.float32)
        expert.action_encoder = widen_action_encoder(expert.action_encoder, spec.width)
        # Override checkpoint freeze configuration for every remaining weight.
        self.requires_grad_(True)
        self.gradient_checkpointing = gradient_checkpointing

    def encode_observation(self, input_ids: Tensor, pixel_values: Tensor, image_mask: Tensor,
                           proprio: Tensor, domain_id: Tensor,
                           memory_pixel_values: Tensor | None = None) -> ObservationContext:
        """One ordinary native VLA encoding, reusing detached clean-view patches.

        User chose original first-frame images and native Florence geometry.
        No separate high-resolution image encoding or fake feature upsampling.
        """
        if memory_pixel_values is not None:
            raise ValueError("Current memory patches reuse native images; do not supply a second image branch")
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            normal, patches = checkpoint(self._forward_vlm_and_patches, input_ids, pixel_values, image_mask,
                                         use_reentrant=False, preserve_rng_state=True)
        else:
            normal, patches = self._forward_vlm_and_patches(input_ids, pixel_values, image_mask)
        return self.context_from_features(normal["vlm_features"], normal["aux_visual_inputs"],
                                          proprio, domain_id, patches)

    def _forward_vlm_and_patches(self, input_ids, pixel_values, image_mask):
        """Native XVLAModel.forward_vlm arithmetic with extra patch return only."""
        batch, views = pixel_values.shape[:2]
        if views != 3 or not image_mask[:, [0,2]].bool().all():
            raise ValueError("Full memory needs native main/prompt/wrist layout and both physical cameras")
        flat_mask=image_mask.reshape(-1).bool()
        image_output=self.base.vlm.get_image_features(pixel_values.flatten(0,1)[flat_mask])
        valid_feats=image_output.pooler_output
        if tuple(image_output.last_hidden_state.shape[-2:]) != (self.spec.vision_grid,self.spec.vision_grid):
            raise ValueError("Unexpected actual native Florence spatial grid")
        tokens, hidden=valid_feats.shape[1:]
        if tokens != self.spec.vision_grid**2+1 or hidden != self.spec.vision_width:
            raise ValueError("Unexpected native projected patch layout")
        images=valid_feats.new_zeros((batch*views,tokens,hidden))
        images[flat_mask]=valid_feats
        images=images.view(batch,views,tokens,hidden)
        text=self.base.vlm.get_input_embeddings()(input_ids)
        merged=torch.cat([images[:,0],text],dim=1)
        mask=torch.ones(merged.shape[:2],dtype=torch.long,device=merged.device)
        encoded=self.base.vlm.language_model.encoder(attention_mask=mask,inputs_embeds=merged)[0]
        normal={"vlm_features":encoded,"aux_visual_inputs":images[:,1:].reshape(batch,-1,hidden)}
        # Projector ordering: firstglobaltoken, then row-major spatial tokens.
        return normal, images[:,[0,2],1:].detach()

    def encode_memory_patches(self, memory_pixel_values: Tensor) -> Tensor:
        """Detached native projected7x7 patches for TRUE episode first frame.

        A random training window's first query is not its episode's anchor.
        Call this separately on episode-initial clean cameras for init_carry.
        """
        if memory_pixel_values is None:
            raise ValueError("Anchor requires original episode-first-frame clean camera pixels")
        b, c, channels, height, width = memory_pixel_values.shape
        if (c, channels, height, width) != (self.spec.cameras, 3, 224, 224):
            raise ValueError("Anchor inputs must be native clean cameras [B,2,3,224,224]")
        # Same current vision weights at native resolution, once for firstframe.
        with torch.no_grad(), temporary_eval(self.base.vlm):
            images = self.base.vlm.get_image_features(memory_pixel_values.reshape(b * c, 3, height, width))
            raw = images.last_hidden_state
            if tuple(raw.shape[-2:]) != (self.spec.vision_grid, self.spec.vision_grid):
                raise ValueError(f"Unexpected actual Florence native grid: {tuple(raw.shape)}")
            # Florence projector emits [global spatial mean, row-major patches].
            projected = images.pooler_output
            expected = 1 + self.spec.vision_grid ** 2
            if projected.shape[1:] != (expected, self.spec.vision_width):
                raise ValueError("Unexpected Florence projected patch layout")
            patches = projected[:, 1:].reshape(b, c, expected - 1, self.spec.vision_width).detach()
        return patches

    def context_from_features(self, vlm_features: Tensor, aux_visual_inputs: Tensor,
                              proprio: Tensor, domain_id: Tensor, patches: Tensor) -> ObservationContext:
        """Public engineering/test interface; production full FT uses encode_observation."""
        state = self.memory.state_token(proprio)
        vision = self.memory.vision_encoding(patches, state)
        return ObservationContext(vlm_features, aux_visual_inputs, proprio, domain_id,
                                  patches, state, vision)

    def init_carry(self, context: ObservationContext, episode_anchor_patches: Tensor | None = None) -> MemoryCarry:
        """Pass true episode-initial patches for windows starting mid-episode."""
        patches = context.patches if episode_anchor_patches is None else episode_anchor_patches
        return self.memory.initial_carry(patches)

    def advance(self, carry: MemoryCarry, previous_h: Tensor, previous_executed_actions: Tensor,
                current_context: ObservationContext, action_mask: Tensor | None = None,
                batch_mask: Tensor | None = None) -> MemoryCarry:
        return self.memory.write(carry, previous_h, previous_executed_actions,
                                 current_context.vision_encoding, action_mask, batch_mask)

    def _block(self, index: int, x: Tensor, S: Tensor, state_index: int, memory_index: int):
        block = self.base.transformer.blocks[index]
        normalized = block.norm1(x)
        read = self.memory.read(index, normalized[:, state_index], S)
        # Out-of-place addition keeps autograd/BPTT and checkpointing safe.
        injection = F.pad(read[:, None], (0, 0, memory_index, x.shape[1] - memory_index - 1))
        x = x + block.attn(normalized + injection)
        x = x + block.mlp(block.norm2(x))
        return x, x[:, state_index]

    def predict(self, context: ObservationContext, noisy_actions: Tensor, t: Tensor,
                carry: MemoryCarry) -> tuple[Tensor, Tensor]:
        """One native clean-action denoising pass; does not update memory.

        Returns clean20 [B,30,20] and per-layer post-FFN STATE [B,24,1024].
        Ten inference iterations must share one carry; use only the final h.
        """
        expert, s = self.base.transformer, self.spec
        b, actions, dimension = noisy_actions.shape
        if dimension != 20 or carry.S.shape != (b, s.layers, s.rank, s.rank):
            raise ValueError("Wrong native noisy action or recurrent state shape")
        time = timestep_embedding(t, expert.dim_time)[:, None].expand(b, actions, -1)
        proprio = context.proprio[:, None].expand(b, actions, -1)
        anchor = self.memory.read_anchor(context.vision_encoding, carry.anchor)
        anchor = anchor[:, None].expand(b, actions, -1).to(noisy_actions.dtype)
        tokens = torch.cat([noisy_actions, proprio, time, anchor], dim=-1)
        x = expert.action_encoder(tokens, context.domain_id)
        if expert.use_hetero_proj:
            v = expert.vlm_proj(context.vlm_features, context.domain_id)
            aux = expert.aux_visual_proj(context.aux_visual_inputs, context.domain_id)
        else:
            v, aux = expert.vlm_proj(context.vlm_features), expert.aux_visual_proj(context.aux_visual_inputs)
        x = torch.cat([x, v, aux], dim=1)
        if x.shape[1] > expert.pos_emb.shape[1]:
            raise ValueError("Native position embedding length exceeded")
        x = x + expert.pos_emb[:, :x.shape[1]]
        if expert.len_soft_prompts:
            soft = expert.soft_prompt_hub(context.domain_id).view(b, expert.len_soft_prompts, s.width)
            x = torch.cat([x, soft], dim=1)
        state_index, memory_index = x.shape[1], x.shape[1] + 1
        # Genuine extra tokens; existing prompts/positions are NOT repurposed.
        x = torch.cat([x, context.state_token[:, None].to(x.dtype),
                       self.memory.memory_token.expand(b, -1, -1).to(x.dtype)], dim=1)
        states = []
        for i in range(s.layers):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                def run(value, state, index=i):
                    return self._block(index, value, state, state_index, memory_index)
                x, h = checkpoint(run, x, carry.S, use_reentrant=False, preserve_rng_state=True)
            else:
                x, h = self._block(i, x, carry.S, state_index, memory_index)
            states.append(h)
        clean = expert.action_decoder(expert.norm(x[:, :actions]), context.domain_id)
        return clean, torch.stack(states, dim=1)

    def forward(self, context: ObservationContext, noisy_actions: Tensor, t: Tensor,
                carry: MemoryCarry):
        return self.predict(context, noisy_actions, t, carry)

    @torch.no_grad()
    def sample(self, context: ObservationContext, carry: MemoryCarry, initial_noise: Tensor,
               denoising_steps: int = 10) -> tuple[Tensor, Tensor]:
        """Native X-VLA clean-action fixed-point sampler, final h only."""
        self.eval()  # Same mode transition as native XVLAModel.generate_actions.
        if initial_noise.shape[-1] != 20 or denoising_steps < 1:
            raise ValueError("Invalid native inference noise/steps")
        action = torch.zeros_like(initial_noise)
        h = None
        for i in range(denoising_steps, 0, -1):
            t = torch.full((initial_noise.shape[0],), i / denoising_steps,
                           device=initial_noise.device, dtype=initial_noise.dtype)
            x_t = initial_noise * t[:, None, None] + action * (1 - t[:, None, None])
            action, h = self.predict(context, x_t, t, carry)
        return action[..., :self.spec.action_dim], h

    def parameter_report(self):
        base = [(n, p) for n, p in self.base.named_parameters()]
        return {"base_parameters": sum(p.numel() for _, p in base),
                "base_trainable_parameters": sum(p.numel() for _, p in base if p.requires_grad),
                "memory_parameters": sum(p.numel() for p in self.memory.parameters()),
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "frozen_parameter_names": [n for n, p in self.named_parameters() if not p.requires_grad],
                "original_action_encoder_input": self.original_action_encoder_input,
                "widened_action_encoder_input": self.base.transformer.action_encoder.input_size,
                "deleted_native_text_decoder": not hasattr(self.base.vlm.language_model, "decoder")}
