# SPDX-License-Identifier: Apache-2.0
# MemBodied / OpenPI architectural source: https://github.com/declare-lab/MemBodied
# Pinned commit: e268134a9e5c4b99a4607b92f466f25d784b3f26
# Reference: RMBench/policy/pi0/src/openpi/models/pi0_mem_vector.py
# OpenPI foundation: https://github.com/Physical-Intelligence/openpi
# MODIFIED for the BranchBox X-VLA port; this is not upstream source.
# Changes: JAX/Flax to PyTorch; X-VLA dimensions24x1024/rank128; explicit immutable carry; two physical cameras; native7x7 to4x4 pooling; one executed30-action chunk per delayed write.
# See NOTICE.txt, FILE_ORIGINS.json, PUBLIC_SOURCE_MAP.json and third_party/.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. Complete texts accompany this release.

"""MemBodied main-architecture associative memory, PyTorch port.

Reference: declare-lab/MemBodied e268134a9e5c4b99a4607b92f466f25d784b3f26,
RMBench/policy/pi0/src/openpi/models/pi0_mem_vector.py. See PORT_MAPPING.md
for the explicit X-VLA adaptations. No hidden recurrent-state mutation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FullMemorySpec:
    width: int = 1024
    vision_width: int = 1024
    proprio_dim: int = 20
    action_dim: int = 7
    layers: int = 24
    rank: int = 128
    memory_alpha: float = 256.0
    alpha_bias: float = 2.0
    beta_bias: float = -1.5
    anchor_rank: int = 64
    anchor_grid: int = 4
    vision_grid: int = 7
    cameras: int = 2
    memory_dropout: float = 0.0

    def as_dict(self):
        return asdict(self)


def linear(inp: int, out: int, *, zero: bool = False) -> nn.Linear:
    """NNX Linear default LeCun truncated-normal variance, zero bias."""
    result = nn.Linear(inp, out)
    if zero:
        nn.init.zeros_(result.weight)
    else:
        std = 1.0 / math.sqrt(inp) / 0.8796256610342398
        nn.init.trunc_normal_(result.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.zeros_(result.bias)
    return result


@dataclass
class MemoryCarry:
    S: Tensor  # [B,L,R,R], one slot for the actually executed 30-action chunk
    anchor: Tensor  # [B,C*16,V], episode-first-frame fixed patch grid

    def detach(self) -> "MemoryCarry":
        return MemoryCarry(self.S.detach(), self.anchor.detach())


class MemoryLayer(nn.Module):
    def __init__(self, spec: FullMemorySpec):
        super().__init__()
        d, r, v = spec.width, spec.rank, spec.vision_width + spec.action_dim
        self.query = linear(d, r)
        self.read_out = linear(r, d, zero=True)
        self.read_gate = linear(d, 1, zero=True)
        self.key = linear(d, r)
        self.value = linear(v, r)
        self.alpha = linear(d + v, r)
        self.beta = linear(d + v, r)


class FullAssociativeMemory(nn.Module):
    def __init__(self, spec: FullMemorySpec = FullMemorySpec()):
        super().__init__()
        self.spec = spec
        d, v, r = spec.width, spec.vision_width, spec.rank
        if spec.vision_grid < spec.anchor_grid:
            raise ValueError("Anchor grid cannot invent more spatial patches")
        self.state_proj = linear(spec.proprio_dim, d)
        self.memory_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.S_init = nn.Parameter(torch.randn(spec.layers, r, r) * 0.02)
        self.layers = nn.ModuleList(MemoryLayer(spec) for _ in range(spec.layers))
        self.pool_q = linear(d, v)
        self.pool_k = linear(v, v)
        self.pool_v = linear(v, v)
        self.anchor_q = linear(v, spec.anchor_rank)
        self.anchor_k = linear(v, spec.anchor_rank)
        self.anchor_v = linear(v, spec.anchor_rank)
        self.anchor_out = linear(spec.anchor_rank, v)
        self.anchor_proj = linear(v, d)

    def _check_patches(self, patches: Tensor):
        s = self.spec
        expected = (s.cameras, s.vision_grid * s.vision_grid, s.vision_width)
        if tuple(patches.shape[1:]) != expected:
            raise ValueError(f"Expected genuine camera grid [B,{expected}], got {tuple(patches.shape)}")

    def state_token(self, proprio: Tensor) -> Tensor:
        with torch.autocast(device_type=proprio.device.type, enabled=False):
            return self.state_proj(proprio.float())

    def vision_encoding(self, patches: Tensor, state_token: Tensor) -> Tensor:
        """State-query attention per camera, then average camera summaries.

        Raw patch gradient is stopped exactly as in the official memory path;
        state/query/key/value projections remain trainable.
        """
        self._check_patches(patches)
        with torch.autocast(device_type=patches.device.type, enabled=False):
            patches = patches.detach().float()
            q = self.pool_q(state_token.float())[:, None, None, :]
            k, v = self.pool_k(patches), self.pool_v(patches)
            weights = (q * k).sum(-1) / math.sqrt(self.spec.vision_width)
            attended = (weights.softmax(-1)[..., None] * v).sum(-2)
            return attended.mean(1)

    def extract_anchor(self, patches: Tensor) -> Tensor:
        self._check_patches(patches)
        s = self.spec
        b, c, _, v = patches.shape
        g = s.anchor_grid
        # User-requested native-image adaptation: native Florence7x7 ->4x4.
        # For a divisible grid (e.g. official16), adaptive pooling reduces to
        # exactly the official nonoverlapping spatial block mean.
        x = patches.detach().float().permute(0, 1, 3, 2).reshape(b*c, v, s.vision_grid, s.vision_grid)
        return F.adaptive_avg_pool2d(x, (g,g)).reshape(b,c,v,g*g).permute(0,1,3,2).reshape(b,c*g*g,v)

    def initial_carry(self, patches: Tensor) -> MemoryCarry:
        return MemoryCarry(self.S_init.unsqueeze(0).expand(patches.shape[0], -1, -1, -1),
                           self.extract_anchor(patches))

    def read_anchor(self, vision: Tensor, anchor: Tensor) -> Tensor:
        with torch.autocast(device_type=vision.device.type, enabled=False):
            q = self.anchor_q(vision.float()).unsqueeze(1)
            k, v = self.anchor_k(anchor.float()), self.anchor_v(anchor.float())
            weights = (q @ k.transpose(-1, -2) / math.sqrt(self.spec.anchor_rank)).softmax(-1)
            fused = self.anchor_out((weights @ v).squeeze(1)) + vision.float()
            return self.anchor_proj(fused)

    def read(self, layer: int, normalized_state_token: Tensor, S: Tensor) -> Tensor:
        p = self.layers[layer]
        original_dtype = normalized_state_token.dtype
        with torch.autocast(device_type=normalized_state_token.device.type, enabled=False):
            h = normalized_state_token.float()
            q = F.normalize(torch.tanh(p.query(h)), dim=-1, eps=1e-12)
            read = torch.einsum("bij,bj->bi", S[:, layer].float(), q)
            memory = p.read_out(read) * (self.spec.memory_alpha / self.spec.rank)
            memory = F.dropout(memory, p=self.spec.memory_dropout, training=self.training)
            return (torch.sigmoid(p.read_gate(h)) * memory).to(original_dtype)

    def write(self, carry: MemoryCarry, previous_h: Tensor, executed_actions: Tensor,
              current_vision: Tensor, action_mask: Tensor | None = None,
              batch_mask: Tensor | None = None) -> MemoryCarry:
        """Delayed write AFTER action execution and receipt of resulting image.

        All action entries are actual normalized controls (demo controls during
        training), never the unexecuted suffix of a predicted action chunk.
        No detach: temporal gradients are retained throughout caller's sequence.
        """
        b, length, ad = executed_actions.shape
        s = self.spec
        if ad != s.action_dim or previous_h.shape != (b, s.layers, s.width):
            raise ValueError("Incorrect executed-action or previous-state-token shape")
        if action_mask is None:
            action_mask = torch.ones((b, length), device=executed_actions.device, dtype=torch.bool)
        if batch_mask is None:
            batch_mask = torch.ones(b, device=executed_actions.device, dtype=torch.bool)
        if action_mask.shape != (b, length) or batch_mask.shape != (b,):
            raise ValueError("Invalid action/batch write mask")
        if torch.any(batch_mask & ~action_mask.any(-1)):
            raise ValueError("A real write requires at least one executed action")
        with torch.autocast(device_type=executed_actions.device.type, enabled=False):
            # where, not multiplication, also rejects padded NaNs from values.
            actions = torch.where(action_mask[..., None], executed_actions.float(), 0).sum(1)
            value = torch.cat([current_vision.float(), actions], dim=-1)
            outputs = []
            for i, p in enumerate(self.layers):
                h = previous_h[:, i].float()
                key = F.normalize(torch.tanh(p.key(h)), dim=-1, eps=1e-12)
                val = p.value(value)
                gates = torch.cat([h, value], dim=-1)
                alpha = torch.sigmoid(p.alpha(gates) + s.alpha_bias)
                beta = torch.sigmoid(p.beta(gates) + s.beta_bias)
                old = carry.S[:, i].float()
                prediction = torch.einsum("bij,bj->bi", old, key) * alpha
                delta = (val - prediction)[..., None] * key[:, None, :]
                updated = alpha[..., None] * old + beta[..., None] * delta
                outputs.append(torch.where(batch_mask[:, None, None], updated, old))
            return MemoryCarry(torch.stack(outputs, dim=1), carry.anchor)
