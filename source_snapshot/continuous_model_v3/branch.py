"""Online latent stages from observed VLM tokens and proprioception only.

True labels enter auxiliary_loss only; forward has no label/progress/oracle input.
The same parameterization is used with and without stage auxiliary supervision.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .observed import ObservedFeaturePool, require


class StageBranch(nn.Module):
    def __init__(self, *, hidden_dim, dit_dim, stage_count, hidden_mlp_dim=256, init_seed=2126104002):
        super().__init__()
        for value in (hidden_dim,dit_dim,stage_count,hidden_mlp_dim):
            require(type(value) is int and value>0, 'Positive integer branch geometry required')
        require(stage_count>=2, 'At least two train-only stage classes required')
        require(type(init_seed) is int and 0<=init_seed<2**63, 'Invalid branch initialization seed')
        self.hidden_dim,self.dit_dim,self.stage_count=hidden_dim,dit_dim,stage_count
        self.settings=dict(hidden_dim=hidden_dim,dit_dim=dit_dim,stage_count=stage_count,
                           hidden_mlp_dim=hidden_mlp_dim,init_seed=init_seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            self.encoder=nn.Sequential(nn.Linear(4*hidden_dim+56,hidden_mlp_dim,device='cpu',dtype=torch.float32),
                                       nn.SiLU())
            self.stage_head=nn.Linear(hidden_mlp_dim,stage_count,device='cpu',dtype=torch.float32)
            self.to_delta=nn.Linear(hidden_mlp_dim,6*dit_dim,device='cpu',dtype=torch.float32)
            nn.init.zeros_(self.to_delta.weight);nn.init.zeros_(self.to_delta.bias)

    # Reuse the already tested causal temporal-group/camera/padding contract.
    # This method only needs self.hidden_dim; no old semantic-decoder weights.
    pool_features=ObservedFeaturePool.pool_features

    def forward(self, hidden, input_ids, attention_mask, video_grid_thw, state,
                *, video_token_id, spatial_merge_size):
        features=self.pool_features(hidden.detach(),input_ids,attention_mask,video_grid_thw,
                                    state.detach(),video_token_id,spatial_merge_size)
        require(features.device==self.encoder[0].weight.device, 'Branch/features device mismatch')
        if not torch.is_autocast_enabled(features.device.type):features=features.to(torch.float32)
        latent=self.encoder(features)
        logits=self.stage_head(latent)
        delta=self.to_delta(latent).reshape(len(logits),6,self.dit_dim)
        require(torch.isfinite(logits).all() and torch.isfinite(delta).all(), 'Nonfinite stage prediction')
        return delta,logits


def auxiliary_loss(logits, target, confidence, *, threshold, enabled):
    require(type(enabled) is bool, 'Explicit supervision mode required')
    require(type(threshold) in (int,float) and math.isfinite(threshold) and 0<=threshold<=1,
            'Threshold must be finite in[0,1]')
    if not enabled:
        # Both arms compute the identical logits/action branch; only this loss differs.
        return logits.sum()*0.0,torch.zeros(len(logits),device=logits.device,dtype=torch.bool)
    require(isinstance(target,torch.Tensor) and target.dtype==torch.long and target.shape==(len(logits),)
            and target.device==logits.device, 'Expected on-device stage_target long[B]')
    require(isinstance(confidence,torch.Tensor) and confidence.is_floating_point() and confidence.shape==target.shape
            and confidence.device==logits.device and not confidence.requires_grad,
            'Expected detached stage confidence[B]')
    require(torch.isfinite(confidence).all() and ((confidence>=0)&(confidence<=1)).all(), 'Confidence outside[0,1]')
    require(((target>=-1)&(target<logits.shape[1])).all(), 'Stage index is outside frozen train vocabulary')
    admitted=(target>=0)&(confidence>=threshold)
    if not admitted.any():return logits.sum()*0.0,admitted
    # Mean over ALL windows: masked rows have zero CE; replica B/global scaling
    # therefore gives the identical global objective regardless of label density.
    return F.cross_entropy(logits[admitted].float(),target[admitted],reduction='sum')/len(logits),admitted
