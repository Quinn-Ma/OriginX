"""Exact checkpoint action normalization for actually executed environment controls.

The installed LeRobot normalizer uses (raw-mean)/(std+eps), whereas its
unnormalizer uses normalized*std+mean. Inverting two postprocessor probes is
therefore not exactly the training transformation. This helper copies the actual
loaded preprocessor step and invokes its public action-only transition API.
"""
from __future__ import annotations
from copy import deepcopy
import math
import torch


class ExactActionNormalizer:
    def __init__(self, preprocessor, postprocessor, device):
        from lerobot.configs import FeatureType, NormalizationMode
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        from lerobot.lerobot_types import TransitionKey
        normalizers=[s for s in preprocessor.steps if isinstance(s,NormalizerProcessorStep)]
        unnormalizers=[s for s in postprocessor.steps if isinstance(s,UnnormalizerProcessorStep)]
        if len(normalizers)!=1 or len(unnormalizers)!=1:
            raise ValueError('requires one saved action normalizer and one saved action unnormalizer')
        original=normalizers[0];inverse=unnormalizers[0]
        for step in (original,inverse):
            if step.norm_map.get(FeatureType.ACTION)!=NormalizationMode.MEAN_STD:
                raise ValueError('existing checkpoint must use action MEAN_STD')
            if tuple(step.features['action'].shape)!=(7,):
                raise ValueError('existing checkpoint must have seven action dimensions')
        saved=original.state_dict();saved_inverse=inverse.state_dict()
        for name in ('action.mean','action.std'):
            if name not in saved or name not in saved_inverse:
                raise ValueError('missing saved '+name)
            if not torch.equal(saved[name],saved_inverse[name]):
                raise ValueError('pre/post action statistics differ: '+name)
            if saved[name].numel()!=7 or not torch.isfinite(saved[name]).all():
                raise ValueError('invalid saved '+name)
        if bool((saved['action.std']<=0).any()):
            raise ValueError('action std must be positive')
        if not math.isfinite(original.eps) or original.eps<0:
            raise ValueError('normalizer eps must be finite and nonnegative')
        self.device=torch.device(device)
        self.step=deepcopy(original).to(device=self.device,dtype=torch.float32)
        self.action_key=TransitionKey.ACTION
        self.mean=saved['action.mean'].to(device=self.device,dtype=torch.float32).clone()
        self.std=saved['action.std'].to(device=self.device,dtype=torch.float32).clone()
        self.device=self.mean.device  # Resolve bare 'cuda' to its concrete ordinal.
        self.eps=float(original.eps)

    @torch.no_grad()
    def __call__(self, raw_action):
        raw=torch.as_tensor(raw_action,device=self.device,dtype=torch.float32)
        if raw.ndim<1 or raw.shape[-1]!=7 or not torch.isfinite(raw).all():
            raise ValueError('executed action must be finite with last dimension seven')
        # Public ProcessorStep call: it sees only an action, never observations,
        # language, unexecuted future controls, or environment outcome labels.
        result=self.step({self.action_key:raw})[self.action_key]
        if result.device!=self.device or result.dtype!=torch.float32:
            raise RuntimeError('checkpoint action normalizer moved device or changed precision')
        return result

    def audit_dict(self):
        return {'recipe':'loaded NormalizerProcessorStep action-only public call',
                'formula':'(raw - mean) / (std + eps)',
                'eps':self.eps,'mean':self.mean.cpu().tolist(),'std':self.std.cpu().tolist(),
                'device':str(self.device),'dtype':'torch.float32','pre_post_stats_identical':True}


def make_action_normalizer(preprocessor,postprocessor,device):
    return ExactActionNormalizer(preprocessor,postprocessor,device)
