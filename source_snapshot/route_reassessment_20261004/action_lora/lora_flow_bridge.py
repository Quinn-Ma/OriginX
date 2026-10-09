"""LoRA-only flow training and base velocity retention, sharing one VLM context.

Requires work/ on PYTHONPATH. Reuses the tested existing loss/encoding protocol,
without calling its constructor (which would unfreeze/cast original weights).
The optional teacher is this frozen base with adapters bypassed: no model copy.
Single-device, single-threaded, non-reentrant; no launcher or model load here.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from training_bridge.bridge import FrozenVLMFlowBridge, encode_sample, validate_processor
from training_bridge.train_core import rng_state
from teacher_anchor_pilot_v1.bridge import equal_rng, tensor_versions
try:
    from .action_lora import ActionLoRALinear, adapters_disabled
except ImportError:
    from action_lora import ActionLoRALinear, adapters_disabled


class LoRAFlowBridge(FrozenVLMFlowBridge):
    def __init__(self, handle, *, retention_coefficient=1.0):
        # Deliberately avoid FrozenVLMFlowBridge.__init__.
        nn.Module.__init__(self)
        if not math.isfinite(retention_coefficient) or retention_coefficient < 0:
            raise ValueError("retention_coefficient must be finite and nonnegative")
        handle.assert_integrity()
        self.handle, self.model = handle, handle.model
        config = self.model.config
        if (config.state_dim, config.action_dim, config.state_length) != (60, 60, 4):
            raise ValueError("Expected released RoboCasa state/action/history dimensions")
        self.retention_coefficient = float(retention_coefficient)
        self._in_forward = False
        self.teacher_calls = 0
        self._base_versions = self._original_parameter_versions()
        self.model.eval()
        self._assert_trainable_contract()

    def _original_parameter_versions(self):
        return [(name, id(value), value._version, tuple(value.shape), value.dtype, value.device)
                for name, value in self.model.named_parameters()
                if name not in self.handle.adapter_names]

    def _assert_trainable_contract(self):
        parameters = dict(self.model.named_parameters())
        observed = {name for name, value in parameters.items() if value.requires_grad}
        if observed != self.handle.adapter_names:
            raise RuntimeError("Trainable parameter set differs from the declared LoRA adapters")
        if any(value.dtype != torch.float32 for name, value in parameters.items() if name in observed):
            raise RuntimeError("LoRA training requires FP32 adapter masters")
        if any(value.grad is not None for name, value in parameters.items() if name not in observed):
            raise RuntimeError("A frozen original parameter has a gradient")
        if self._original_parameter_versions() != self._base_versions:
            raise RuntimeError("A frozen original parameter changed")
        modules = dict(self.model.named_modules())
        if any(type(modules[t['name']]) is not ActionLoRALinear or not modules[t['name']].adapter_enabled
               for t in self.handle.metadata['targets']):
            raise RuntimeError("All student adapters must remain installed and enabled")

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # Original modules remain eval; adapters have no dropout/train-only path.
        self.model.eval()
        self._assert_trainable_contract()
        return self

    def trainable_parameters(self):
        self._assert_trainable_contract()
        return (value for name, value in self.model.named_parameters() if name in self.handle.adapter_names)

    def forward(self, inputs, actions, valid_steps, *, noise=None, times=None):
        if self._in_forward:
            raise RuntimeError("LoRAFlowBridge is non-reentrant")
        self._assert_trainable_contract()
        self.model.eval()
        self._in_forward = True
        try:
            return super().forward(inputs, actions, valid_steps, noise=noise, times=times)
        finally:
            self._in_forward = False

    def _loss(self, inputs, actions, valid_steps, noise, times):
        if self.retention_coefficient == 0:
            # Exact original forward without an additional teacher or random draw.
            result = super()._loss(inputs, actions, valid_steps, noise, times)
            result['flow_loss'] = result['loss']
            result['retention_loss'] = result['loss'].new_zeros(())
            return result
        model = self.model
        original = model.dit_forward
        had_instance_method = 'dit_forward' in model.__dict__
        previous_instance_method = model.__dict__.get('dit_forward')
        anchors = []

        def capture(**arguments):
            student = original(**arguments)
            # All nonadapter weights are frozen, including state_projector:
            # the exact same state_embed is therefore the correct teacher input.
            before_rng = rng_state(actions.device)
            before_inputs = tensor_versions(arguments)
            with adapters_disabled(self.handle), torch.no_grad():
                teacher = original(**arguments)
            if not equal_rng(before_rng, rng_state(actions.device)):
                raise RuntimeError("Teacher consumed RNG")
            if tensor_versions(arguments) != before_inputs:
                raise RuntimeError("Teacher mutated shared context or input tensors")
            if teacher.requires_grad or teacher.grad_fn is not None:
                raise RuntimeError("Teacher retained an autograd graph")
            if teacher.shape != student.shape:
                raise RuntimeError("Teacher/student velocity shape mismatch")
            self._assert_trainable_contract()
            self.teacher_calls += 1
            mask = inputs['action_mask'].bool() & valid_steps[..., None]
            anchors.append((student.float() - teacher.float())[mask].square().mean())
            return student

        model.dit_forward = capture
        try:
            result = super()._loss(inputs, actions, valid_steps, noise, times)
        finally:
            if had_instance_method:
                model.dit_forward = previous_instance_method
            else:
                del model.dit_forward
        if len(anchors) != 1:
            raise RuntimeError("Expected exactly one student velocity call")
        result['flow_loss'] = result['loss']
        result['retention_loss'] = anchors[0]
        result['loss'] = result['flow_loss'] + self.retention_coefficient * result['retention_loss']
        return result
