"""Add a frozen action-expert target to the unchanged control flow forward.

The teacher owns independent copies of the seven action modules, not a second
VLM. The control bridge alone draws noise/time and builds the frozen VLM cache.
A temporary, non-reentrant method interception sees those exact tensors and
leaves the control implementation and exported HF model unchanged.
"""
import copy
import math

import numpy as np
import torch
from torch import nn

from training_bridge.bridge import FrozenVLMFlowBridge, TRAINABLE_MODULES
from training_bridge.train_core import rng_state


def equal_rng(left, right):
    if left.keys() != right.keys() or left['python'] != right['python']:
        return False
    a, b = left['numpy'], right['numpy']
    if a[0] != b[0] or not np.array_equal(a[1], b[1]) or a[2:] != b[2:]:
        return False
    return all(torch.equal(left[k], right[k]) for k in left if k.startswith('torch_'))


def tree_tensors(value):
    """Read-only traversal of the pinned HF cache and input trees."""
    rows = []
    def walk(item):
        if isinstance(item, torch.Tensor):
            rows.append(item)
        elif isinstance(item, dict):
            for key in sorted(item):
                walk(item[key])
        elif isinstance(item, (tuple, list)):
            for part in item:
                walk(part)
        elif item is not None:
            # transformers DynamicCache exposes its read-only layer pairs.
            if not hasattr(item, '__len__') or not hasattr(item, '__getitem__'):
                raise TypeError('Unsupported cache/input tree: ' + type(item).__name__)
            for index in range(len(item)):
                walk(item[index])
    walk(value)
    return rows


def tensor_versions(value):
    return [(id(item), item.data_ptr(), item._version, tuple(item.shape), item.dtype)
            for item in tree_tensors(value)]


class FrozenActionTeacher(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Called after student BF16 checkpoint values become FP32 masters.
        for name in TRAINABLE_MODULES:
            self.add_module(name, copy.deepcopy(getattr(model, name)))
        self._velocity_function = type(model).dit_forward
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        self.eval()
        students = {p.data_ptr() for p in model.parameters()}
        if any(p.data_ptr() in students for p in self.parameters()):
            raise ValueError('Teacher parameters alias the student')
        self._frozen_versions = self.versions()
        self.shared_value_audit_complete = False

    def versions(self):
        return tensor_versions(list(self.parameters()) + list(self.buffers()))

    def assert_frozen(self):
        if self.training or any(m.training for m in self.modules()):
            raise RuntimeError('Teacher was placed in training mode')
        if any(p.requires_grad or p.grad is not None for p in self.parameters()):
            raise RuntimeError('Teacher gradient isolation was violated')
        if self.versions() != self._frozen_versions:
            raise RuntimeError('Teacher parameter or buffer changed')

    def forward(self, state, arguments):
        self.assert_frozen()
        before_rng = rng_state(state.device)
        before_inputs = tensor_versions([state, arguments])
        # One real forward per process also checks values, covering writes that
        # bypass a PyTorch version counter. No host transfer or RNG is involved.
        # Subsequent forwards retain the inexpensive always-on version guards.
        values = None
        if not self.shared_value_audit_complete:
            unique = {id(tensor): tensor for tensor in tree_tensors([state, arguments])}
            values = [(tensor, tensor.detach().clone()) for tensor in unique.values()]
        with torch.no_grad():
            teacher_arguments = dict(arguments)
            # The teacher must use its own projector; student's embedding would
            # silently remove the anchor from the state projector.
            teacher_arguments['state_embed'] = self.state_projector(state)
            prediction = self._velocity_function(self, **teacher_arguments)
        if not equal_rng(before_rng, rng_state(state.device)):
            raise RuntimeError('Teacher consumed RNG; the matched noise schedule is invalid')
        if tensor_versions([state, arguments]) != before_inputs:
            raise RuntimeError('Teacher mutated the shared cache or inputs')
        if values is not None:
            if any(not torch.equal(tensor, saved) for tensor, saved in values):
                raise RuntimeError('Teacher changed shared tensor values without a version update')
            self.shared_value_audit_complete = True
        self.assert_frozen()
        if prediction.requires_grad or prediction.grad_fn is not None:
            raise RuntimeError('Teacher output retained a gradient graph')
        return prediction


class TeacherAnchoredFlowBridge(FrozenVLMFlowBridge):
    def __init__(self, model, *, anchor_coefficient=1.0):
        if not math.isfinite(anchor_coefficient) or anchor_coefficient < 0:
            raise ValueError('Anchor coefficient must be finite and nonnegative')
        super().__init__(model, fp32_trainable=True)
        self.anchor_coefficient = float(anchor_coefficient)
        self.teacher = FrozenActionTeacher(model)
        self._intercepting = False
        self.teacher_calls = 0

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _loss(self, inputs, actions, valid_steps, noise, times):
        # Exact ablation: execute the original control path without interception,
        # extra draws or even a numerically redundant loss addition.
        if self.anchor_coefficient == 0:
            return super()._loss(inputs, actions, valid_steps, noise, times)
        if self._intercepting:
            raise RuntimeError('Training bridge is single-device and non-reentrant')
        self._intercepting = True
        model = self.model
        original = model.dit_forward
        had_instance_method = 'dit_forward' in model.__dict__
        previous_instance_method = model.__dict__.get('dit_forward')
        anchors = []

        def capture(**arguments):
            prediction = original(**arguments)
            teacher_prediction = self.teacher(inputs['state'], arguments)
            if teacher_prediction.shape != prediction.shape:
                raise ValueError('Teacher velocity shape differs')
            mask = inputs['action_mask'].bool() & valid_steps[..., None]
            anchors.append((prediction.float() - teacher_prediction.float())[mask].square().mean())
            self.teacher_calls += 1
            return prediction

        model.dit_forward = capture
        try:
            result = super()._loss(inputs, actions, valid_steps, noise, times)
        finally:
            if had_instance_method:
                model.dit_forward = previous_instance_method
            else:
                del model.dit_forward
            self._intercepting = False
        if len(anchors) != 1:
            raise RuntimeError('Expected exactly one original velocity evaluation')
        result['flow_loss'] = result['loss']
        result['anchor_loss'] = anchors[0]
        result['loss'] = result['flow_loss'] + self.anchor_coefficient * result['anchor_loss']
        return result
