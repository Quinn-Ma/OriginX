"""Contract tests on a tiny CPU model, never a claimed real-model GPU result."""
import copy
import json
import random
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_bridge.bridge import FrozenVLMFlowBridge, TRAINABLE_MODULES
from training_bridge.train_core import SingleDeviceTrainer, rng_state
from training_bridge.tests.test_bridge import TinyPolicy, batch
from training_bridge.tests.test_train_core import Provider
from teacher_anchor_pilot_v1.bridge import TeacherAnchoredFlowBridge, equal_rng
from teacher_anchor_pilot_v1.contract import control_trace, regenerate_trace
from teacher_anchor_pilot_v1.train import assert_tree_equal, reconcile_metrics

torch.set_num_threads(1)


def seed(value=193):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def models(coefficient=1.0):
    seed(32)
    base = TinyPolicy()
    return FrozenVLMFlowBridge(copy.deepcopy(base)), TeacherAnchoredFlowBridge(copy.deepcopy(base), anchor_coefficient=coefficient)


def optimizer(bridge):
    return torch.optim.AdamW(bridge.trainable_parameters(), lr=1e-5, betas=(.9,.95), eps=1e-8, weight_decay=0, foreach=False)


def test_lambda0_random_loss_gradients_optimizer_and_all_rng_are_exact():
    control, candidate = models(0)
    inputs, actions, valid = batch()
    observations = []
    for bridge in [control, candidate]:
        opt = optimizer(bridge); bridge.train(); seed()
        output = bridge(inputs, actions, valid)
        output['loss'].backward()
        gradients = {n:p.grad.clone() for n,p in bridge.model.named_parameters() if p.requires_grad}
        torch.nn.utils.clip_grad_norm_(list(bridge.trainable_parameters()), 1)
        opt.step()
        observations.append((output, gradients, copy.deepcopy(bridge.model.state_dict()), copy.deepcopy(opt.state_dict()), rng_state('cpu')))
    assert_tree_equal(observations[0], observations[1])
    assert candidate.teacher_calls == 0


def test_same_weights_zero_anchor_one_vlm_call_and_exact_shared_noise_time_cache():
    _, candidate = models()
    candidate.train()
    observed = []
    original = candidate.model.dit_forward
    def student(**kwargs):
        observed.append(('student', kwargs))
        return original(**kwargs)
    original_teacher = candidate.teacher._velocity_function
    def teacher(holder, **kwargs):
        observed.append(('teacher', kwargs))
        return original_teacher(holder, **kwargs)
    candidate.model.dit_forward = student
    candidate.teacher._velocity_function = teacher
    calls = []
    hook = candidate.model.vlm.model.register_forward_hook(lambda *args: calls.append(1))
    seed(); result = candidate(*batch()); hook.remove()
    assert result['anchor_loss'].item() == 0
    assert len(calls) == 1 and candidate.teacher_calls == 1
    student_args, teacher_args = observed[0][1], observed[1][1]
    for key in ['noisy_action', 't', 'action_mask', 'position_embeds', 'past_key_values', 'attn_mask']:
        assert student_args[key] is teacher_args[key]
    assert student_args['state_embed'] is not teacher_args['state_embed']
    assert_tree_equal(student_args['state_embed'], teacher_args['state_embed'])
    assert candidate.model.dit_forward is student


def test_added_teacher_does_not_change_original_noise_or_rng_schedule():
    control, candidate = models()
    inputs, actions, valid = batch()
    seed(); a = control(inputs, actions, valid); control_rng = rng_state('cpu')
    seed(); b = candidate(inputs, actions, valid); candidate_rng = rng_state('cpu')
    assert_tree_equal(a['loss'], b['flow_loss'])
    assert equal_rng(control_rng, candidate_rng)


@pytest.mark.parametrize('coefficient', [0.0, 1.0])
def test_cpu_bf16_autocast_matches_control_arithmetic_and_noise_dtype(coefficient):
    control, candidate = models(coefficient)
    inputs, actions, valid = batch()
    inputs['state'] = inputs['state'].bfloat16()
    inputs['action_mask'] = inputs['action_mask'].bfloat16()
    outputs = []
    for bridge in [control, candidate]:
        seed()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = bridge(inputs, actions, valid)
        result['loss'].backward()
        outputs.append((result['loss'], {name:p.grad.clone() for name,p in bridge.model.named_parameters() if p.requires_grad}, rng_state('cpu')))
    assert_tree_equal(outputs[0], outputs[1])
    if coefficient:
        assert result['anchor_loss'].item() == 0


def test_teacher_remains_unchanged_and_gradient_free_after_student_updates():
    _, candidate = models()
    candidate.train()
    before = copy.deepcopy(candidate.teacher.state_dict())
    opt = optimizer(candidate)
    seed()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        candidate(*batch())['loss'].backward()
        assert all(p.grad is None and not p.requires_grad for p in candidate.teacher.parameters())
        assert all(p.grad is None and not p.requires_grad for p in candidate.model.vlm.parameters())
        assert all(p.grad is not None for p in candidate.trainable_parameters())
        opt.step(); candidate.teacher.assert_frozen()
    assert_tree_equal(candidate.teacher.state_dict(), before)
    assert not any(key.startswith('teacher.') for key in candidate.model.state_dict())


def test_perturbed_student_anchor_gradient_pulls_active_values_to_teacher():
    _, candidate = models()
    with torch.no_grad():
        candidate.model.action_output_layer.bias[:12].add_(.25)
    result = candidate(*batch())
    assert result['anchor_loss'].item() > 0
    result['anchor_loss'].backward()
    gradient = candidate.model.action_output_layer.bias.grad
    assert (gradient[:12] > 0).all()
    assert torch.equal(gradient[12:], torch.zeros_like(gradient[12:]))
    assert all(p.grad is None for p in candidate.teacher.parameters())


def test_teacher_uses_independent_state_projector():
    _, candidate = models()
    with torch.no_grad():
        candidate.model.state_projector.bias.add_(1)
    result = candidate(*batch())
    assert result['anchor_loss'].item() > 0
    result['anchor_loss'].backward()
    assert candidate.model.state_projector.bias.grad.abs().sum().item() > 0
    assert candidate.teacher.state_projector.bias.grad is None


@pytest.mark.parametrize('mutation', ['rng', 'cache', 'teacher_buffer'])
def test_teacher_side_effects_fail_and_restore_student_method(mutation):
    _, candidate = models()
    original = candidate.teacher._velocity_function
    def corrupt(holder, **kwargs):
        if mutation == 'rng':
            torch.rand(1)
        elif mutation == 'cache':
            kwargs['past_key_values'][0][0].add_(1)
        else:
            holder.sink.weight.add_(1)
        return original(holder, **kwargs)
    candidate.teacher._velocity_function = corrupt
    with pytest.raises(RuntimeError, match='RNG|mutated|changed'):
        candidate(*batch())
    assert 'dit_forward' not in candidate.model.__dict__
    assert candidate._intercepting is False


def test_first_real_teacher_call_checks_cache_values_even_without_version_change():
    _, candidate = models()
    original = candidate.teacher._velocity_function
    def corrupt(holder, **kwargs):
        kwargs['past_key_values'][0][0].data.add_(1)
        return original(holder, **kwargs)
    candidate.teacher._velocity_function = corrupt
    with pytest.raises(RuntimeError, match='shared tensor values'):
        candidate(*batch())
    assert candidate.teacher.shared_value_audit_complete is False


def tiny_trainer():
    seed(77)
    bridge = TeacherAnchoredFlowBridge(TinyPolicy())
    trainer = SingleDeviceTrainer(bridge, optimizer(bridge), Provider(), identity={'pilot': 'frozen'}, device='cpu', accumulate=8)
    seed()
    return trainer


def test_anchor_resume_keeps_base_teacher_and_matches_next_update(tmp_path):
    original = tiny_trainer()
    teacher = copy.deepcopy(original.bridge.teacher.state_dict())
    original.step()
    checkpoint = original.save(tmp_path / 'checkpoint-step-00000001.pt')
    expected_event = original.step()
    expected_state = copy.deepcopy(original.bridge.model.state_dict())
    restarted = tiny_trainer()
    restarted.load(checkpoint)
    assert_tree_equal(restarted.bridge.teacher.state_dict(), teacher)
    assert restarted.step() == expected_event
    assert_tree_equal(restarted.bridge.model.state_dict(), expected_state)
    assert_tree_equal(restarted.bridge.teacher.state_dict(), teacher)


def test_full_trace_rejects_missing_or_repeated_optimizer_steps(tmp_path):
    metrics = tmp_path / 'metrics.jsonl'
    rows = [dict(step=i, samples=[{'draw': (i-1)*8+j+1} for j in range(8)]) for i in range(1,201)]
    metrics.write_text('\n'.join(json.dumps(row) for row in rows))
    trace, _ = control_trace(metrics)
    assert len(trace) == 1600 and trace[-1]['draw'] == 1600
    rows[-1]['step'] = 199
    metrics.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match='200'):
        control_trace(metrics)


def test_regenerated_sampler_matches_original_three_draw_order():
    manifest = {'seed':193, 'datasets':[{'task':'A','episode_ids':[8,2]}, {'task':'B','episode_ids':[4]}]}
    lengths = {'A':{8:9,2:4}, 'B':{4:20}}
    expected = []; rng = random.Random(193)
    for index in range(1600):
        dataset = manifest['datasets'][rng.randrange(2)]
        episode = dataset['episode_ids'][rng.randrange(len(dataset['episode_ids']))]
        expected.append(dict(task=dataset['task'], episode=episode, timestep=rng.randrange(lengths[dataset['task']][episode]), draw=index+1))
    assert regenerate_trace(manifest, lengths) == expected


def test_no_cuda_context_initialized_by_tests():
    assert not torch.cuda.is_initialized()


def test_resume_archives_incomplete_tail_without_duplicate_metric_steps(tmp_path):
    prefix = ''.join(json.dumps({'step':step})+'\n' for step in range(1,51)).encode()
    tail = b'{"step":51}\n{"step":52}\n{"step":53,"partial":'
    (tmp_path / 'metrics.jsonl').write_bytes(prefix + tail)
    reconcile_metrics(tmp_path, 50)
    assert (tmp_path / 'metrics.jsonl').read_bytes() == prefix
    archives = list(tmp_path.glob('metrics-abandoned-*.jsonl'))
    assert len(archives) == 1 and archives[0].read_bytes() == tail
    reconcile_metrics(tmp_path, 50)
    assert len(list(tmp_path.glob('metrics-abandoned-*.jsonl'))) == 1
    with pytest.raises(ValueError, match='new output'):
        reconcile_metrics(tmp_path, 0)
