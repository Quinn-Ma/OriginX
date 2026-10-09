from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training_bridge.bridge import FrozenVLMFlowBridge, encode_sample, gradient_report


class TinyContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 8)

    def forward(self, input_ids, attention_mask, use_cache):
        hidden = self.embed(input_ids)
        positions = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return SimpleNamespace(position_ids=positions, past_key_values=[(hidden, hidden)])


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyContext()
        self.unused_lm_head = nn.Linear(8, 10)


class TinyPolicy(nn.Module):
    """Exercise graph/mask contracts without pretending to test the real HF model."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(state_dim=60, action_dim=60, state_length=4)
        self.vlm = TinyVLM()
        self.dit = nn.Linear(8, 8)
        self.state_projector = nn.Linear(60, 8)
        self.action_projector = nn.Linear(60, 8)
        self.action_output_layer = nn.Linear(8, 60)
        self.t_embedder = nn.Linear(1, 8)
        self.t_projector = nn.Linear(8, 8)
        self.sink = nn.Embedding(1, 8)

    def rotary_emb(self, mask, positions):
        self.observed_positions = positions.detach().clone()
        return (positions, positions)

    def dit_forward(self, noisy_action, t, action_mask, state_embed, position_embeds,
                    past_key_values, attn_mask):
        self.observed_mask = attn_mask.detach().clone()
        hidden = (self.action_projector(noisy_action * action_mask)
                  + state_embed.mean(1, keepdim=True)
                  + self.t_projector(self.t_embedder(t)) + self.sink.weight[None]
                  + past_key_values[0][0].mean(1, keepdim=True))
        return self.action_output_layer(self.dit(hidden).tanh())


def batch():
    state = torch.zeros(2, 4, 60)
    state[..., :14] = torch.randn(2, 4, 14)
    mask = torch.zeros(2, 16, 60)
    mask[..., :12] = 1
    inputs = {"state": state, "action_mask": mask,
              "input_ids": torch.tensor([[1, 2, 3], [3, 4, 0]]),
              "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])}
    actions = torch.randn(2, 16, 60)
    valid = torch.zeros(2, 16, dtype=torch.bool)
    valid[0, :16] = True
    valid[1, :3] = True
    return inputs, actions, valid


def test_gradients_reach_all_heads_and_vlm_remains_frozen():
    torch.manual_seed(31)
    model = TinyPolicy()
    bridge = FrozenVLMFlowBridge(model).train()
    assert not model.vlm.training
    inputs, actions, valid = batch()
    result = bridge(inputs, actions, valid)
    assert result["active_values"].item() == (16 + 3) * 12
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    report = gradient_report(bridge)
    assert report["vlm"]["gradient_tensors"] == 0
    assert all(value["gradient_l2"] > 0 for key, value in report.items() if key != "vlm")
    # Exact official causal/query layout including VLM padding and no +10 shift.
    assert model.observed_mask.shape == (2, 1, 21, 24)
    assert not model.observed_mask[1, :, :, 2].any()
    assert torch.equal(model.observed_mask[0, 0, :, 3:], torch.ones(21, 21).tril().bool())
    assert torch.equal(model.observed_positions[0, 0], torch.arange(3, 24))


def test_inactive_dimensions_and_padded_future_do_not_contribute_to_loss():
    torch.manual_seed(71)
    bridge = FrozenVLMFlowBridge(TinyPolicy())
    inputs, actions, valid = batch()
    noise, times = torch.randn_like(actions), torch.full((2, 1, 1), 0.37)
    initial = bridge(inputs, actions, valid, noise=noise, times=times)["loss"]
    changed = actions.clone()
    changed[..., 12:] = 10_000
    changed[1, 3:, :12] = 20_000
    changed_noise = noise.clone()
    changed_noise[..., 12:] = -10_000
    other = bridge(inputs, changed, valid, noise=changed_noise, times=times)["loss"]
    torch.testing.assert_close(initial, other, rtol=0, atol=0)


def test_empty_supervision_and_accidental_processor_broadcast_are_rejected():
    bridge = FrozenVLMFlowBridge(TinyPolicy())
    inputs, actions, valid = batch()
    with pytest.raises(ValueError, match="at least one"):
        bridge(inputs, actions, torch.zeros_like(valid))
    inputs["action_mask"] = inputs["action_mask"][:1]
    with pytest.raises(ValueError, match="processor defaults"):
        bridge(inputs, actions, valid)


def test_encoding_matches_server_float_cast_and_preserves_integer_inputs():
    mean, std = torch.zeros(1, 16, 60), torch.zeros(1, 16, 60)
    std[..., :12] = 1
    class Processor:
        action_config = {"robocasa365": {"mean": mean, "std": std}}
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[2, 3]]),
                    "video_grid_thw": torch.tensor([[2, 16, 16]]),
                    "pixel_values_videos": torch.ones(4, 8, dtype=torch.float32),
                    "state": torch.zeros(1, 4, 60, dtype=torch.bfloat16),
                    "action_mask": std}
    sample = SimpleNamespace(messages=[], state=torch.zeros(1, 4, 60),
                             action=torch.zeros(1, 16, 60), valid_steps=torch.ones(1, 16, dtype=torch.bool))
    inputs, actions, valid = encode_sample(sample, Processor(), "cpu", floating_dtype=torch.bfloat16)
    assert inputs["pixel_values_videos"].dtype == inputs["state"].dtype == inputs["action_mask"].dtype == torch.bfloat16
    assert inputs["input_ids"].dtype == inputs["video_grid_thw"].dtype == torch.int64
    assert actions.dtype == torch.float32 and valid.dtype == torch.bool
