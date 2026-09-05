from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from csi_slt.engine.trainability import (
    SltTrainabilityPlan,
    apply_trainability_plan,
)
from csi_slt.modeling_slt.visual_backbones.c_radio_v4_backbone import (
    CRadioV4Backbone,
)


class _FakeRadioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward_intermediates(self, x, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                summary=x.new_full((x.shape[0], 3), float(index)),
                features=x.new_full((x.shape[0], 2, 3), float(index)),
            )
            for index in kwargs["indices"]
        ]


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.radio_model = _FakeRadioModel()
        self.config = SimpleNamespace()


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.scale = 0.5
        self.qkv = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(self, hidden_states):
        return hidden_states


class _AttentionFakeRadioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeAttention(), _FakeAttention()])

    def forward_intermediates(self, x, **kwargs):
        hidden_states = x.new_ones((x.shape[0], 3, 4))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return [
            SimpleNamespace(
                summary=hidden_states[:, 0],
                features=hidden_states[:, 1:],
            )
            for _ in kwargs["indices"]
        ]


class _AttentionFakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.radio_model = _AttentionFakeRadioModel()
        self.config = SimpleNamespace()


def test_default_feature_and_attention_layers_are_independent():
    encoder = _FakeEncoder()
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=encoder)
    lengths = torch.tensor([1, 1])

    output = backbone(torch.rand(2, 3, 2, 2), t_lengths=lengths)

    assert backbone.output_layer == -8
    assert backbone.attention_layer == -25
    assert encoder.radio_model.calls == [
        {
            "indices": [-8],
            "return_prefix_tokens": True,
            "norm": True,
            "output_fmt": "NLC",
            "intermediates_only": True,
            "aggregation": "sparse",
        }
    ]
    torch.testing.assert_close(output.visual_features, torch.full((2, 2, 3), -8.0))
    torch.testing.assert_close(output.pooled_visual_features, torch.full((2, 3), -8.0))
    assert output.visual_length is lengths
    assert encoder.radio_model.weight.requires_grad is False
    assert output.extras is None


def test_frozen_encoder_stays_in_eval_while_backbone_trains():
    encoder = _FakeEncoder()
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=encoder)

    backbone.train()

    assert backbone.training is True
    assert encoder.training is False


def test_trainable_adapter_does_not_implicitly_change_runtime_mode():
    encoder = _FakeEncoder()
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=encoder)
    encoder.radio_model.lora_adapter = nn.Linear(1, 1, bias=False)

    backbone.train()

    assert encoder.training is False
    assert encoder.radio_model.weight.requires_grad is False
    assert encoder.radio_model.lora_adapter.weight.requires_grad is True

    backbone.set_runtime_mode("follow")
    assert encoder.training is True


def test_single_output_layer_remains_supported():
    backbone = CRadioV4Backbone(
        {"id": "fake", "output_layer": -2}, c_radio_v4=_FakeEncoder()
    )

    output = backbone(torch.rand(1, 3, 2, 2))

    assert backbone.output_layer == -2
    assert output.extras is None
    torch.testing.assert_close(output.visual_features, torch.full((1, 2, 3), -2.0))


@pytest.mark.parametrize("output_layer", [[-2], OmegaConf.create([-2])])
def test_legacy_single_item_output_layer_list_is_unwrapped(output_layer):
    encoder = _FakeEncoder()
    backbone = CRadioV4Backbone(
        {"id": "fake", "output_layer": output_layer}, c_radio_v4=encoder
    )

    output = backbone(torch.rand(1, 3, 2, 2))

    assert backbone.output_layer == -2
    assert encoder.radio_model.calls[0]["indices"] == [-2]
    torch.testing.assert_close(output.visual_features, torch.full((1, 2, 3), -2.0))


def test_attention_maps_are_opt_in_cls_to_patch_probabilities():
    backbone = CRadioV4Backbone(
        {"id": "fake", "output_layer": -1, "attention_layer": -1},
        c_radio_v4=_AttentionFakeEncoder(),
    )

    output = backbone(
        torch.rand(2, 3, 2, 2),
        return_attention_maps=True,
    )

    assert output.extras["attention_maps"].shape == (2, 2, 2)
    assert output.extras["attention_layer"] == -1
    assert torch.isfinite(output.extras["attention_maps"]).all()


def test_attention_layer_is_independent_from_output_layer():
    backbone = CRadioV4Backbone(
        {"id": "fake", "output_layer": -1, "attention_layer": -2},
        c_radio_v4=_AttentionFakeEncoder(),
    )

    output = backbone(torch.rand(1, 3, 2, 2), return_attention_maps=True)

    assert output.extras["attention_layer"] == -2
    assert output.extras["attention_maps"].shape == (1, 2, 2)


def test_construction_starts_encoder_frozen_and_eval():
    encoder = _FakeEncoder()
    CRadioV4Backbone(
        {"id": "fake", "output_layer": -1},
        c_radio_v4=encoder,
    )

    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_retired_freeze_visual_encoder_config_is_ignored(caplog):
    encoder = _FakeEncoder()

    CRadioV4Backbone(
        {"id": "fake", "freeze_visual_encoder": False},
        c_radio_v4=encoder,
    )

    assert "Ignoring retired C-RADIO freeze_visual_encoder=False" in caplog.text
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_runtime_mode_is_validated():
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=_FakeEncoder())

    with pytest.raises(ValueError, match="runtime_mode"):
        backbone.set_runtime_mode("auto")
    with pytest.raises(ValueError, match="runtime_mode"):
        backbone.set_runtime_mode("train")


@pytest.mark.parametrize(
    ("runtime_mode", "expected_training"),
    [("eval", False), ("follow", True)],
)
def test_engine_plan_controls_lora_runtime_mode(runtime_mode, expected_training):
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=_FakeEncoder())
    backbone.visual_encoder.radio_model.lora_adapter = nn.Linear(1, 1, bias=False)

    model = nn.Module()
    model.llm = nn.Linear(1, 1)
    model.visual_backbone = backbone
    model.visual_adapter = nn.Linear(1, 1)
    model.ctc_head = None
    model.visual_position_embedding = None

    apply_trainability_plan(
        model,
        SltTrainabilityPlan.from_mapping(
            {
                "llm": {"parameter_mode": "frozen"},
                "visual_backbone": {
                    "parameter_mode": "lora",
                    "runtime_mode": runtime_mode,
                },
                "visual_adapter": {"parameter_mode": "frozen"},
                "ctc_head": {"parameter_mode": "frozen"},
                "ctc_codebook": {"parameter_mode": "frozen"},
                "visual_position_embedding": {"parameter_mode": "frozen"},
                "visual_boundary_embeddings": {"parameter_mode": "frozen"},
            }
        ),
    )
    model.train()

    encoder = backbone.visual_encoder
    assert encoder.training is expected_training
    assert not encoder.radio_model.weight.requires_grad
    assert encoder.radio_model.lora_adapter.weight.requires_grad

    encoder.radio_model.lora_adapter(torch.ones(1, 1)).sum().backward()
    assert encoder.radio_model.lora_adapter.weight.grad is not None


@pytest.mark.parametrize("output_layer", [[], [-1, -2], ["-1"], "-1", True])
def test_invalid_output_layers_are_rejected(output_layer):
    with pytest.raises((TypeError, ValueError)):
        CRadioV4Backbone(
            {"id": "fake", "output_layer": output_layer},
            c_radio_v4=_FakeEncoder(),
        )
