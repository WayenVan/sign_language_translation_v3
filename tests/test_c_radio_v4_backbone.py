from types import SimpleNamespace

import pytest
import torch
from torch import nn

from csi_slt.engine.trainability import (
    SltTrainabilityPlan,
    VisualBackboneTrainabilityPlan,
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


def test_default_layers_start_as_uniform_attention_fusion():
    encoder = _FakeEncoder()
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=encoder)
    lengths = torch.tensor([1, 1])

    output = backbone(torch.rand(2, 3, 2, 2), t_lengths=lengths)

    assert backbone.output_layers == (-1, -2, -3, -4)
    assert encoder.radio_model.calls == [
        {
            "indices": [-1, -2, -3, -4],
            "return_prefix_tokens": True,
            "norm": True,
            "output_fmt": "NLC",
            "intermediates_only": True,
            "aggregation": "sparse",
        }
    ]
    torch.testing.assert_close(output.visual_features, torch.full((2, 2, 3), -2.5))
    torch.testing.assert_close(output.pooled_visual_features, torch.full((2, 3), -2.5))
    assert output.visual_length is lengths
    assert encoder.radio_model.weight.requires_grad is False
    assert backbone.summary_layer_fusion.layer_bias.requires_grad is True
    assert backbone.feature_layer_fusion.layer_bias.requires_grad is True
    torch.testing.assert_close(
        output.extras["summary_layer_weights"], torch.full((2, 4), 0.25)
    )
    torch.testing.assert_close(
        output.extras["feature_layer_weights"], torch.full((2, 4, 2), 0.25)
    )
    (output.visual_features.sum() + output.pooled_visual_features.sum()).backward()
    assert backbone.summary_layer_fusion.score_mlp[-1].weight.grad.abs().sum() > 0
    assert backbone.feature_layer_fusion.score_mlp[-1].weight.grad.abs().sum() > 0


def test_frozen_encoder_stays_in_eval_while_fusion_trains():
    encoder = _FakeEncoder()
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=encoder)

    backbone.train()

    assert backbone.training is True
    assert backbone.summary_layer_fusion.training is True
    assert encoder.training is False


def test_trainable_adapter_does_not_implicitly_change_runtime_mode():
    encoder = _FakeEncoder()
    backbone = CRadioV4Backbone({"id": "fake"}, c_radio_v4=encoder)
    encoder.radio_model.lora_adapter = nn.Linear(1, 1, bias=False)

    backbone.train()

    assert encoder.training is False
    assert encoder.radio_model.weight.requires_grad is False
    assert encoder.radio_model.lora_adapter.weight.requires_grad is True

    backbone.set_runtime_mode("train")
    assert encoder.training is True


def test_single_output_layer_remains_supported():
    backbone = CRadioV4Backbone(
        {"id": "fake", "output_layer": -2}, c_radio_v4=_FakeEncoder()
    )

    output = backbone(torch.rand(1, 3, 2, 2))

    assert backbone.output_layers == (-2,)
    assert backbone.summary_layer_fusion is None
    assert backbone.feature_layer_fusion is None
    assert output.extras is None
    torch.testing.assert_close(output.visual_features, torch.full((1, 2, 3), -2.0))


def test_construction_starts_encoder_frozen_and_eval():
    encoder = _FakeEncoder()
    CRadioV4Backbone(
        {"id": "fake", "output_layer": [-1]},
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


@pytest.mark.parametrize(
    ("runtime_mode", "expected_training"),
    [("eval", False), ("train", True)],
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
        SltTrainabilityPlan(
            visual_backbone=VisualBackboneTrainabilityPlan(
                mode="lora", runtime_mode=runtime_mode
            )
        ),
    )
    model.train()

    encoder = backbone.visual_encoder
    assert encoder.training is expected_training
    assert not encoder.radio_model.weight.requires_grad
    assert encoder.radio_model.lora_adapter.weight.requires_grad

    encoder.radio_model.lora_adapter(torch.ones(1, 1)).sum().backward()
    assert encoder.radio_model.lora_adapter.weight.grad is not None


@pytest.mark.parametrize("output_layer", [[], [-1, -1], [-1, "-2"], True])
def test_invalid_output_layers_are_rejected(output_layer):
    with pytest.raises((TypeError, ValueError)):
        CRadioV4Backbone(
            {"id": "fake", "output_layer": output_layer},
            c_radio_v4=_FakeEncoder(),
        )
