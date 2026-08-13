from types import SimpleNamespace

import pytest
import torch
from torch import nn

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


def test_default_layers_are_mean_fused_without_changing_feature_width():
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
    torch.testing.assert_close(
        output.visual_features, torch.full((2, 2, 3), -2.5)
    )
    torch.testing.assert_close(
        output.pooled_visual_features, torch.full((2, 3), -2.5)
    )
    assert output.visual_length is lengths
    assert encoder.radio_model.weight.requires_grad is False


def test_single_output_layer_remains_supported():
    backbone = CRadioV4Backbone(
        {"id": "fake", "output_layer": -2}, c_radio_v4=_FakeEncoder()
    )

    output = backbone(torch.rand(1, 3, 2, 2))

    assert backbone.output_layers == (-2,)
    torch.testing.assert_close(
        output.visual_features, torch.full((1, 2, 3), -2.0)
    )


@pytest.mark.parametrize("output_layer", [[], [-1, -1], [-1, "-2"], True])
def test_invalid_output_layers_are_rejected(output_layer):
    with pytest.raises((TypeError, ValueError)):
        CRadioV4Backbone(
            {"id": "fake", "output_layer": output_layer},
            c_radio_v4=_FakeEncoder(),
        )
