import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from csi_slt.modeling_slt.visual_backbones.siglip2_backbone import Siglip2Backbone


def _tiny_encoder() -> SiglipVisionModel:
    return SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        )
    )


def test_single_layer_returns_native_patch_and_pooler_outputs():
    encoder = _tiny_encoder()
    backbone = Siglip2Backbone({"id": "fake"}, siglip2=encoder)
    frames = torch.randn(2, 3, 8, 8)
    lengths = torch.tensor([1, 1])

    expected = encoder(frames, output_hidden_states=True)
    output = backbone(frames, t_lengths=lengths)

    assert output.visual_features.shape == (2, 4, 8)
    assert output.pooled_visual_features.shape == (2, 8)
    torch.testing.assert_close(output.visual_features, expected.last_hidden_state)
    torch.testing.assert_close(output.pooled_visual_features, expected.pooler_output)
    assert output.visual_length is lengths
    assert output.extras is None
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_multiple_layers_start_as_uniform_attention_fusion():
    encoder = _tiny_encoder()
    backbone = Siglip2Backbone(
        {"id": "fake", "output_layer": [-1, -2]}, siglip2=encoder
    )
    frames = torch.randn(2, 3, 8, 8)

    output = backbone(frames, t_lengths=torch.tensor([2]))

    assert output.visual_features.shape == (2, 4, 8)
    assert output.pooled_visual_features.shape == (2, 8)
    torch.testing.assert_close(
        output.extras["summary_layer_weights"], torch.full((2, 2), 0.5)
    )
    torch.testing.assert_close(
        output.extras["feature_layer_weights"], torch.full((2, 2, 4), 0.5)
    )
    assert backbone.summary_layer_fusion.layer_bias.requires_grad
    assert backbone.feature_layer_fusion.layer_bias.requires_grad


def test_frozen_encoder_stays_in_eval_while_fusion_trains():
    encoder = _tiny_encoder()
    backbone = Siglip2Backbone(
        {"id": "fake", "output_layer": [-1, -2]}, siglip2=encoder
    )

    backbone.train()

    assert backbone.training
    assert backbone.feature_layer_fusion.training
    assert not encoder.training


def test_encoder_can_be_unfrozen_by_backbone_config():
    encoder = _tiny_encoder()
    backbone = Siglip2Backbone(
        {"id": "fake", "freeze_visual_encoder": False}, siglip2=encoder
    )

    assert all(parameter.requires_grad for parameter in encoder.parameters())
    backbone.train()
    assert encoder.training


@pytest.mark.parametrize("output_layer", [[], [-1, -1], [-1, "-2"], True])
def test_invalid_output_layers_are_rejected(output_layer):
    with pytest.raises((TypeError, ValueError)):
        Siglip2Backbone(
            {"id": "fake", "output_layer": output_layer}, siglip2=_tiny_encoder()
        )


def test_input_size_and_packed_lengths_are_validated():
    backbone = Siglip2Backbone({"id": "fake"}, siglip2=_tiny_encoder())

    with pytest.raises(ValueError, match="spatial size"):
        backbone(torch.randn(2, 3, 12, 12), torch.tensor([2]))
    with pytest.raises(ValueError, match=r"t_lengths.sum\(\)"):
        backbone(torch.randn(2, 3, 8, 8), torch.tensor([1]))


def test_position_interpolation_allows_other_image_sizes():
    backbone = Siglip2Backbone(
        {"id": "fake", "interpolate_pos_encoding": True},
        siglip2=_tiny_encoder(),
    )

    output = backbone(torch.randn(1, 3, 12, 12), torch.tensor([1]))

    assert output.visual_features.shape == (1, 9, 8)
