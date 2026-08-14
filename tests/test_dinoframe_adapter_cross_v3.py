import torch

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v3 import (
    DINOFrameAdapterCrossV3,
)


def test_v3_builds_three_frame_templates_per_video():
    lengths = torch.tensor([2, 4])
    total_frames = int(lengths.sum())
    backbone_output = VisualBackboneOutput(
        visual_features=torch.randn(total_frames, 4, 8),
        pooled_visual_features=torch.randn(total_frames, 10),
        visual_length=lengths,
    )
    adapter = DINOFrameAdapterCrossV3(
        input_dim=8,
        cls_input_dim=10,
        output_dim=12,
        num_queries=2,
        num_attention_heads=2,
        spatial_window_radius=1,
    )

    output = adapter(backbone_output)

    assert output.visual_length.tolist() == [2, 4]
    assert output.visual_features.shape == (6, 12)
    assert output.position_ids.tolist() == [0, 0, 0, 0, 1, 1]
    assert output.extras["temporal_attention_weights"].shape == (3, 2, 2, 12)
    assert output.extras["patch_attention_weights"].shape == (3, 2, 2, 12)

    temporal_mask = output.extras["temporal_source_valid_mask"]
    assert temporal_mask.shape == (3, 12)
    assert not temporal_mask[0, 8:].any()
    assert not temporal_mask[2, 8:].any()
    assert torch.equal(
        output.extras["temporal_attention_weights"][0, :, :, 8:],
        torch.zeros(2, 2, 4),
    )


def test_v3_shares_queries_but_keeps_attention_modules_independent():
    adapter = DINOFrameAdapterCrossV3(
        input_dim=8,
        output_dim=12,
        num_queries=2,
        num_attention_heads=2,
    )

    assert adapter.query_bank.queries.shape == (1, 2, 12)
    assert adapter.temporal_cross_attention is not adapter.patch_cross_attention


def test_v3_even_window_uses_two_middle_cls_frames_as_residual():
    lengths = torch.tensor([2])
    cls_features = torch.randn(2, 8)
    backbone_output = VisualBackboneOutput(
        visual_features=torch.randn(2, 4, 8),
        pooled_visual_features=cls_features,
        visual_length=lengths,
    )
    adapter = DINOFrameAdapterCrossV3(
        input_dim=8,
        output_dim=8,
        num_queries=2,
        num_attention_heads=2,
        temporal_window_size=2,
        temporal_window_stride=2,
        spatial_window_radius=1,
    )
    for parameter in adapter.cls_window_mlp.parameters():
        torch.nn.init.zeros_(parameter)

    output = adapter(backbone_output)

    mapped_cls = adapter.cls_frame_projection(adapter.cls_frame_norm(cls_features))
    expected_residual = mapped_cls.mean(dim=0, keepdim=True)
    torch.testing.assert_close(output.extras["cls_context"], expected_residual)
    assert output.extras["temporal_attention_weights"].shape == (1, 2, 2, 8)
