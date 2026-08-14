import torch

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v3 import (
    DINOFrameAdapterCrossV2,
)


def test_v3_builds_three_frame_templates_per_video():
    lengths = torch.tensor([2, 4])
    total_frames = int(lengths.sum())
    backbone_output = VisualBackboneOutput(
        visual_features=torch.randn(total_frames, 4, 8),
        pooled_visual_features=torch.randn(total_frames, 10),
        visual_length=lengths,
    )
    adapter = DINOFrameAdapterCrossV2(
        input_dim=8,
        cls_input_dim=10,
        output_dim=12,
        spatial_window_radius=1,
    )

    output = adapter(backbone_output)

    assert output.visual_length.tolist() == [2, 4]
    assert output.visual_features.shape == (6, 12)
    assert output.position_ids.tolist() == [0, 0, 0, 0, 1, 1]
    assert output.extras["patch_weights"].shape == (3, 12)
    assert torch.allclose(
        output.extras["patch_weights"].sum(dim=1), torch.ones(3)
    )
