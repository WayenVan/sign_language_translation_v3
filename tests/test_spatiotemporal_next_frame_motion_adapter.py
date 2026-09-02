import pytest
import torch

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_motion_adapter import (
    MotionTemporalFusion,
    NextFramePatchFusion,
    SpatiotemporalNextFrameMotionAdapter,
)


def _backbone_output(
    patches: torch.Tensor,
    lengths: tuple[int, ...] = (2, 4),
) -> VisualBackboneOutput:
    return VisualBackboneOutput(
        visual_features=patches,
        pooled_visual_features=torch.randn(patches.shape[0], 11),
        visual_length=torch.tensor(lengths),
    )


def _adapter(projection_rank: int | None = 3):
    return SpatiotemporalNextFrameMotionAdapter(
        input_dim=4,
        output_dim=6,
        projection_rank=projection_rank,
        patch_grid_size=(2, 2),
        patch_fusion_hidden_dim=5,
        motion_hidden_dim=7,
    )


def test_adapter_and_both_fusions_are_owned_by_the_new_module() -> None:
    module_name = (
        "csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_motion_adapter"
    )

    assert NextFramePatchFusion.__module__ == module_name
    assert MotionTemporalFusion.__module__ == module_name
    assert SpatiotemporalNextFrameMotionAdapter.__module__ == module_name
    assert (
        VISUAL_ADAPTERS["spatiotemporal_next_frame_motion"]
        is SpatiotemporalNextFrameMotionAdapter
    )


def test_forward_is_patch_fusion_then_temporal_fusion_then_projection() -> None:
    torch.manual_seed(0)
    patches = torch.randn(6, 4, 4, requires_grad=True)
    backbone_output = _backbone_output(patches)
    adapter = _adapter()

    output = adapter(backbone_output)
    fused_patches = adapter.next_frame_patch_fusion(
        patches, backbone_output.visual_length
    )
    frame_features = fused_patches.mean(dim=1)
    pooled_features = adapter.temporal_fusion(frame_features)
    expected = adapter.projection(adapter.norm(pooled_features))

    torch.testing.assert_close(output.visual_features, expected)
    torch.testing.assert_close(output.visual_length, torch.tensor([1, 2]))
    assert output.visual_features.shape == (3, 6)

    output.visual_features.sum().backward()
    assert patches.grad is not None


def test_next_frame_fusion_preserves_each_video_final_frame() -> None:
    torch.manual_seed(1)
    patches = torch.randn(6, 4, 4)
    lengths = torch.tensor([2, 4])
    fusion = NextFramePatchFusion(
        hidden_dim=4,
        fusion_hidden_dim=5,
        grid_size=(2, 2),
    )

    fused = fusion(patches, lengths)

    torch.testing.assert_close(fused[1], patches[1])
    torch.testing.assert_close(fused[5], patches[5])


def test_projection_rank_remains_configurable() -> None:
    default = _adapter(projection_rank=None)
    narrow = _adapter(projection_rank=3)

    assert default.projection_rank == 6
    assert default.projection[0].weight.shape == (6, 4)
    assert narrow.projection_rank == 3
    assert narrow.projection[0].weight.shape == (3, 4)
    assert narrow.projection[2].weight.shape == (6, 3)
    assert narrow.trainable_parameter_count == sum(
        parameter.numel() for parameter in narrow.parameters()
    )


def test_both_motion_diagnostics_are_exposed() -> None:
    adapter = _adapter()

    # The two gates carry different defaults on purpose: the patch fusion's
    # residual is diluted by a spatial mean over every patch, the window
    # fusion's is not, so they do not start from the same place.
    assert adapter.patch_motion_weight == pytest.approx(
        torch.sigmoid(torch.tensor(1.0)).item()
    )
    assert adapter.temporal_motion_weight == pytest.approx(
        torch.sigmoid(torch.tensor(-2.0)).item()
    )
    assert adapter.mean_displacement is None

    adapter(_backbone_output(torch.randn(6, 4, 4)))
    assert adapter.mean_displacement is not None


@pytest.mark.parametrize(
    ("patches", "lengths", "error", "message"),
    [
        (torch.randn(4, 4), (2, 2), ValueError, "shape"),
        (torch.randn(4, 0, 4), (2, 2), ValueError, "one patch"),
        (torch.randn(4, 4, 7), (2, 2), ValueError, "dimension"),
        (torch.randn(4, 4, 4), (2, 1), ValueError, "sum"),
        (torch.randn(4, 4, 4), (1, 3), ValueError, "divisible"),
    ],
)
def test_adapter_validates_inputs(
    patches: torch.Tensor,
    lengths: tuple[int, ...],
    error: type[Exception],
    message: str,
) -> None:
    adapter = _adapter()

    with pytest.raises(error, match=message):
        adapter(_backbone_output(patches, lengths))
