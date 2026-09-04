import pytest
import torch

from csi_slt.configuration_slt.configuration_scorer import HandPatchScorerConfig
from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS
from csi_slt.modeling_slt.scorer import HandPatchScorer
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_hand_roi_conv_adapter import (
    SpatiotemporalNextFrameHandRoiConvAdapter,
)

INPUT_DIM = 6
PATCHES = 9


@pytest.fixture
def scorer_dir(tmp_path):
    scorer = HandPatchScorer(HandPatchScorerConfig(input_dim=INPUT_DIM))
    with torch.no_grad():
        scorer.linear.weight.copy_(torch.arange(INPUT_DIM).float()[None])
        scorer.linear.bias.zero_()
    scorer.set_feature_statistics(torch.zeros(INPUT_DIM), torch.ones(INPUT_DIM))
    path = tmp_path / "scorer"
    scorer.save_pretrained(path)
    return str(path)


def _adapter(scorer_dir: str, fusion_mode: str, **kwargs) -> SpatiotemporalNextFrameHandRoiConvAdapter:
    adapter = SpatiotemporalNextFrameHandRoiConvAdapter(
        input_dim=INPUT_DIM,
        output_dim=5,
        scorer_path=scorer_dir,
        top_k=3,
        projection_rank=7,
        temporal_scale_factor=2,
        fusion_mode=fusion_mode,
        roi_projection_rank=4,
        patch_grid_size=(3, 3),
        patch_fusion_hidden_dim=8,
        **kwargs,
    )
    adapter.roi_pool.load_pretrained_components()
    return adapter


def _backbone_output(patches: torch.Tensor) -> VisualBackboneOutput:
    return VisualBackboneOutput(
        visual_features=patches,
        visual_length=torch.tensor([2, 4]),
    )


def test_conv_adapter_is_registered() -> None:
    assert (
        VISUAL_ADAPTERS["spatiotemporal_next_frame_hand_roi_conv"]
        is SpatiotemporalNextFrameHandRoiConvAdapter
    )


@pytest.mark.parametrize("fusion_mode", ["concat", "gated"])
def test_forward_matches_mean_at_default_radius(
    scorer_dir: str, fusion_mode: str
) -> None:
    """At the default radius=0, the conv is initialized to reproduce the exact
    mean it replaces, so this adapter's first forward pass must be numerically
    identical to SpatiotemporalNextFrameHandRoiAdapter's."""
    torch.manual_seed(0)
    adapter = _adapter(scorer_dir, fusion_mode)
    patches = torch.randn(6, PATCHES, INPUT_DIM, requires_grad=True)
    lengths = torch.tensor([2, 4])

    output = adapter(_backbone_output(patches))

    fused = adapter.next_frame_patch_fusion(patches, lengths)
    mask = adapter.roi_pool.select(patches)
    global_features = fused.mean(dim=1)
    roi_features = torch.stack(
        [fused[frame][mask[frame]].mean(dim=0) for frame in range(6)]
    )
    frame_features = torch.cat([global_features, roi_features], dim=-1)
    pooled = torch.cat(
        [
            frame_features[:2].unflatten(0, (-1, 2)).mean(dim=1),
            frame_features[2:].unflatten(0, (-1, 2)).mean(dim=1),
        ]
    )
    if fusion_mode == "concat":
        expected = adapter.projection(adapter.norm(pooled))
    else:
        pooled_global, pooled_roi = pooled.split(INPUT_DIM, dim=-1)
        expected = adapter.projection(adapter.norm(pooled_global)) + torch.sigmoid(
            adapter.fusion_gate
        ) * adapter.roi_projection(adapter.roi_norm(pooled_roi))

    torch.testing.assert_close(output.visual_features, expected)
    torch.testing.assert_close(output.visual_length, torch.tensor([1, 2]))
    assert output.visual_features.shape == (3, 5)

    output.visual_features.sum().backward()
    assert patches.grad is not None


def test_conv_kernel_widens_with_radius(scorer_dir: str) -> None:
    adapter = _adapter(scorer_dir, "concat", temporal_conv_radius=1)
    conv = adapter.temporal_conv_downsample.conv
    assert conv.kernel_size == (4,)  # scale_factor(2) + 2 * radius(1)
    assert conv.weight.requires_grad

    patches = torch.randn(6, PATCHES, INPUT_DIM)
    output = adapter(_backbone_output(patches))
    assert output.visual_features.shape == (3, 5)
    assert output.visual_length.tolist() == [1, 2]


def test_wide_radius_never_leaks_across_videos(scorer_dir: str) -> None:
    """A wide context radius must still respect packed-video boundaries:
    changing one video's frames must not change another video's tokens."""
    adapter = _adapter(scorer_dir, "concat", temporal_conv_radius=2)
    patches = torch.randn(6, PATCHES, INPUT_DIM)
    baseline = adapter(_backbone_output(patches)).visual_features

    perturbed = patches.clone()
    perturbed[:2] += 100.0  # only the first (2-frame) video changes
    output = adapter(_backbone_output(perturbed)).visual_features

    # Token 0 belongs to the first video and is expected to change; tokens
    # 1-2 belong to the second video and must be untouched.
    torch.testing.assert_close(output[1:], baseline[1:])


def test_disabled_transformer_is_not_built(scorer_dir: str) -> None:
    adapter = _adapter(scorer_dir, "gated")
    assert adapter.transformer_context is None


def test_disabled_conv_is_not_built(scorer_dir: str) -> None:
    adapter = _adapter(scorer_dir, "concat", use_temporal_conv=False)
    assert adapter.temporal_conv_downsample is None


def test_disabled_conv_rejects_a_nonzero_radius(scorer_dir: str) -> None:
    with pytest.raises(ValueError):
        _adapter(
            scorer_dir,
            "concat",
            use_temporal_conv=False,
            temporal_conv_radius=1,
        )


@pytest.mark.parametrize("fusion_mode", ["concat", "gated"])
def test_disabled_conv_matches_the_mean_pooling_parent(
    scorer_dir: str, fusion_mode: str
) -> None:
    """use_temporal_conv=False must match SpatiotemporalNextFrameHandRoiAdapter
    structurally, not only numerically at step 0: no conv module is built, and
    training the rest of the model cannot move this path away from the mean."""
    torch.manual_seed(0)
    adapter = _adapter(scorer_dir, fusion_mode, use_temporal_conv=False)
    patches = torch.randn(6, PATCHES, INPUT_DIM, requires_grad=True)
    lengths = torch.tensor([2, 4])

    output = adapter(_backbone_output(patches))

    fused = adapter.next_frame_patch_fusion(patches, lengths)
    mask = adapter.roi_pool.select(patches)
    global_features = fused.mean(dim=1)
    roi_features = torch.stack(
        [fused[frame][mask[frame]].mean(dim=0) for frame in range(6)]
    )
    frame_features = torch.cat([global_features, roi_features], dim=-1)
    pooled = torch.cat(
        [
            frame_features[:2].unflatten(0, (-1, 2)).mean(dim=1),
            frame_features[2:].unflatten(0, (-1, 2)).mean(dim=1),
        ]
    )
    if fusion_mode == "concat":
        expected = adapter.projection(adapter.norm(pooled))
    else:
        pooled_global, pooled_roi = pooled.split(INPUT_DIM, dim=-1)
        expected = adapter.projection(adapter.norm(pooled_global)) + torch.sigmoid(
            adapter.fusion_gate
        ) * adapter.roi_projection(adapter.roi_norm(pooled_roi))

    torch.testing.assert_close(output.visual_features, expected)


def test_transformer_context_starts_as_identity(scorer_dir: str) -> None:
    """num_transformer_layers>0 must not change the forward pass at init:
    both output projections inside the block are zero-initialized."""
    torch.manual_seed(0)
    without_transformer = _adapter(scorer_dir, "concat")
    torch.manual_seed(0)
    with_transformer = _adapter(
        scorer_dir,
        "concat",
        num_transformer_layers=2,
        transformer_num_heads=3,
    )
    assert with_transformer.transformer_context is not None
    assert without_transformer.transformer_context is None

    patches = torch.randn(6, PATCHES, INPUT_DIM)
    backbone_output = _backbone_output(patches)

    baseline = without_transformer(backbone_output).visual_features
    output = with_transformer(backbone_output).visual_features
    torch.testing.assert_close(output, baseline)
