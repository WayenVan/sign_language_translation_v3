import pytest
import torch

from csi_slt.configuration_slt.configuration_scorer import HandPatchScorerConfig
from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS
from csi_slt.modeling_slt.scorer import HandPatchScorer
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_hand_roi_adapter import (
    SpatiotemporalNextFrameHandRoiAdapter,
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


def _adapter(scorer_dir: str, fusion_mode: str):
    adapter = SpatiotemporalNextFrameHandRoiAdapter(
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
    )
    adapter.roi_pool.load_pretrained_components()
    return adapter


def _backbone_output(patches: torch.Tensor) -> VisualBackboneOutput:
    return VisualBackboneOutput(
        visual_features=patches,
        visual_length=torch.tensor([2, 4]),
    )


def test_next_frame_hand_roi_adapter_is_registered() -> None:
    assert (
        VISUAL_ADAPTERS["spatiotemporal_next_frame_hand_roi"]
        is SpatiotemporalNextFrameHandRoiAdapter
    )


@pytest.mark.parametrize("fusion_mode", ["concat", "gated"])
def test_forward_matches_fused_global_and_raw_selected_roi(
    scorer_dir: str,
    fusion_mode: str,
) -> None:
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
    assert set(output.logging_scalars) == {
        "mean_displacement",
        "motion_gate",
        "roi_global_distance",
        "selection_margin",
        *({"roi_gate"} if fusion_mode == "gated" else set()),
    }

    output.visual_features.sum().backward()
    assert patches.grad is not None


def test_roi_selection_reads_raw_features(scorer_dir: str, monkeypatch) -> None:
    adapter = _adapter(scorer_dir, "concat")
    patches = torch.randn(6, PATCHES, INPUT_DIM)
    selected_inputs = []
    original_select = adapter.roi_pool.select

    def capture_select(score_features):
        selected_inputs.append(score_features)
        return original_select(score_features)

    monkeypatch.setattr(adapter.roi_pool, "select", capture_select)
    adapter(_backbone_output(patches))

    assert selected_inputs[0] is patches


def test_frozen_scorer_is_excluded_from_trainable_count(scorer_dir: str) -> None:
    adapter = _adapter(scorer_dir, "gated")

    assert adapter.roi_pool.scorer.frozen is True
    assert adapter.trainable_parameter_count == sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )
    assert all(
        not parameter.requires_grad
        for parameter in adapter.roi_pool.scorer.parameters()
    )
