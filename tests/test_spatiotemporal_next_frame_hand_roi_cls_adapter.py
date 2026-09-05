import pytest
import torch

from csi_slt.configuration_slt.configuration_scorer import HandPatchScorerConfig
from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS
from csi_slt.modeling_slt.scorer import HandPatchScorer
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_hand_roi_cls_adapter import (
    SpatiotemporalNextFrameHandRoiClsAdapter,
)

INPUT_DIM = 6
CLS_INPUT_DIM = 8
PATCHES = 9
OUTPUT_DIM = 5


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


def _adapter(scorer_dir: str, **overrides):
    kwargs = dict(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        cls_input_dim=CLS_INPUT_DIM,
        scorer_path=scorer_dir,
        top_k=3,
        projection_rank=7,
        temporal_scale_factor=2,
        roi_projection_rank=4,
        cls_projection_rank=4,
        patch_grid_size=(3, 3),
        patch_fusion_hidden_dim=8,
    )
    kwargs.update(overrides)
    adapter = SpatiotemporalNextFrameHandRoiClsAdapter(**kwargs)
    adapter.roi_pool.load_pretrained_components()
    return adapter


def _backbone_output(patches: torch.Tensor, cls: torch.Tensor) -> VisualBackboneOutput:
    return VisualBackboneOutput(
        visual_features=patches,
        pooled_visual_features=cls,
        visual_length=torch.tensor([2, 4]),
    )


def test_adapter_is_registered() -> None:
    assert (
        VISUAL_ADAPTERS["spatiotemporal_next_frame_hand_roi_cls"]
        is SpatiotemporalNextFrameHandRoiClsAdapter
    )


def test_concat_fusion_mode_is_rejected(scorer_dir: str) -> None:
    with pytest.raises(ValueError, match="only supports fusion_mode='gated'"):
        _adapter(scorer_dir, fusion_mode="concat")


def test_forward_adds_gated_cls_residual_on_top_of_next_frame_hand_roi(
    scorer_dir: str,
) -> None:
    torch.manual_seed(0)
    adapter = _adapter(scorer_dir)
    patches = torch.randn(6, PATCHES, INPUT_DIM, requires_grad=True)
    cls = torch.randn(6, CLS_INPUT_DIM, requires_grad=True)
    lengths = torch.tensor([2, 4])

    output = adapter(_backbone_output(patches, cls))

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
    pooled_cls = torch.cat(
        [
            cls[:2].unflatten(0, (-1, 2)).mean(dim=1),
            cls[2:].unflatten(0, (-1, 2)).mean(dim=1),
        ]
    )
    pooled_global, pooled_roi = pooled.split(INPUT_DIM, dim=-1)
    expected = (
        adapter.projection(adapter.norm(pooled_global))
        + torch.sigmoid(adapter.fusion_gate)
        * adapter.roi_projection(adapter.roi_norm(pooled_roi))
        + torch.sigmoid(adapter.cls_gate)
        * adapter.cls_projection(adapter.cls_norm(pooled_cls))
    )

    torch.testing.assert_close(output.visual_features, expected)
    torch.testing.assert_close(output.visual_length, torch.tensor([1, 2]))
    assert output.visual_features.shape == (3, OUTPUT_DIM)
    assert set(output.logging_scalars) == {
        "mean_displacement",
        "motion_gate",
        "roi_global_distance",
        "selection_margin",
        "roi_gate",
        "cls_gate",
    }

    output.visual_features.sum().backward()
    assert patches.grad is not None
    assert cls.grad is not None


def test_zero_cls_gate_reduces_to_next_frame_hand_roi(scorer_dir: str) -> None:
    torch.manual_seed(0)
    adapter = _adapter(scorer_dir, cls_gate_init=-30.0)
    patches = torch.randn(6, PATCHES, INPUT_DIM)
    cls = torch.randn(6, CLS_INPUT_DIM)
    lengths = torch.tensor([2, 4])

    output = adapter(_backbone_output(patches, cls))

    fused = adapter.next_frame_patch_fusion(patches, lengths)
    mask = adapter.roi_pool.select(patches)
    frame_features = torch.cat(
        [
            fused.mean(dim=1),
            torch.stack([fused[f][mask[f]].mean(dim=0) for f in range(6)]),
        ],
        dim=-1,
    )
    pooled = torch.cat(
        [
            frame_features[:2].unflatten(0, (-1, 2)).mean(dim=1),
            frame_features[2:].unflatten(0, (-1, 2)).mean(dim=1),
        ]
    )
    pooled_global, pooled_roi = pooled.split(INPUT_DIM, dim=-1)
    baseline = adapter.projection(adapter.norm(pooled_global)) + torch.sigmoid(
        adapter.fusion_gate
    ) * adapter.roi_projection(adapter.roi_norm(pooled_roi))

    torch.testing.assert_close(output.visual_features, baseline, atol=1e-6, rtol=0)


def test_gate_group_exposes_all_three_scalar_gates(scorer_dir: str) -> None:
    adapter = _adapter(scorer_dir)
    groups = adapter.optimization_parameter_groups()

    assert set(groups) == {"gates"}
    gates = groups["gates"]
    assert len(gates) == 3
    gate_ids = {id(p) for p in gates}
    assert gate_ids == {
        id(adapter.next_frame_patch_fusion.fusion_gate),
        id(adapter.fusion_gate),
        id(adapter.cls_gate),
    }
    for gate in gates:
        assert gate.requires_grad


def test_missing_cls_features_is_rejected(scorer_dir: str) -> None:
    adapter = _adapter(scorer_dir)
    patches = torch.randn(6, PATCHES, INPUT_DIM)
    with pytest.raises(ValueError, match="pooled_visual_features must carry"):
        adapter(
            VisualBackboneOutput(
                visual_features=patches,
                visual_length=torch.tensor([2, 4]),
            )
        )
