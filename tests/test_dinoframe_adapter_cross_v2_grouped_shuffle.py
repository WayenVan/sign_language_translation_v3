import pytest
import torch

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2_grouped_shuffle import (
    DINOFrameAdapterCrossV2GroupedShuffle,
    PackedTemporalWindowAdapter,
)


def test_window_indices_replicate_without_crossing_video_boundaries():
    adapter = PackedTemporalWindowAdapter(1, 1, window_size=3, stride=2)
    features = torch.tensor(
        [[0.0], [1.0], [2.0], [3.0], [10.0], [11.0]]
    )

    windows, output_lengths = adapter._make_windows(
        features, torch.tensor([4, 2])
    )

    assert output_lengths.tolist() == [2, 1]
    assert windows.squeeze(-1).tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [10.0, 10.0, 11.0],
    ]


def test_grouped_shuffle_supports_variable_lengths():
    lengths = torch.tensor([2, 4])
    total_frames = int(lengths.sum())
    backbone_output = VisualBackboneOutput(
        visual_features=torch.randn(total_frames, 5, 8),
        pooled_visual_features=torch.randn(total_frames, 10),
        visual_length=lengths,
    )
    adapter = DINOFrameAdapterCrossV2GroupedShuffle(
        input_dim=8,
        cls_input_dim=10,
        output_dim=12,
        temporal_window_size=3,
        temporal_window_stride=2,
    )

    output = adapter(backbone_output)

    assert output.visual_length.tolist() == [2, 4]
    assert output.visual_features.shape == (6, 12)
    assert output.position_ids.tolist() == [0, 0, 0, 0, 1, 1]
    assert output.extras["patch_weights"].shape == (total_frames, 5)


def test_grouped_shuffle_rejects_lengths_not_divisible_by_stride():
    adapter = PackedTemporalWindowAdapter(4, 4, window_size=3, stride=2)

    with pytest.raises(ValueError, match="divisible by stride"):
        adapter(torch.randn(7, 4), torch.tensor([3, 4]))


def test_grouped_shuffle_rejects_temporal_permutation():
    output = VisualBackboneOutput(
        visual_features=torch.randn(2, 3, 4),
        pooled_visual_features=torch.randn(2, 4),
        visual_length=torch.tensor([2]),
    )
    adapter = DINOFrameAdapterCrossV2GroupedShuffle(4, 6)

    with pytest.raises(ValueError, match="incompatible"):
        adapter(output, permute_video_tokens=True)
