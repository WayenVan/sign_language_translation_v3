import pytest
import torch

from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2 import (
    DINOFrameAdapterCrossV2,
)


def test_default_radius_disables_the_cross_frame_spatial_window():
    adapter = DINOFrameAdapterCrossV2(input_dim=2, output_dim=2, temperature=1.0)
    base = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    shifted = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    aligned = adapter.similarity_aggregate(base, shifted)
    similarity = torch.einsum("bnd,btd->bnt", base, shifted)
    expected = torch.einsum("bnt,btd->bnd", similarity.softmax(dim=-1), shifted)

    assert adapter.spatial_window_radius is None
    torch.testing.assert_close(aligned, expected)


def test_radius_zero_matches_only_the_same_patch_position():
    adapter = DINOFrameAdapterCrossV2(
        input_dim=2,
        output_dim=2,
        spatial_window_radius=0,
        spatial_grid_size=(2, 2),
    )
    base = torch.randn(1, 4, 2)
    shifted = torch.randn(1, 4, 2)

    aligned = adapter.similarity_aggregate(base, shifted)

    torch.testing.assert_close(aligned, shifted)


def test_radius_one_uses_a_two_dimensional_neighbourhood_without_row_wrap():
    adapter = DINOFrameAdapterCrossV2(
        input_dim=2,
        output_dim=2,
        spatial_window_radius=1,
        spatial_grid_size=(2, 3),
    )

    mask = adapter._spatial_neighbourhood_mask(6, torch.device("cpu"))

    assert mask[2, 1]
    assert mask[2, 4]
    assert not mask[2, 3]  # Adjacent flattened indices, but opposite row edges.


def test_non_square_patch_count_requires_an_explicit_grid_size():
    adapter = DINOFrameAdapterCrossV2(
        input_dim=2,
        output_dim=2,
        spatial_window_radius=1,
    )

    with pytest.raises(ValueError, match="cannot infer a square patch grid"):
        adapter.similarity_aggregate(
            torch.randn(1, 6, 2), torch.randn(1, 6, 2)
        )
