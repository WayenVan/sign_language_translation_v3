import pytest
import torch

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_pooled_linear_adapter import (
    SpatiotemporalPooledLinearAdapter,
)


def _backbone_output(patches: torch.Tensor) -> VisualBackboneOutput:
    return VisualBackboneOutput(
        visual_features=patches,
        # Deliberately use an incompatible CLS width: the adapter must ignore it.
        pooled_visual_features=torch.randn(patches.shape[0], 11),
        visual_length=torch.tensor([2, patches.shape[0] - 2]),
    )


def test_spatiotemporal_pooled_linear_is_registered() -> None:
    assert (
        VISUAL_ADAPTERS["spatiotemporal_pooled_linear"]
        is SpatiotemporalPooledLinearAdapter
    )


@pytest.mark.parametrize("projection_rank", [None, 3, 12])
def test_spatiotemporal_pooling_matches_explicit_means_and_projection(
    projection_rank: int | None,
) -> None:
    torch.manual_seed(0)
    patches = torch.randn(6, 5, 4, requires_grad=True)
    backbone_output = _backbone_output(patches)
    adapter = SpatiotemporalPooledLinearAdapter(
        input_dim=4,
        output_dim=6,
        projection_rank=projection_rank,
    )

    output = adapter(backbone_output)
    spatial_mean = patches.mean(dim=1)
    temporal_mean = torch.cat(
        (
            spatial_mean[:2].reshape(1, 2, 4).mean(dim=1),
            spatial_mean[2:].reshape(2, 2, 4).mean(dim=1),
        )
    )
    expected = adapter.projection(adapter.norm(temporal_mean))

    torch.testing.assert_close(output.visual_features, expected)
    torch.testing.assert_close(output.visual_length, torch.tensor([1, 2]))
    assert output.visual_features.shape == (3, 6)

    output.visual_features.sum().backward()
    assert patches.grad is not None


def test_projection_rank_controls_exact_active_parameter_count() -> None:
    input_dim = 4
    output_dim = 6
    rank = 3
    dense = SpatiotemporalPooledLinearAdapter(input_dim, output_dim)
    factorized = SpatiotemporalPooledLinearAdapter(
        input_dim, output_dim, projection_rank=rank
    )

    expected_dense = 2 * input_dim + input_dim * output_dim + output_dim
    expected_factorized = 2 * input_dim + rank * (input_dim + output_dim) + output_dim

    assert dense.trainable_parameter_count == expected_dense
    assert factorized.trainable_parameter_count == expected_factorized
    assert all(parameter.requires_grad for parameter in factorized.parameters())


def test_layer_norm_can_be_excluded_from_parameter_budget() -> None:
    adapter = SpatiotemporalPooledLinearAdapter(
        input_dim=4,
        output_dim=6,
        projection_rank=3,
        use_layer_norm=False,
    )

    assert adapter.trainable_parameter_count == 3 * (4 + 6) + 6


@pytest.mark.parametrize(
    ("patches", "lengths", "error", "message"),
    [
        (torch.randn(4, 4), torch.tensor([2, 2]), ValueError, "shape"),
        (torch.randn(4, 0, 4), torch.tensor([2, 2]), ValueError, "one patch"),
        (torch.randn(4, 5, 7), torch.tensor([2, 2]), ValueError, "dimension"),
        (torch.randn(4, 5, 4), torch.tensor([2, 1]), ValueError, "sum"),
        (torch.randn(4, 5, 4), torch.tensor([2.0, 2.0]), TypeError, "integer"),
        (torch.randn(4, 5, 4), torch.tensor([1, 3]), ValueError, "divisible"),
    ],
)
def test_spatiotemporal_pooled_linear_validates_inputs(
    patches: torch.Tensor,
    lengths: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    adapter = SpatiotemporalPooledLinearAdapter(input_dim=4, output_dim=6)
    backbone_output = VisualBackboneOutput(
        visual_features=patches,
        visual_length=lengths,
    )

    with pytest.raises(error, match=message):
        adapter(backbone_output)
