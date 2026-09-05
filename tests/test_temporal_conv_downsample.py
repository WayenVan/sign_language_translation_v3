"""Contract tests for the reusable temporal downsampling filter.

``SpatiotemporalNextFrameHandRoiConvAdapter`` is the first consumer, but the
module is meant to replace the boundary-safe temporal mean anywhere it appears,
so the properties its docstring promises are tested here directly rather than
only through that one adapter.
"""

import pytest
import torch

from csi_slt.modeling_slt.visual_adapters.temporal_conv_downsample import (
    TemporalConvDownsample,
)


def _mean_baseline(features: torch.Tensor, lengths: torch.Tensor, scale: int):
    """The exact expression this module generalizes."""
    return torch.cat(
        [
            video.unflatten(0, (-1, scale)).mean(dim=1)
            for video in torch.split(features, lengths.tolist(), dim=0)
        ],
        dim=0,
    )


def test_radius_zero_reproduces_the_temporal_mean_exactly():
    """The whole point of the 1/kernel_size init: swapping the mean for this
    module must not perturb a run before the first optimizer step."""
    module = TemporalConvDownsample(hidden_dim=6, scale_factor=2, radius=0)
    features = torch.randn(10, 6)
    lengths = torch.tensor([4, 6])

    torch.testing.assert_close(
        module(features, lengths), _mean_baseline(features, lengths, 2)
    )


@pytest.mark.parametrize("scale_factor,radius", [(2, 0), (2, 1), (3, 2), (1, 1)])
def test_output_length_is_exactly_input_length_over_scale_factor(
    scale_factor, radius
):
    """Downstream token counts (video_token_scale, the CTC head) depend on this
    holding for every radius -- which is what parameterizing by radius buys."""
    module = TemporalConvDownsample(
        hidden_dim=4, scale_factor=scale_factor, radius=radius
    )
    lengths = torch.tensor([scale_factor * 2, scale_factor * 3])
    features = torch.randn(int(lengths.sum()), 4)

    output = module(features, lengths)

    assert module.conv.kernel_size == (scale_factor + 2 * radius,)
    assert output.shape == (int(lengths.sum()) // scale_factor, 4)


def test_padding_replicates_the_edge_frame_instead_of_inserting_zeros():
    """A constant video must pool to that same constant everywhere, boundary
    windows included. Zero padding would pull the first and last windows toward
    the origin and leave a padding-shaped artifact for training to undo."""
    module = TemporalConvDownsample(hidden_dim=3, scale_factor=2, radius=1)
    features = torch.full((6, 3), 7.0)

    output = module(features, torch.tensor([6]))

    torch.testing.assert_close(output, torch.full((3, 3), 7.0))


def test_a_window_never_reads_across_a_video_boundary():
    """Padding is applied per video, so a neighbour packed into the same batch
    cannot reach a boundary window even at a radius wide enough to span it."""
    module = TemporalConvDownsample(hidden_dim=3, scale_factor=2, radius=2)
    lengths = torch.tensor([4, 4])
    features = torch.randn(8, 3)

    baseline = module(features, lengths)
    perturbed = features.clone()
    perturbed[4:] += 100.0

    torch.testing.assert_close(module(perturbed, lengths)[:2], baseline[:2])


def test_the_filter_is_depthwise_and_trainable():
    module = TemporalConvDownsample(hidden_dim=5, scale_factor=2, radius=1)

    assert module.conv.groups == 5
    assert module.conv.weight.shape == (5, 1, 4)
    torch.testing.assert_close(module.conv.weight, torch.full((5, 1, 4), 0.25))
    torch.testing.assert_close(module.conv.bias, torch.zeros(5))

    module(torch.randn(4, 5), torch.tensor([4])).sum().backward()
    assert module.conv.weight.grad is not None
    assert module.conv.bias.grad is not None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"hidden_dim": 0, "scale_factor": 2}, "hidden_dim"),
        ({"hidden_dim": 4, "scale_factor": 0}, "scale_factor"),
        ({"hidden_dim": 4, "scale_factor": 2, "radius": -1}, "radius"),
        ({"hidden_dim": 4, "scale_factor": 2, "radius": True}, "radius"),
    ],
)
def test_construction_rejects_invalid_dimensions(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TemporalConvDownsample(**kwargs)


@pytest.mark.parametrize(
    "features,lengths,error,match",
    [
        (torch.randn(4), torch.tensor([4]), ValueError, r"\[sum\(T\), C\]"),
        (torch.randn(4, 3), torch.tensor([4]), ValueError, "feature dimension"),
        (torch.randn(4, 4), torch.tensor([]), ValueError, "non-empty"),
        (torch.randn(4, 4), torch.tensor([4.0]), TypeError, "integer dtype"),
        (torch.randn(4, 4), torch.tensor([-4]), ValueError, "must be positive"),
        (torch.randn(4, 4), torch.tensor([2]), ValueError, "must equal"),
        (torch.randn(3, 4), torch.tensor([3]), ValueError, "divisible"),
    ],
)
def test_forward_validates_its_packed_inputs(features, lengths, error, match):
    module = TemporalConvDownsample(hidden_dim=4, scale_factor=2)
    with pytest.raises(error, match=match):
        module(features, lengths)
