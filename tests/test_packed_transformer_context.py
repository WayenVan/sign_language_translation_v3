"""Contract tests for the reusable packed self-attention context block.

Its two load-bearing claims -- exact identity at construction while every
parameter still receives gradient, and attention that cannot cross a video
boundary by construction -- are the reasons it can be dropped into an adapter
without perturbing a run, so they are tested here rather than only through the
one adapter that currently uses it.
"""

import pytest
import torch

from csi_slt.modeling_slt.visual_adapters.packed_transformer_context import (
    PackedTransformerContext,
)


def test_the_stack_is_an_exact_identity_at_construction():
    module = PackedTransformerContext(hidden_dim=8, num_layers=3, num_heads=2)
    features = torch.randn(9, 8)

    output = module(features, torch.tensor([4, 5]))

    torch.testing.assert_close(output, features)


def test_the_identity_init_is_not_a_dead_end():
    """A zero weight is not a zero gradient: a linear layer's weight gradient
    depends on its input, so the zero-initialized output projections move on
    the very first step and the block escapes the identity.

    Their gradient is also the *only* nonzero one at step 0 -- everything
    upstream of them inside the block backpropagates through that zero weight.
    That is the ordinary cost of an exact identity at init, and it is asserted
    here so the property is documented rather than discovered."""
    module = PackedTransformerContext(hidden_dim=8, num_layers=2, num_heads=2)

    module(torch.randn(6, 8), torch.tensor([2, 4])).sum().backward()

    trained_at_step_zero = {
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and parameter.grad.any()
    }
    assert trained_at_step_zero == {
        f"blocks.{index}.{suffix}"
        for index in (0, 1)
        for suffix in (
            "attn.out_proj.weight",
            "attn.out_proj.bias",
            "mlp.2.weight",
            "mlp.2.bias",
        )
    }


def test_one_step_off_the_identity_unblocks_the_rest_of_the_block():
    """Follow-on to the above: once the output projections are nonzero, every
    remaining parameter receives gradient, so nothing is permanently frozen."""
    module = PackedTransformerContext(hidden_dim=8, num_layers=1, num_heads=2)
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)

    module(torch.randn(6, 8), torch.tensor([2, 4])).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    module(torch.randn(6, 8), torch.tensor([2, 4])).sum().backward()

    ungradiented = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is None or not parameter.grad.any()
    ]
    assert ungradiented == []


def test_attention_cannot_cross_a_video_boundary():
    """The key_padding_mask derived from packed lengths is what enforces this;
    a plain [B, T, C] reshape without it would let a short video attend into
    whatever padded the batch."""
    torch.manual_seed(0)
    module = PackedTransformerContext(hidden_dim=8, num_layers=2, num_heads=2)
    # Move the block off its identity init, otherwise this passes trivially.
    for parameter in module.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    lengths = torch.tensor([3, 5])
    features = torch.randn(8, 8)
    baseline = module(features, lengths)

    perturbed = features.clone()
    perturbed[3:] += 100.0

    torch.testing.assert_close(module(perturbed, lengths)[:3], baseline[:3])


def test_padding_length_does_not_change_a_shorter_videos_output():
    """Same property from the other side: a video's tokens must not depend on
    how long the longest video it was batched with happens to be."""
    torch.manual_seed(0)
    module = PackedTransformerContext(hidden_dim=8, num_layers=1, num_heads=2)
    for parameter in module.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    first = torch.randn(3, 8)
    alone = module(first, torch.tensor([3]))
    batched = module(torch.cat([first, torch.randn(6, 8)]), torch.tensor([3, 6]))

    torch.testing.assert_close(batched[:3], alone)


@pytest.mark.parametrize("hidden_dim", [8, 9])
def test_odd_hidden_dim_fills_the_position_table_without_a_shape_error(hidden_dim):
    """An odd width has one fewer cosine slot than sine slot; the encoding
    trims the frequency vector rather than assuming the width is even."""
    module = PackedTransformerContext(
        hidden_dim=hidden_dim, num_layers=1, num_heads=1
    )
    encoding = module._sinusoidal_position_encoding(5, torch.device("cpu"), torch.float32)

    assert encoding.shape == (5, hidden_dim)
    assert torch.isfinite(encoding).all()


def test_the_position_cache_grows_with_the_longest_sequence_seen():
    module = PackedTransformerContext(hidden_dim=8, num_layers=1, num_heads=2)

    module(torch.randn(3, 8), torch.tensor([3]))
    short_cache = module._position_encoding.shape[0]
    module(torch.randn(9, 8), torch.tensor([9]))

    assert short_cache == 3
    assert module._position_encoding.shape[0] == 9
    # Non-persistent: a cache is a runtime detail, not checkpoint state.
    assert "_position_encoding" not in module.state_dict()


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"hidden_dim": 8, "num_layers": 0}, ValueError, "num_layers"),
        ({"hidden_dim": 8, "num_layers": 1, "num_heads": 3}, ValueError, "divisible"),
        ({"hidden_dim": 8, "num_layers": 1, "mlp_ratio": 0}, ValueError, "mlp_ratio"),
    ],
)
def test_construction_rejects_invalid_configurations(kwargs, error, match):
    with pytest.raises(error, match=match):
        PackedTransformerContext(**kwargs)


@pytest.mark.parametrize(
    "features,lengths,error,match",
    [
        (torch.randn(4), torch.tensor([4]), ValueError, r"\[sum\(T\), C\]"),
        (torch.randn(4, 3), torch.tensor([4]), ValueError, "feature dimension"),
        (torch.randn(4, 8), torch.tensor([4.0]), TypeError, "integer dtype"),
        (torch.randn(4, 8), torch.tensor([0, 4]), ValueError, "must be positive"),
        (torch.randn(4, 8), torch.tensor([3]), ValueError, "must equal"),
    ],
)
def test_forward_validates_its_packed_inputs(features, lengths, error, match):
    module = PackedTransformerContext(hidden_dim=8, num_layers=1, num_heads=2)
    with pytest.raises(error, match=match):
        module(features, lengths)
