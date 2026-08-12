import pytest
import torch

from csi_slt.modeling_slt.misc import packed_to_padded, padded_to_packed


def test_feature_conversion_round_trip() -> None:
    packed = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    lengths = torch.tensor([2, 1, 3])

    padded, mask = packed_to_padded(packed, lengths)
    restored, restored_lengths = padded_to_packed(padded, mask)

    assert padded.shape == (3, 3, 3)
    assert mask.tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
    assert torch.equal(padded[~mask.bool()], torch.zeros(3, 3))
    assert torch.equal(restored, packed)
    assert torch.equal(restored_lengths, lengths)


def test_packed_to_padded_supports_trailing_dimensions_and_empty_sequences() -> None:
    packed = torch.arange(16).reshape(2, 2, 2, 2)

    padded, mask = packed_to_padded(packed, [0, 2, 0], padding_value=-1)

    assert padded.shape == (3, 2, 2, 2, 2)
    assert mask.tolist() == [[0, 0], [1, 1], [0, 0]]
    assert torch.equal(padded[1], packed)
    assert torch.all(padded[0] == -1)


def test_padded_to_packed_accepts_boolean_mask() -> None:
    padded = torch.tensor([[[1.0], [99.0]], [[2.0], [3.0]]])
    mask = torch.tensor([[True, False], [True, True]])

    packed, lengths = padded_to_packed(padded, mask)

    assert packed.squeeze(-1).tolist() == [1.0, 2.0, 3.0]
    assert lengths.tolist() == [1, 2]


def test_packed_to_padded_propagates_gradients() -> None:
    packed = torch.randn(6, 4, requires_grad=True)

    padded, _ = packed_to_padded(packed, [2, 1, 3])
    padded.square().sum().backward()

    assert packed.grad is not None
    assert torch.allclose(packed.grad, 2 * packed.detach())


def test_padded_to_packed_propagates_only_valid_position_gradients() -> None:
    padded = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    packed, _ = padded_to_packed(padded, mask)
    packed.sum().backward()

    assert padded.grad is not None
    expected_grad = mask.unsqueeze(-1).expand_as(padded)
    assert torch.equal(padded.grad, expected_grad)


def test_feature_conversion_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match=r"lengths.sum\(\)"):
        packed_to_padded(torch.zeros(3, 2), [1, 1])
    with pytest.raises(ValueError, match="mask shape"):
        padded_to_packed(torch.zeros(2, 3, 4), torch.ones(2, 2))
    with pytest.raises(ValueError, match="only 0"):
        padded_to_packed(torch.zeros(1, 2, 4), torch.tensor([[1, 2]]))


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                f"{__file__}::test_feature_conversion_round_trip",
            ]
        )
    )
