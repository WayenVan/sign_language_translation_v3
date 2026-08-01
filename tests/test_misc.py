import pytest
import torch

from csi_slt.modeling_slt.misc import random_derangement


def test_random_derangement_has_no_fixed_points_within_each_video():
    lengths = [2, 3, 8]

    for _ in range(20):
        permutation = random_derangement(lengths)
        offset = 0
        for length in lengths:
            segment = permutation[offset : offset + length]
            identity = torch.arange(offset, offset + length)
            assert torch.equal(segment.sort().values, identity)
            assert torch.all(segment != identity)
            offset += length


def test_random_derangement_supports_empty_input_and_empty_segments():
    assert random_derangement([]).tolist() == []
    assert random_derangement([0, 2, 0]).tolist() == [1, 0]


@pytest.mark.parametrize("lengths", ([1], [2, 1]))
def test_random_derangement_rejects_single_frame_videos(lengths):
    with pytest.raises(ValueError, match="one frame"):
        random_derangement(lengths)


def test_random_derangement_rejects_negative_lengths():
    with pytest.raises(ValueError, match="non-negative"):
        random_derangement([2, -1])
