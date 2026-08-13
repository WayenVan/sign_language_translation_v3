import pytest
import torch

from csi_slt.modeling_slt.minimal_null_ot_alignment import (
    MinimalNullOTAlignment,
    group_alignment_by_position_ids,
)
from csi_slt.modeling_slt.misc import packed_position_ids_to_padded


def test_grouped_adapter_style_packed_position_ids_are_padded_per_video():
    packed_position_ids = torch.tensor(
        [0, 0, 1, 1, 2, 2, 0, 0, 1, 1]
    )

    padded, mask = packed_position_ids_to_padded(
        packed_position_ids,
        torch.tensor([6, 4]),
    )

    assert padded.tolist() == [
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, -100, -100],
    ]
    assert mask.tolist() == [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0],
    ]


def test_group_alignment_averages_rows_with_shared_positions():
    alignment = torch.tensor(
        [
            [
                [0.0, 0.8, 0.2],
                [0.0, 0.6, 0.4],
                [0.0, 0.2, 0.8],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.5],
                [0.0, 0.5, 0.5],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.4, 0.6],
                [0.0, 0.2, 0.8],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )
    video_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
    )
    position_ids = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, -100, -100]]
    )

    grouped, grouped_mask = group_alignment_by_position_ids(
        alignment, video_mask, position_ids
    )

    expected = torch.tensor(
        [
            [[0.0, 0.7, 0.3], [0.0, 0.1, 0.9], [0.0, 0.5, 0.5]],
            [[0.0, 0.5, 0.5], [0.0, 0.3, 0.7], [0.0, 0.0, 0.0]],
        ]
    )
    torch.testing.assert_close(grouped, expected)
    assert grouped_mask.tolist() == [[True, True, True], [True, True, False]]


@pytest.mark.parametrize(
    ("position_ids", "message"),
    [
        ([1, 1, 2, 2], "start at 0"),
        ([0, 0, 2, 2], "consecutive"),
        ([0, 1, 0, 1], "consecutive"),
    ],
)
def test_group_alignment_rejects_invalid_temporal_ids(position_ids, message):
    alignment = torch.ones(1, 4, 3)
    video_mask = torch.ones(1, 4, dtype=torch.long)

    with pytest.raises(ValueError, match=message):
        group_alignment_by_position_ids(
            alignment,
            video_mask,
            torch.tensor([position_ids]),
        )


def test_group_alignment_requires_minus_100_for_padding():
    alignment = torch.ones(1, 3, 2)
    video_mask = torch.tensor([[1, 1, 0]])

    with pytest.raises(ValueError, match="must be -100"):
        group_alignment_by_position_ids(
            alignment,
            video_mask,
            torch.tensor([[0, 1, -1]]),
        )


def test_grouped_tv_operates_between_positions_not_token_types():
    module = MinimalNullOTAlignment(video_dim=2, text_dim=2)
    # Within each position the two token types disagree completely, while the
    # position-level average is identical across time. Grouped TV must be zero.
    alignment = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    video_mask = torch.ones(1, 4, dtype=torch.long)
    position_ids = torch.tensor([[0, 0, 1, 1]])

    grouped, grouped_mask = group_alignment_by_position_ids(
        alignment, video_mask, position_ids
    )
    tv_loss = module._compute_tv_loss(grouped, grouped_mask)

    torch.testing.assert_close(tv_loss, torch.tensor(0.0))


def test_alignment_info_exposes_the_position_grouping_used_by_tv():
    module = MinimalNullOTAlignment(
        video_dim=2,
        text_dim=2,
        eps=0.1,
        n_iters=2,
    )
    video_features = torch.tensor(
        [[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]]
    )
    pseudo_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    video_mask = torch.ones(1, 4, dtype=torch.long)
    pseudo_mask = torch.ones(1, 2, dtype=torch.long)

    _, info = module(
        video_features=video_features,
        pseudo_embeddings=pseudo_embeddings,
        video_mask=video_mask,
        pseudo_mask=pseudo_mask,
        visual_position_ids=torch.tensor([[0, 0, 1, 1]]),
    )

    expected_grouped, expected_mask = group_alignment_by_position_ids(
        info["alignment"],
        video_mask,
        torch.tensor([[0, 0, 1, 1]]),
    )
    torch.testing.assert_close(info["grouped_alignment"], expected_grouped)
    assert torch.equal(info["grouped_video_mask"], expected_mask)
    assert info["grouped_alignment"].shape == (1, 2, 3)
    assert info["grouped_alignment"].requires_grad is False
