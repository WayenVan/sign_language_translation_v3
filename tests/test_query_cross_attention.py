import pytest
import torch

from csi_slt.modeling_slt.visual_adapters.query_cross_attention import (
    LearnedQueryBank,
    QueryCrossAttention,
)


def test_shared_query_bank_supports_independent_feature_branches():
    query_bank = LearnedQueryBank(num_queries=4, hidden_size=8)
    static_attention = QueryCrossAttention(6, 8, num_heads=2, output_dim=10)
    motion_attention = QueryCrossAttention(5, 8, num_heads=2, output_dim=10)
    queries = query_bank(batch_size=3)

    static_output = static_attention(queries, torch.randn(3, 12, 6))
    motion_output = motion_attention(queries, torch.randn(3, 7, 5))

    assert static_output.query_features.shape == (3, 4, 10)
    assert motion_output.query_features.shape == (3, 4, 10)
    assert static_output.attention_weights.shape == (3, 2, 4, 12)
    assert motion_output.attention_weights.shape == (3, 2, 4, 7)
    assert static_attention.cross_attention is not motion_attention.cross_attention


def test_source_mask_excludes_invalid_tokens_from_attention():
    module = QueryCrossAttention(6, 8, num_heads=2)
    queries = torch.randn(4, 8)
    source = torch.randn(2, 5, 6)
    valid_mask = torch.tensor(
        [[True, True, False, False, False], [True, True, True, True, False]]
    )

    output = module(queries, source, source_valid_mask=valid_mask)

    invalid_weights = output.attention_weights.masked_select(
        (~valid_mask)[:, None, None, :]
    )
    assert torch.count_nonzero(invalid_weights) == 0


def test_attention_can_be_skipped():
    module = QueryCrossAttention(6, 8, num_heads=2)
    output = module(
        torch.randn(1, 4, 8),
        torch.randn(3, 5, 6),
        return_attention=False,
    )

    assert output.query_features.shape == (3, 4, 8)
    assert output.attention_weights is None


def test_rejects_fully_masked_source_row():
    module = QueryCrossAttention(6, 8, num_heads=2)

    with pytest.raises(ValueError, match="at least one valid token"):
        module(
            torch.randn(4, 8),
            torch.randn(2, 5, 6),
            source_valid_mask=torch.tensor(
                [[True, False, False, False, False], [False] * 5]
            ),
        )
