"""Reusable learned-query cross-attention building blocks.

The query bank is deliberately separate from the attention branch.  A caller
can therefore reuse one semantic set of queries for independent static and
motion branches while keeping each branch's normalization, projections, and
attention parameters separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class QueryCrossAttentionOutput:
    """Outputs from :class:`QueryCrossAttention`.

    Attributes:
        query_features: Updated query tokens with shape ``[B, K, D_out]``.
        attention_weights: Per-head attention probabilities with shape
            ``[B, num_heads, K, N]``. It is ``None`` when
            ``return_attention=False``.
    """

    query_features: torch.Tensor
    attention_weights: torch.Tensor | None = None


class LearnedQueryBank(nn.Module):
    """Own and expand a reusable set of learned query tokens."""

    def __init__(
        self,
        num_queries: int,
        hidden_size: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.queries = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        nn.init.normal_(self.queries, std=init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self.queries.expand(batch_size, -1, -1)


class QueryCrossAttention(nn.Module):
    """Update externally supplied queries from one source feature stream.

    The source stream is normalized and projected into ``hidden_size`` before
    cross-attention. Queries must already use ``hidden_size``. Keeping queries
    external lets static and motion instances of this module share a single
    :class:`LearnedQueryBank` without sharing their attention parameters.

    Args:
        source_dim: Feature width of the source tokens.
        hidden_size: Internal query and attention width.
        num_heads: Number of cross-attention heads.
        output_dim: Output query width. Defaults to ``hidden_size``.
        ffn_hidden_size: Residual FFN width. Defaults to ``4 * hidden_size``.
        dropout: Dropout used by attention and the residual FFN.

    Inputs:
        queries: ``[K, H]`` or batched ``[B, K, H]`` query tokens.
        source: Source tokens with shape ``[B, N, D_source]``.
        source_valid_mask: Optional boolean mask ``[B, N]`` where ``True``
            denotes a valid source token. At least one token per row must be
            valid.
        return_attention: Return per-head attention probabilities for
            visualization and diversity regularization.
    """

    def __init__(
        self,
        source_dim: int,
        hidden_size: int,
        num_heads: int,
        output_dim: int | None = None,
        ffn_hidden_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if source_dim <= 0 or hidden_size <= 0:
            raise ValueError("source_dim and hidden_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        output_dim = output_dim or hidden_size
        ffn_hidden_size = ffn_hidden_size or hidden_size * 4
        if output_dim <= 0 or ffn_hidden_size <= 0:
            raise ValueError("output_dim and ffn_hidden_size must be positive")

        self.source_dim = source_dim
        self.hidden_size = hidden_size

        self.source_norm = nn.LayerNorm(source_dim)
        self.source_projection = nn.Linear(source_dim, hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.output_projection = (
            nn.Identity()
            if output_dim == hidden_size
            else nn.Linear(hidden_size, output_dim)
        )

    def forward(
        self,
        queries: torch.Tensor,
        source: torch.Tensor,
        source_valid_mask: torch.Tensor | None = None,
        return_attention: bool = True,
    ) -> QueryCrossAttentionOutput:
        self._validate_inputs(queries, source, source_valid_mask)

        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        elif queries.shape[0] == 1 and source.shape[0] != 1:
            queries = queries.expand(source.shape[0], -1, -1)

        source_hidden = self.source_projection(self.source_norm(source))
        key_padding_mask = (
            None if source_valid_mask is None else ~source_valid_mask
        )
        attended, attention_weights = self.cross_attention(
            query=self.query_norm(queries),
            key=source_hidden,
            value=source_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        hidden = queries + self.attention_dropout(attended)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        query_features = self.output_projection(hidden)

        return QueryCrossAttentionOutput(
            query_features=query_features,
            attention_weights=attention_weights if return_attention else None,
        )

    def _validate_inputs(
        self,
        queries: torch.Tensor,
        source: torch.Tensor,
        source_valid_mask: torch.Tensor | None,
    ) -> None:
        if source.ndim != 3:
            raise ValueError(
                f"source must have shape [B, N, D], got {tuple(source.shape)}"
            )
        if source.shape[0] == 0 or source.shape[1] == 0:
            raise ValueError("source batch and token dimensions must be non-empty")
        if source.shape[-1] != self.source_dim:
            raise ValueError(
                f"source feature dimension must be {self.source_dim}, got "
                f"{source.shape[-1]}"
            )
        if queries.ndim not in (2, 3):
            raise ValueError(
                "queries must have shape [K, H] or [B, K, H], got "
                f"{tuple(queries.shape)}"
            )
        if queries.shape[-1] != self.hidden_size:
            raise ValueError(
                f"query feature dimension must be {self.hidden_size}, got "
                f"{queries.shape[-1]}"
            )
        if queries.shape[-2] == 0:
            raise ValueError("queries must contain at least one query token")
        if queries.ndim == 3 and queries.shape[0] not in (1, source.shape[0]):
            raise ValueError(
                "batched queries must have batch size 1 or match the source batch"
            )

        if source_valid_mask is None:
            return
        if source_valid_mask.dtype != torch.bool:
            raise TypeError("source_valid_mask must have dtype torch.bool")
        if source_valid_mask.shape != source.shape[:2]:
            raise ValueError(
                "source_valid_mask must have shape [B, N] matching source"
            )
        if bool((~source_valid_mask.any(dim=1)).any()):
            raise ValueError("each source row must contain at least one valid token")
