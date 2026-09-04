"""Small self-attention context over packed, variable-length video tokens.

``TemporalConvDownsample`` only ever looks within a fixed-radius window, so
its output tokens have no way to attend to arbitrarily distant frames of the
same video. This module adds that: standard pre-norm self-attention + MLP
blocks, applied per video, over the packed token sequence produced downstream
of the temporal downsampling.

Data flow, one block::

    x -----------------------------------------+
    |                                           |
    v                                           |
    LN -> (+ sinusoidal position, Q/K/V) -> MHA-out_proj(=0 init) -> +
    |                                           |
    v                                           |
    LN -> Linear -> GELU -> Linear(=0 init) --> +

Packing is handled with ``packed_to_padded``/``padded_to_packed``: the packed
sequence becomes a ``[B, T, C]`` batch plus a per-video ``key_padding_mask``,
so attention can never cross a video boundary by construction, without a
Python loop over videos the way the boundary-safe convolutions use.

Both branches' final linear layer is zero-initialized, following
``SpatiotemporalSeparableConv``'s convention: the block is an exact identity
at construction time -- ``num_transformer_layers=0`` need not even be a
special case elsewhere, since stacking zero-initialized identity blocks is
itself the identity -- while every parameter still receives gradient from the
first step, because a linear layer's weight gradient depends on its input,
not on the weight's own value.

Position is injected only into the attention query and key, never added to
the residual stream itself. Combined with the zero-initialized output
projection this keeps the identity-at-init property exact: whatever the
position-dependent attention weights compute, it is multiplied by a zero
matrix before it can reach ``x``.
"""

import math

import torch
from torch import Tensor, nn

from csi_slt.modeling_slt.misc import packed_to_padded, padded_to_packed


def _validate_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class _TransformerContextBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        # Zero-initialized so each branch starts as an exact identity while
        # still receiving gradient from the first step -- see module
        # docstring and SpatiotemporalSeparableConv, which established the
        # idiom in this codebase.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: Tensor, key_padding_mask: Tensor, position: Tensor) -> Tensor:
        normed = self.attn_norm(x)
        query_key = normed + position
        attn_out, _ = self.attn(
            query_key,
            query_key,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x


class PackedTransformerContext(nn.Module):
    """Stack of identity-initialized self-attention blocks over packed videos."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dimension("hidden_dim", hidden_dim)
        _validate_dimension("num_layers", num_layers)
        _validate_dimension("num_heads", num_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio!r}")

        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList(
            [
                _TransformerContextBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        # Built lazily against the runtime max sequence length, and cached
        # only as long as it covers what is asked of it -- mirrors the
        # neighbourhood/position caches in next_frame_fusion.py.
        self.register_buffer("_position_encoding", None, persistent=False)

    def forward(self, packed_features: Tensor, visual_length: Tensor) -> Tensor:
        """Contextualize ``[sum(T), C]`` packed tokens without crossing videos."""
        self._validate_inputs(packed_features, visual_length)
        padded, mask = packed_to_padded(packed_features, visual_length)
        key_padding_mask = mask == 0
        position = self._sinusoidal_position_encoding(
            padded.shape[1], padded.device, padded.dtype
        ).unsqueeze(0)

        x = padded
        for block in self.blocks:
            x = block(x, key_padding_mask, position)

        packed, _ = padded_to_packed(x, mask)
        return packed

    def _sinusoidal_position_encoding(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        cached = self._position_encoding
        if cached is None or cached.shape[0] < length or cached.device != device:
            position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
            frequency = torch.exp(
                torch.arange(0, self.hidden_dim, 2, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / self.hidden_dim)
            )
            encoding = torch.zeros(length, self.hidden_dim, device=device)
            # frequency has ceil(hidden_dim / 2) entries to fill the sine
            # slots exactly; an odd hidden_dim has one fewer cosine slot, so
            # it is trimmed to match rather than assuming hidden_dim is even.
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency[: self.hidden_dim // 2])
            self._position_encoding = encoding
            cached = encoding
        return cached[:length].to(dtype=dtype)

    def _validate_inputs(self, packed_features: Tensor, visual_length: Tensor) -> None:
        if packed_features.ndim != 2:
            raise ValueError(
                "packed_features must have shape [sum(T), C], got "
                f"{tuple(packed_features.shape)}"
            )
        if packed_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"feature dimension must be {self.hidden_dim}, got "
                f"{packed_features.shape[-1]}"
            )
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if visual_length.is_floating_point() or visual_length.is_complex():
            raise TypeError("visual_length must use an integer dtype")
        if bool((visual_length <= 0).any()):
            raise ValueError("all entries in visual_length must be positive")
        if int(visual_length.sum().item()) != packed_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed tokens"
            )
