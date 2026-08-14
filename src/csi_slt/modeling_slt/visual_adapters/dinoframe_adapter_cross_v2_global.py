"""DINOFrame Cross V2 adapter followed by a global semantic transformer."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2 import (
    DINOFrameAdapterCrossV2,
)
from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized


class DINOFrameAdapterCrossV2Global(nn.Module):
    """Add video-level context and a GLOBAL token to Cross V2 features.

    The wrapped :class:`DINOFrameAdapterCrossV2` first emits two interleaved
    tokens per frame::

        [CLS_0, PATCH_0, CLS_1, PATCH_1, ...]

    This module projects those tokens into a smaller semantic space, adds
    temporal and token-type embeddings, prepends one learned GLOBAL token per
    video, and applies a bidirectional Transformer encoder. Every Transformer
    output is mapped into the LLM embedding space. Local outputs use a gated
    residual so initialization preserves the wrapped adapter's features.

    The packed output for each video is::

        [GLOBAL, CLS_0, PATCH_0, CLS_1, PATCH_1, ...]

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        cls_input_dim: int | None = None,
        temporal_hidden_dim: int | None = None,
        temperature: float = 0.1,
        temporal_gate_init: float = -2.0,
        spatial_window_radius: int | None = 3,
        spatial_grid_size: Sequence[int] | None = None,
        semantic_hidden_dim: int = 512,
        semantic_num_layers: int = 2,
        semantic_num_heads: int = 8,
        semantic_ffn_dim: int | None = None,
        semantic_dropout: float = 0.1,
        max_frames: int = 1024,
        local_residual_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if semantic_hidden_dim <= 0:
            raise ValueError("semantic_hidden_dim must be positive")
        if semantic_num_layers <= 0:
            raise ValueError("semantic_num_layers must be positive")
        if semantic_num_heads <= 0:
            raise ValueError("semantic_num_heads must be positive")
        if semantic_hidden_dim % semantic_num_heads != 0:
            raise ValueError(
                "semantic_hidden_dim must be divisible by semantic_num_heads"
            )
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self.output_dim = output_dim
        self.semantic_hidden_dim = semantic_hidden_dim
        self.max_frames = max_frames

        self.frame_adapter = DINOFrameAdapterCrossV2(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cls_input_dim=cls_input_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            temperature=temperature,
            temporal_gate_init=temporal_gate_init,
            spatial_window_radius=spatial_window_radius,
            spatial_grid_size=spatial_grid_size,
        )

        self.input_projection = nn.Linear(output_dim, semantic_hidden_dim)
        self.output_projection = nn.Linear(semantic_hidden_dim, output_dim)

        self.global_token = nn.Parameter(torch.zeros(1, semantic_hidden_dim))
        self.global_type_embedding = nn.Parameter(torch.zeros(1, semantic_hidden_dim))
        self.cls_type_embedding = nn.Parameter(torch.zeros(1, semantic_hidden_dim))
        self.patch_type_embedding = nn.Parameter(torch.zeros(1, semantic_hidden_dim))
        self.temporal_position_embedding = nn.Embedding(max_frames, semantic_hidden_dim)

        semantic_ffn_dim = semantic_ffn_dim or semantic_hidden_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=semantic_hidden_dim,
            nhead=semantic_num_heads,
            dim_feedforward=semantic_ffn_dim,
            dropout=semantic_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=semantic_num_layers,
            norm=nn.LayerNorm(semantic_hidden_dim),
            enable_nested_tensor=False,
        )

        self.local_residual_gate = nn.Parameter(
            torch.tensor(float(local_residual_gate_init))
        )

        nn.init.trunc_normal_(self.global_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_position_embedding.weight, std=0.02)

        mark_module_tree_as_initialized(self.temporal_position_embedding)

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
        return_weights: bool = True,
    ) -> VisualAdapterOutput:
        frame_output = self.frame_adapter(
            visual_backbone_output,
            permute_video_tokens=permute_video_tokens,
            return_weights=return_weights,
        )

        frame_lengths = visual_backbone_output.visual_length
        if frame_lengths is None:
            raise ValueError(
                "visual_length must be provided for DINOFrameAdapterCrossV2Global"
            )
        if bool((frame_lengths > self.max_frames).any()):
            raise ValueError(
                f"frame length exceeds configured max_frames={self.max_frames}"
            )

        local_lengths = (
            frame_lengths.to(
                device=frame_output.visual_features.device, dtype=torch.long
            )
            * 2
        )
        local_sequences = torch.split(
            frame_output.visual_features, local_lengths.tolist(), dim=0
        )

        semantic_sequences = [
            self._build_semantic_sequence(local_tokens)
            for local_tokens in local_sequences
        ]
        padded_semantic = pad_sequence(
            semantic_sequences, batch_first=True, padding_value=0.0
        )
        semantic_lengths = local_lengths + 1
        token_indices = torch.arange(
            padded_semantic.shape[1], device=padded_semantic.device
        )
        padding_mask = token_indices.unsqueeze(0) >= semantic_lengths.unsqueeze(1)

        contextualized = self.semantic_transformer(
            padded_semantic, src_key_padding_mask=padding_mask
        )
        global_features = self.output_projection(contextualized[:, 0])

        packed_features = []
        packed_position_ids = []
        residual_gate = torch.tanh(self.local_residual_gate)
        for batch_index, (local_tokens, local_length) in enumerate(
            zip(local_sequences, local_lengths.tolist())
        ):
            local_context = contextualized[batch_index, 1 : 1 + local_length]
            refined_local = local_tokens + residual_gate * self.output_projection(
                local_context
            )
            packed_features.append(
                torch.cat(
                    [global_features[batch_index].unsqueeze(0), refined_local], dim=0
                )
            )

            frame_count = local_length // 2
            local_position_ids = torch.arange(
                frame_count, device=refined_local.device
            ).repeat_interleave(2)
            packed_position_ids.append(
                torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long, device=refined_local.device),
                        local_position_ids + 1,
                    ]
                )
            )

        extras = dict(frame_output.extras or {})
        extras.update(
            {
                "local_residual_gate": residual_gate,
            }
        )

        return VisualAdapterOutput(
            visual_features=torch.cat(packed_features, dim=0),
            visual_length=local_lengths + 1,
            position_ids=torch.cat(packed_position_ids, dim=0),
            extras=extras,
        )

    def _build_semantic_sequence(self, local_tokens: torch.Tensor) -> torch.Tensor:
        """Build ``[GLOBAL, CLS_0, PATCH_0, ...]`` in semantic space."""
        if local_tokens.ndim != 2 or local_tokens.shape[-1] != self.output_dim:
            raise ValueError(
                "local tokens must have shape [2T, output_dim], got "
                f"{tuple(local_tokens.shape)}"
            )
        if local_tokens.shape[0] % 2 != 0:
            raise ValueError("Cross V2 local token count must be even")

        frame_count = local_tokens.shape[0] // 2
        hidden = self.input_projection(local_tokens)

        type_embeddings = torch.stack(
            (
                self.cls_type_embedding.squeeze(0),
                self.patch_type_embedding.squeeze(0),
            ),
            dim=0,
        ).repeat(frame_count, 1)
        temporal_positions = torch.arange(frame_count, device=local_tokens.device)
        temporal_embeddings = self.temporal_position_embedding(
            temporal_positions
        ).repeat_interleave(2, dim=0)
        hidden = hidden + type_embeddings + temporal_embeddings

        global_hidden = self.global_token + self.global_type_embedding
        return torch.cat([global_hidden, hidden], dim=0)


if __name__ == "__main__":
    from csi_slt.modeling_slt.output_utils import VisualBackboneOutput

    torch.manual_seed(42)
    frame_lengths = torch.tensor([2, 3])
    total_frames = int(frame_lengths.sum().item())
    backbone_output = VisualBackboneOutput(
        visual_features=torch.randn(total_frames, 4, 16),
        pooled_visual_features=torch.randn(total_frames, 16),
        visual_length=frame_lengths,
    )
    adapter = DINOFrameAdapterCrossV2Global(
        input_dim=16,
        cls_input_dim=16,
        output_dim=32,
        semantic_hidden_dim=16,
        semantic_num_layers=1,
        semantic_num_heads=4,
    )
    output = adapter(backbone_output)
    print("visual features:", output.visual_features.shape)
    print("visual lengths:", output.visual_length)


# Model structure
# ===============
#
#                             Sign video
#                          [sum(T), C, H, W]
#                                  |
#                                  v
#                       +---------------------+
#                       | DINO visual backbone|
#                       +---------------------+
#                          |               |
#                          v               v
#                    Frame CLS         Patch features
#                    [sum(T), D]       [sum(T), P, D]
#                          |               |
#                          +-------+-------+
#                                  |
#                                  v
#                +----------------------------------+
#                | DINOFrameAdapterCrossV2          |
#                |                                  |
#                | next-frame patch alignment       |
#                |          -> temporal difference  |
#                |          -> gated motion fusion  |
#                |          -> spatial pooling      |
#                |          -> LLM projection       |
#                +----------------------------------+
#                                  |
#                                  v
#                  [CLS_0, PATCH_0, CLS_1, PATCH_1, ...]
#                                  |
#                         input projection
#                         output_dim -> semantic_dim
#                                  |
#                 +----------------+----------------+
#                 |                |                |
#                 v                v                v
#             CLS type         PATCH type      temporal position
#             embedding        embedding          embedding
#                 |                |                |
#                 +----------------+----------------+
#                                  |
#                        prepend GLOBAL token
#                                  |
#                                  v
#          [GLOBAL, CLS_0, PATCH_0, CLS_1, PATCH_1, ...]
#                                  |
#                                  v
#                +----------------------------------+
#                | Bidirectional semantic Transformer|
#                |                                  |
#                |     GLOBAL <-> CLS <-> PATCH     |
#                +----------------------------------+
#                         |                  |
#                         v                  v
#               Contextual GLOBAL      Contextual local tokens
#                         |                  |
#                  output projection    output projection
#                         |                  |
#                         |          tanh(residual_gate) * delta
#                         |                  |
#                         |        original local + delta
#                         |                  |
#                         +---------+--------+
#                                   |
#                                   v
#          [GLOBAL, CLS_0, PATCH_0, CLS_1, PATCH_1, ...]
#                     packed length per video = 2T + 1
#                                   |
#                                   |
#                                   v
#                    All visual tokens are exposed
#                    through visual_features only
