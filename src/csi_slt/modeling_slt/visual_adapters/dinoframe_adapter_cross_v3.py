"""DINO frame adapter with static-motion query fusion.

Default data flow (``window_size=3``, ``stride=2``, ``num_queries=4``)::

    packed frames:       x0   x1   x2   x3   ...       temporal length T
                         |         |
    window centres:      0         2        ...
                         |         |
                         v         v
    boundary-aware:  [x0,x0,x1] [x1,x2,x3] ...        T / 2 windows
                       |  |  |
             +---------+  |  +---------+
             |            |            |
             v            v            v
       static patches   motion delta   three CLS
             |            |            |
             |     shared learned Q    |
             v            v            v
       static CrossAttn  motion CrossAttn  CLS MLP
             |            |          + centre-CLS residual
             +------ query-wise gated fusion ------+
                                |
                                v
                     4 fused tokens / window

The temporal axis is compressed by ``stride`` (2x by default): ``T -> T/2``
window positions. Token count is not reduced relative to CrossV2: four tokens
per window give ``4 * T/2 = 2T`` tokens, matching two tokens per input frame.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from csi_slt.modeling_slt.misc import packed_temporal_windows
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.query_cross_attention import (
    LearnedQueryBank,
    QueryCrossAttention,
)


class DINOFrameAdapterCrossV3(nn.Module):
    """Summarize three-frame patch templates with two learned-query branches.

    Current-to-next-frame differences form the motion source, while original
    patch features form the appearance source. Both branches share one learned
    query bank but keep independent cross-attention parameters. A direct CLS
    window fusion conditions their query-wise gated fusion.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cls_input_dim: int | None = None,
        num_queries: int = 4,
        query_hidden_dim: int | None = None,
        num_attention_heads: int = 4,
        query_ffn_hidden_dim: int | None = None,
        attention_dropout: float = 0.0,
        temporal_window_size: int = 3,
        temporal_window_stride: int = 2,
        temperature: float = 0.1,
        spatial_window_radius: int | None = 3,
        spatial_grid_size: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if isinstance(temporal_window_size, bool) or not isinstance(
            temporal_window_size, int
        ):
            raise TypeError("temporal_window_size must be an integer")
        if temporal_window_size <= 0:
            raise ValueError("temporal_window_size must be positive")
        if temporal_window_stride <= 0:
            raise ValueError("temporal_window_stride must be positive")
        if spatial_window_radius is not None and (
            isinstance(spatial_window_radius, bool)
            or not isinstance(spatial_window_radius, int)
            or spatial_window_radius < 0
        ):
            raise ValueError("spatial_window_radius must be a non-negative integer")
        if spatial_grid_size is not None:
            if (
                isinstance(spatial_grid_size, (str, bytes))
                or len(spatial_grid_size) != 2
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in spatial_grid_size
                )
            ):
                raise ValueError("spatial_grid_size must contain two positive integers")
            spatial_grid_size = tuple(spatial_grid_size)

        query_hidden_dim = query_hidden_dim or output_dim
        cls_input_dim = cls_input_dim or input_dim
        fusion_ffn_hidden_dim = query_ffn_hidden_dim or query_hidden_dim * 4
        self.input_dim = input_dim
        self.cls_input_dim = cls_input_dim
        self.num_queries = num_queries
        self.temporal_window_size = temporal_window_size
        self.temporal_window_stride = temporal_window_stride
        self.temperature = temperature
        self.spatial_window_radius = spatial_window_radius
        self.spatial_grid_size = spatial_grid_size

        # Shared relative frame positions align static and motion tokens from
        # the same ordered window slots, including half-centred even windows.
        self.frame_position_embedding = nn.Parameter(
            torch.empty(1, temporal_window_size, 1, input_dim)
        )  # [1, W, 1, D_patch]
        nn.init.normal_(self.frame_position_embedding, std=0.02)

        # One semantic query bank reads both sources through independent
        # attention branches, aligning static and motion slot k.
        self.query_bank = LearnedQueryBank(num_queries, query_hidden_dim)
        self.temporal_cross_attention = QueryCrossAttention(
            source_dim=input_dim,
            hidden_size=query_hidden_dim,
            num_heads=num_attention_heads,
            output_dim=query_hidden_dim,
            ffn_hidden_size=query_ffn_hidden_dim,
            dropout=attention_dropout,
        )
        self.patch_cross_attention = QueryCrossAttention(
            source_dim=input_dim,
            hidden_size=query_hidden_dim,
            num_heads=num_attention_heads,
            output_dim=query_hidden_dim,
            ffn_hidden_size=query_ffn_hidden_dim,
            dropout=attention_dropout,
        )

        # Map each CLS independently, fuse the complete window, then add the
        # mapped centre CLS as an explicit residual.
        self.cls_frame_norm = nn.LayerNorm(cls_input_dim)
        self.cls_frame_projection = nn.Linear(cls_input_dim, query_hidden_dim)
        cls_window_dim = temporal_window_size * query_hidden_dim
        self.cls_window_norm = nn.LayerNorm(cls_window_dim)
        self.cls_window_mlp = nn.Sequential(
            nn.Linear(cls_window_dim, fusion_ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_ffn_hidden_dim, query_hidden_dim),
        )

        self.gate_norm = nn.LayerNorm(query_hidden_dim * 3)
        self.motion_gate = nn.Sequential(
            nn.Linear(query_hidden_dim * 3, query_hidden_dim),
            nn.GELU(),
            nn.Linear(query_hidden_dim, query_hidden_dim),
        )
        self.cls_condition_projection = nn.Linear(query_hidden_dim, query_hidden_dim)
        self.fusion_norm = nn.LayerNorm(query_hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(query_hidden_dim, fusion_ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(fusion_ffn_hidden_dim, query_hidden_dim),
            nn.Dropout(attention_dropout),
        )
        self.output_projection = nn.Linear(query_hidden_dim, output_dim)

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
        return_weights: bool = True,
    ) -> VisualAdapterOutput:
        patch_features = visual_backbone_output.visual_features  # [F, P, D_patch]
        cls_token = visual_backbone_output.pooled_visual_features  # [F, D_cls]
        visual_length = visual_backbone_output.visual_length  # [B]

        if visual_length is None:
            raise ValueError(
                "visual_length must be provided for DINOFrameAdapterCrossV3"
            )
        if patch_features is None or cls_token is None:
            raise ValueError(
                "patch_features and cls_token must be provided for "
                "DINOFrameAdapterCrossV3"
            )

        self._validate_inputs(patch_features, cls_token, visual_length)

        # A left content shift makes position t receive x_(t+1), so V2
        # combines the current frame with the next frame (not the previous one).
        next_patches, has_next = self._next_frame_shift(patch_features, visual_length)

        aligned_next = self.similarity_aggregate(patch_features, next_patches)

        # WARN: Do not create artificial motion at the last frame of a video.
        # Although its shifted value is itself, self-aggregation can still mix
        # spatial patches, so masking the residual is necessary.
        temporal_delta = aligned_next - patch_features
        temporal_delta = temporal_delta * has_next[:, None, None].to(
            dtype=temporal_delta.dtype
        )

        window_size = self.temporal_window_size
        window_stride = self.temporal_window_stride
        # G = sum(visual_length // stride).
        temporal_delta, grouped_visual_length = packed_temporal_windows(
            temporal_delta,
            visual_length,
            window_size=window_size,
            stride=window_stride,
        )  # [G, W, P, D_patch], [B]
        patch_features, patch_grouped_visual_length = packed_temporal_windows(
            patch_features,
            visual_length,
            window_size=window_size,
            stride=window_stride,
        )  # [G, W, P, D_patch], [B]
        cls_windows, cls_grouped_visual_length = packed_temporal_windows(
            cls_token,
            visual_length,
            window_size=window_size,
            stride=window_stride,
        )  # [G, W, D_cls], [B]
        grouped_has_next, mask_grouped_visual_length = packed_temporal_windows(
            has_next,
            visual_length,
            window_size=window_size,
            stride=window_stride,
        )  # [G, W], [B]
        if not (
            torch.equal(grouped_visual_length, patch_grouped_visual_length)
            and torch.equal(grouped_visual_length, cls_grouped_visual_length)
            and torch.equal(grouped_visual_length, mask_grouped_visual_length)
        ):
            raise RuntimeError("packed temporal group lengths diverged")

        # Encode each patch's relative frame slot before merging W and P. The
        # same embedding is shared by static and motion branches so their query
        # slots use aligned temporal semantics.
        patch_features = patch_features + self.frame_position_embedding
        temporal_delta = temporal_delta + self.frame_position_embedding

        patches_per_frame = patch_features.shape[2]
        temporal_delta = temporal_delta.flatten(1, 2)  # [G, WP, D_patch]
        patch_features = patch_features.flatten(1, 2)  # [G, WP, D_patch]
        grouped_has_next = (
            grouped_has_next[..., None].expand(-1, -1, patches_per_frame).flatten(1, 2)
        )  # [G, WP]

        group_count = temporal_delta.shape[0]
        shared_queries = self.query_bank(group_count)  # [G, N, H]
        temporal_attention = self.temporal_cross_attention(
            queries=shared_queries,
            source=temporal_delta,
            source_valid_mask=grouped_has_next,
            return_attention=return_weights,
        )  # query_features: [G, N, H]
        patch_attention = self.patch_cross_attention(
            queries=shared_queries,
            source=patch_features,
            return_attention=return_weights,
        )  # query_features: [G, N, H]

        mapped_cls = self.cls_frame_projection(
            self.cls_frame_norm(cls_windows)
        )  # [G, W, H]
        middle = window_size // 2
        if window_size % 2 == 1:
            centre_cls = mapped_cls[:, middle]  # [G, H]
        else:
            # An even window is centred between its two middle frames, so its
            # CLS residual treats both sides symmetrically.
            centre_cls = (
                mapped_cls[:, middle - 1] + mapped_cls[:, middle]
            ) * 0.5  # [G, H]
        cls_context = centre_cls + self.cls_window_mlp(
            self.cls_window_norm(mapped_cls.flatten(1))
        )  # [G, H], with the centre-frame residual

        expanded_cls = cls_context[:, None, :].expand(-1, self.num_queries, -1)
        static_queries = patch_attention.query_features
        motion_queries = temporal_attention.query_features
        gate_input = torch.cat((static_queries, motion_queries, expanded_cls), dim=-1)
        motion_gate = torch.sigmoid(
            self.motion_gate(self.gate_norm(gate_input))
        )  # [G, N, H]
        # Both cross-attention branches contain the same learned-query identity
        # residual. Static is the base path, so inject only the motion branch's
        # learned update instead of counting the shared queries twice.
        motion_residual = motion_queries - shared_queries
        fused_queries = (
            static_queries
            + motion_gate * motion_residual
            + self.cls_condition_projection(cls_context)[:, None, :]
        )
        fused_queries = fused_queries + self.fusion_ffn(self.fusion_norm(fused_queries))
        fused_queries = self.output_projection(fused_queries)  # [G, N, D_out]

        # if permute_video_tokens is True, randomly shuffle the order of frames within each video.
        if permute_video_tokens:
            raise NotImplementedError(
                "permute_video_tokens is not implemented for DINOFrameAdapterCrossV3"
            )

        # Temporal compression is T -> T / stride (2x by default). Each window
        # emits N query tokens, so total tokens are N * T / stride; with N=4
        # and stride=2 this is 2T, matching CrossV2's token budget.
        tokens_per_group = self.num_queries
        visual_features = fused_queries.flatten(0, 1)  # [G * N, D_out]
        visual_length = grouped_visual_length * tokens_per_group  # [B]
        position_ids = torch.cat(
            [
                torch.arange(length, device=visual_features.device).repeat_interleave(
                    tokens_per_group
                )
                for length in grouped_visual_length
            ]
        )  # [G * N]
        extras = None
        if return_weights:
            extras = {
                "temporal_attention_weights": temporal_attention.attention_weights,
                "patch_attention_weights": patch_attention.attention_weights,
                "temporal_source_valid_mask": grouped_has_next,
                "motion_gate": motion_gate,
                "motion_residual": motion_residual,
                "cls_context": cls_context,
            }

        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=visual_length,
            position_ids=position_ids,
            extras=extras,
        )

    def similarity_aggregate(
        self,
        base: Tensor,
        shifted: Tensor,
    ) -> Tensor:
        """Align next-frame patches to current-frame patches."""
        base_norm = F.normalize(base, dim=-1)
        shifted_norm = F.normalize(shifted, dim=-1)
        similarity = torch.einsum("bnd,btd->bnt", base_norm, shifted_norm)

        # Restrict every current-frame patch to the same spatial neighbourhood
        # in the next frame. Chebyshev distance gives a (2r + 1) x (2r + 1)
        # window and, unlike flattened-index distance, never wraps across rows.
        if self.spatial_window_radius is not None:
            spatial_mask = self._spatial_neighbourhood_mask(
                num_patches=base.shape[1], device=base.device
            )
            similarity = similarity.masked_fill(
                ~spatial_mask, torch.finfo(similarity.dtype).min
            )

        # NOTE: Cosine logits lie in [-1, 1]; temperature prevents attention
        # over many patches from becoming excessively uniform.
        weights = F.softmax(similarity / self.temperature, dim=-1)
        return torch.einsum("bnt,btd->bnd", weights, shifted)

    def _spatial_neighbourhood_mask(
        self, num_patches: int, device: torch.device
    ) -> Tensor:
        """Return a ``[P, P]`` local matching mask for a row-major patch grid."""
        if self.spatial_window_radius is None:
            raise RuntimeError("spatial_window_radius is disabled")

        if self.spatial_grid_size is None:
            side = math.isqrt(num_patches)
            if side * side != num_patches:
                raise ValueError(
                    "cannot infer a square patch grid from "
                    f"{num_patches} patches; set spatial_grid_size=[height, width]"
                )
            grid_height, grid_width = side, side
        else:
            grid_height, grid_width = self.spatial_grid_size
            if grid_height * grid_width != num_patches:
                raise ValueError(
                    "spatial_grid_size does not match the patch count: "
                    f"{grid_height} * {grid_width} != {num_patches}"
                )

        indices = torch.arange(num_patches, device=device)
        rows = torch.div(indices, grid_width, rounding_mode="floor")
        columns = indices.remainder(grid_width)
        row_distance = (rows[:, None] - rows[None, :]).abs()
        column_distance = (columns[:, None] - columns[None, :]).abs()
        return torch.maximum(row_distance, column_distance).le(
            self.spatial_window_radius
        )

    @staticmethod
    def _next_frame_shift(
        visual_features: Tensor,
        visual_length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Move each next frame to the current position within each video.

        Returns:
            shifted: Next-frame features. A video's final position keeps its
                own features to ensure every gather index is valid.
            has_next: Boolean mask with shape ``[F]``. It is false at the last
                frame of every video.
        """
        total_frames = visual_features.shape[0]
        device = visual_features.device
        boundaries = torch.cumsum(visual_length.to(device=device), dim=0)
        ends = boundaries - 1

        base = torch.arange(total_frames, device=device)
        has_next = torch.ones(total_frames, dtype=torch.bool, device=device)
        has_next[ends] = False
        source_idx = base + has_next.to(dtype=base.dtype)

        # WARN: There is deliberately no torch.no_grad() here. Index gathering
        # should preserve gradients to next-frame features when DINO is tuned.
        return visual_features[source_idx], has_next

    def _validate_inputs(
        self,
        patch_features: Tensor,
        cls_token: Tensor,
        visual_length: Tensor,
    ) -> None:
        if patch_features.ndim != 3:
            raise ValueError(
                "patch_features must have shape [F, P, D_patch], got "
                f"{tuple(patch_features.shape)}"
            )
        if cls_token.ndim != 2:
            raise ValueError(
                f"cls_token must have shape [F, D_cls], got {tuple(cls_token.shape)}"
            )
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if bool((visual_length <= 0).any()):
            raise ValueError("all entries in visual_length must be positive")
        if patch_features.shape[0] != cls_token.shape[0]:
            raise ValueError("patch_features and cls_token must have the same F")
        if patch_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"patch feature dimension must be {self.input_dim}, got "
                f"{patch_features.shape[-1]}"
            )
        if cls_token.shape[-1] != self.cls_input_dim:
            raise ValueError(
                f"CLS feature dimension must be {self.cls_input_dim}, got "
                f"{cls_token.shape[-1]}"
            )
        if int(visual_length.sum().item()) != patch_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed frames"
            )


if __name__ == "__main__":
    import torch
    from torch import Tensor

    # Original adapter test
    B, N, D_PATCH, D_CLS = 10, 16, 768, 1024
    cls_token = torch.randn(B, D_CLS).cuda()
    patch_features = torch.randn(B, N, D_PATCH).cuda()
    visual_length = torch.tensor([4, 6]).cuda()

    visual_backbone_output = VisualBackboneOutput(
        visual_features=patch_features,
        pooled_visual_features=cls_token,
        visual_length=visual_length,
    )

    adapter = DINOFrameAdapterCrossV3(
        input_dim=D_PATCH,
        cls_input_dim=D_CLS,
        output_dim=512,
    ).cuda()
    adapter.eval()
    with torch.no_grad():
        output = adapter(visual_backbone_output, return_weights=True)
        print("Output shape:", output.visual_features.shape)
        print(
            "Temporal attention shape:",
            output.extras["temporal_attention_weights"].shape,
        )
        print("Visual length:", output.visual_length)
