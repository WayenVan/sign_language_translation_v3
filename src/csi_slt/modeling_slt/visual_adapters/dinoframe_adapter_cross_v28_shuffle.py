"""Temporal-compression wrapper for :class:`DINOFrameAdapterCrossV2`."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput


class TemporalShuffleAdapter(nn.Module):
    def __init__(
        self,
        input_hidden_size,  # per-token input width
        output_hidden_size,  # per-token output width (also the base path's width)
        scale_factor,  # how many consecutive frame tokens fuse into one; must divide every video length
        mlp_depth=1,  # NOTE: currently unused/dead, kept only for config compatibility (see comment below)
        motion_hidden_dim: int | None = None,  # SwiGLU gate branch's internal width; defaults to output_hidden_size
        dropout: float = 0.1,  # dropout on the SwiGLU activation, inside the gated motion branch only
    ):
        """Fuse every ``s`` consecutive frame tokens into one token.

        ``z = [x_t; ...; x_{t+s-1}; Δx_t; ...; Δx_{t+s-2}]``
        ``y = Project(mean(x)) + sigmoid(g) * SwiGLU(LN(z))``
        ``T_out = T_in / s``

        ``z`` concatenates ordered frame features and adjacent-frame
        differences (``Δx_i = x_{i+1} - x_i``). ``g`` is a learnable scalar
        gate that controls the motion residual strength. The gated branch's
        internal width is ``motion_hidden_dim``, decoupled from
        ``output_hidden_size`` so it can be given its own expansion instead
        of being forced to exactly the output width.

        Each video length must be divisible by ``s`` so a window never spans
        two videos in the packed batch.
        """
        super().__init__()
        if scale_factor < 2:
            raise ValueError(
                "Motion-aware temporal fusion requires scale_factor to be at least 2"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        motion_hidden_dim = motion_hidden_dim or output_hidden_size

        self.scale_factor = scale_factor
        # Kept as an attribute for configuration compatibility.  The previous
        # stack of GELU MLP layers is replaced by a single SwiGLU fusion block.
        self.mlp_depth = mlp_depth

        # The base path preserves information shared by the frames in a local
        # window.  It gives the adapter a stable pooling-like initial route.
        self.base_norm = nn.LayerNorm(input_hidden_size)
        self.base_projection = nn.Linear(input_hidden_size, output_hidden_size)

        # Besides the ordered frame features, feed first-order temporal
        # differences to the nonlinear path.  For scale_factor=2 this is
        # exactly [x_t, x_{t+1}, x_{t+1} - x_t].
        fusion_input_size = input_hidden_size * (2 * scale_factor - 1)
        self.fusion_norm = nn.LayerNorm(fusion_input_size)
        self.fusion_in_projection = nn.Linear(fusion_input_size, motion_hidden_dim * 2)
        # NOTE: Dropout sits on the SwiGLU activation, i.e. strictly inside the
        # gated motion branch.  The base path stays deterministic, so a window's
        # pooled content is never dropped.
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_out_projection = nn.Linear(motion_hidden_dim, output_hidden_size)

        # Begin close to the stable base path, then learn how much motion
        # residual to inject. sigmoid(-2) is approximately 0.12.
        # Shape (1,) rather than a scalar: FSDP2 shards along dim 0 and
        # rejects 0-dim parameters outright. Broadcasting is unchanged.
        self.motion_gate = nn.Parameter(torch.tensor([-2.0]))

    def temporal_shuffle(self, x, t_length, scale_factor=2):
        # x [BT, D]
        #
        assert t_length.fmod(scale_factor).eq(0).all(), (
            "temporal length of all frames must be divisible by scale_factor"
        )
        _, D = x.size()
        x = rearrange(x, "(n s) d -> n s d", s=scale_factor, d=D)
        return x

    def forward(self, hidden_states, t_length):
        """
        hidden_states: shape of [B1+B2+B3..., D] , the concatenation of all temporal tokens in the batch
        t_length: exact value of [B1, B2, B3...], the temporal length of each sample in the batch
        """
        if hidden_states is None or t_length is None:
            raise ValueError(
                "TemporalShuffleAdapter requires pooled_visual_features and visual_length from visual_backbone_output"
            )
        frame_windows = self.temporal_shuffle(
            hidden_states, t_length, self.scale_factor
        )

        # Static/context path: average frame content in each local window.
        base = self.base_projection(self.base_norm(frame_windows.mean(dim=1)))

        # Motion path: preserve ordered frames and explicitly expose their
        # frame-to-frame changes to a gated nonlinear projection.
        frame_features = frame_windows.flatten(start_dim=1)
        temporal_deltas = (frame_windows[:, 1:] - frame_windows[:, :-1]).flatten(
            start_dim=1
        )
        fusion_input = torch.cat((frame_features, temporal_deltas), dim=-1)
        value, gate = self.fusion_in_projection(self.fusion_norm(fusion_input)).chunk(
            2, dim=-1
        )
        motion = self.fusion_out_projection(self.fusion_dropout(value * F.silu(gate)))
        hidden_states = base + torch.sigmoid(self.motion_gate) * motion

        if t_length is not None:
            t_length = t_length // self.scale_factor

        return hidden_states, t_length


class DINOFrameAdapterCrossV2(nn.Module):
    """Convert packed DINOv2 features into two tokens per frame.

    For each current frame, patches from the next frame are first aligned with
    the current patches using cosine-similarity aggregation. The aligned
    temporal difference is then fused through a residual MLP:

        fused_t = x_t + gate * MLP(LN(aligned(x_{t+1}) - x_t))

    Frames are packed along dimension 0 and ``visual_length`` defines the video
    boundaries. The last frame of every video has no next frame, so its
    temporal residual is explicitly set to zero. Each frame produces two
    interleaved LLM tokens: ``[mapped_cls_t, mapped_fused_patch_t]``. The
    global summary and spatial patch features may have different input widths;
    each token type therefore has its own input mapper before a shared output
    projection into the LLM embedding space.
    """

    def __init__(
        self,
        input_dim: int,  # DINO patch feature width
        output_dim: int,  # final per-token width handed to the LLM
        hidden_dim: int | None = None,  # cls_mapper/fused_patch_mapper width before output_projection; defaults to output_dim
        cls_input_dim: int | None = None,  # DINO global CLS feature width; defaults to input_dim
        temporal_hidden_dim: int | None = None,  # temporal_mlp's hidden width; defaults to input_dim
        temporal_mlp_depth: int = 2,  # temporal_mlp Linear-layer count; 2 = original Linear->GELU->Linear
        patch_score_hidden_dim: int | None = None,  # patch_score's hidden width; defaults to input_dim
        patch_score_depth: int = 2,  # patch_score Linear-layer count; 1 = original bare linear scorer
        temperature: float = 0.1,  # softmax temperature for next-frame patch alignment
        temporal_gate_init: float = -2.0,  # will be passed through sigmoid to get initial gate value , sigmoid(-2) ~= 0.12
        spatial_window_radius: int | None = None,  # if set, only match patches within this Chebyshev-distance window; None = match against every next-frame patch
        spatial_grid_size: Sequence[int] | None = None,  # [height, width] of the patch grid; None = inferred as a square from patch count
        proj_dropout: float = 0.1,  # dropout after the GELU of cls_mapper/fused_patch_mapper, before the shared output_projection
    ) -> None:
        super().__init__()

        if not 0.0 <= proj_dropout < 1.0:
            raise ValueError(f"proj_dropout must be in [0, 1), got {proj_dropout}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if temporal_mlp_depth < 1:
            raise ValueError(
                f"temporal_mlp_depth must be at least 1, got {temporal_mlp_depth}"
            )
        if patch_score_depth < 1:
            raise ValueError(
                f"patch_score_depth must be at least 1, got {patch_score_depth}"
            )
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

        hidden_dim = hidden_dim or output_dim
        if cls_input_dim is None:
            cls_input_dim = input_dim
        temporal_hidden_dim = temporal_hidden_dim or input_dim
        patch_score_hidden_dim = patch_score_hidden_dim or input_dim
        self.input_dim = input_dim
        self.cls_input_dim = cls_input_dim
        self.temperature = temperature
        self.spatial_window_radius = spatial_window_radius
        self.spatial_grid_size = spatial_grid_size

        # NOTE: V2 learns an explicit residual transformation of the aligned
        # next-frame difference instead of concatenating two frame features.
        # temporal_mlp_depth counts Linear layers, so depth=2 (the default)
        # reproduces the original Linear -> GELU -> Linear shape.
        self.temporal_norm = nn.LayerNorm(input_dim)
        self.temporal_mlp = self._build_mlp(
            input_dim, temporal_hidden_dim, input_dim, num_layers=temporal_mlp_depth
        )

        # NOTE: Start with a small temporal contribution (sigmoid(-2) ~= 0.12)
        # so that early training remains close to the original DINO features.
        # Shape (1,) for FSDP2 compatibility; see motion_gate above.
        self.temporal_gate = nn.Parameter(
            torch.tensor([float(temporal_gate_init)])
        )

        # NOTE: Fused patches remain D-dimensional in V2, rather than becoming
        # 2D-dimensional through concatenation. bias=False on the last layer is
        # harmless either way since softmax is shift-invariant, but keeping it
        # off avoids a free parameter with no effect on the pooling weights.
        self.patch_score = nn.Sequential(
            nn.LayerNorm(input_dim),
            *self._build_mlp(
                input_dim,
                patch_score_hidden_dim,
                1,
                num_layers=patch_score_depth,
                output_bias=False,
            ),
        )

        # NOTE: CLS summaries and spatial features may come from different
        # feature spaces and may not have the same width. Map them independently
        # into a common hidden space, then share only the final LLM projection.
        # NOTE: Dropout goes after each mapper's GELU, so it regularizes the
        # hidden projector space rather than the LLM-facing tokens themselves.
        self.cls_mapper = nn.Sequential(
            nn.LayerNorm(cls_input_dim),
            nn.Linear(cls_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.fused_patch_mapper = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # NOTE: Token-type embeddings let the LLM distinguish global CLS tokens
        # from local motion-aware patch tokens despite the shared mapper.
        self.cls_type_embedding = nn.Parameter(torch.zeros(1, output_dim))
        self.fused_patch_type_embedding = nn.Parameter(torch.zeros(1, output_dim))

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        output_bias: bool = True,
    ) -> nn.Sequential:
        """Build a ``num_layers``-deep MLP: ``(Linear -> GELU) * (n-1) -> Linear``.

        ``num_layers`` counts Linear layers, so ``num_layers=1`` is a single
        Linear straight from ``input_dim`` to ``output_dim`` with no hidden
        layer at all.
        """
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim, bias=output_bias))
        return nn.Sequential(*layers)

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
        return_weights: bool = True,
    ) -> VisualAdapterOutput:
        patch_features = visual_backbone_output.visual_features
        cls_token = visual_backbone_output.pooled_visual_features
        visual_length = visual_backbone_output.visual_length

        if visual_length is None:
            raise ValueError(
                "visual_length must be provided for DINOFrameAdapterCrossV2"
            )
        if patch_features is None or cls_token is None:
            raise ValueError(
                "patch_features and cls_token must be provided for "
                "DINOFrameAdapterCrossV2"
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

        # Motion-aware residual fusion proposed for V2.
        temporal_residual = self.temporal_mlp(self.temporal_norm(temporal_delta))
        # Mask again after the biased MLP so an all-zero input cannot
        # produce a learned non-zero residual at a video's final frame.
        temporal_residual = temporal_residual * has_next[:, None, None].to(
            dtype=temporal_residual.dtype
        )
        fused_patches = (
            patch_features + torch.sigmoid(self.temporal_gate) * temporal_residual
        )

        patch_weights = self.patch_score(fused_patches).squeeze(-1)
        patch_weights = patch_weights.softmax(dim=1)
        pooled_patches = torch.bmm(patch_weights.unsqueeze(1), fused_patches).squeeze(1)

        # Map each token type into a common hidden space, apply the shared LLM
        # projection, then interleave them frame by frame:
        # [CLS_0, PATCH_0, CLS_1, PATCH_1, ...].
        mapped_cls = (
            self.output_projection(self.cls_mapper(cls_token)) + self.cls_type_embedding
        )
        mapped_fused_patches = (
            self.output_projection(self.fused_patch_mapper(pooled_patches))
            + self.fused_patch_type_embedding
        )

        # if permute_video_tokens is True, randomly shuffle the order of frames within each video.
        if permute_video_tokens:
            mapped_cls, mapped_fused_patches, patch_weights = (
                self._permute_video_tokens(
                    mapped_cls, mapped_fused_patches, patch_weights, visual_length
                )
            )

        visual_features = torch.stack(
            (mapped_cls, mapped_fused_patches), dim=1
        ).flatten(0, 1)

        # NOTE: Both tokens belonging to frame t share temporal position t.
        position_ids = torch.cat(
            [
                torch.arange(length, device=visual_features.device).repeat_interleave(2)
                for length in visual_length
            ]
        )

        return VisualAdapterOutput(
            visual_features=visual_features,
            # NOTE: V2 emits two visual tokens per input frame.
            visual_length=visual_length * 2,
            position_ids=position_ids,
            extras={"patch_weights": patch_weights} if return_weights else None,
        )

    @staticmethod
    def _permute_video_tokens(cls, fused_patch, patch_weights, visual_length):
        permutation = random_derangement(visual_length, device=cls.device)
        return (cls[permutation], fused_patch[permutation], patch_weights[permutation])

    def similarity_aggregate(
        self,
        base: Tensor,
        shifted: Tensor,
    ) -> Tensor:
        """Align next-frame patches to current-frame patches.

        By default, every current-frame patch can match every next-frame
        patch. A local matching window is applied only when
        ``spatial_window_radius`` is explicitly configured.
        """
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


class PackedShortTemporalConv(nn.Module):
    """Add a local temporal residual to packed variable-length sequences.

    Videos are padded and convolved as separate batch entries, so a temporal
    kernel can never cross a packed-video boundary.  The depthwise convolution
    gives every feature channel a short-range temporal bias without introducing
    a large channel-mixing projection.  Its zero initialization preserves the
    input exactly at construction while still allowing convolution weights to
    receive gradients immediately through the non-zero residual gate.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.norm = nn.LayerNorm(hidden_size)
        self.temporal_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
            bias=False,
        )
        self.activation = nn.GELU()
        # Shape (1,) for FSDP2 compatibility; see motion_gate above.
        self.residual_gate = nn.Parameter(torch.tensor([float(gate_init)]))

        nn.init.zeros_(self.temporal_conv.weight)
        mark_module_tree_as_initialized(self)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [sum(T), D], got {tuple(features.shape)}"
            )
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty 1D tensor")
        if lengths.is_floating_point() or lengths.is_complex():
            raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
        if bool((lengths <= 0).any()):
            raise ValueError("all temporal lengths must be positive")
        if int(lengths.sum().item()) != features.shape[0]:
            raise ValueError("lengths.sum() must equal the packed token count")

        sequences = torch.split(features, lengths.tolist(), dim=0)
        normalized = [self.norm(sequence) for sequence in sequences]
        padded = pad_sequence(normalized, batch_first=True, padding_value=0.0)
        temporal_residual = self.temporal_conv(padded.transpose(1, 2)).transpose(1, 2)
        temporal_residual = self.activation(temporal_residual)
        packed_residual = torch.cat(
            [
                temporal_residual[index, :length]
                for index, length in enumerate(lengths.tolist())
            ],
            dim=0,
        )
        return features + torch.sigmoid(self.residual_gate) * packed_residual


class DINOFrameAdapterCrossV2Shuffle(nn.Module):
    """Apply V2, then independently temporally compress its CLS/PATCH streams.

    The wrapped V2 adapter is unchanged and produces packed interleaved tokens
    ``[CLS_0, PATCH_0, CLS_1, PATCH_1, ...]``.  This wrapper de-interleaves
    those tokens, applies separate temporal shuffle modules to the two streams,
    and interleaves the compressed streams again.

    Every input video length must be divisible by ``temporal_scale_factor``.
    Consequently, a video with ``T`` frames emits
    ``2 * (T / temporal_scale_factor)`` visual tokens.
    """

    def __init__(
        self,
        input_dim: int,  # DINO patch feature width, forwarded to DINOFrameAdapterCrossV2
        output_dim: int,  # final per-token width handed to the LLM
        hidden_dim: int | None = None,  # see DINOFrameAdapterCrossV2
        cls_input_dim: int | None = None,  # see DINOFrameAdapterCrossV2; defaults to input_dim
        temporal_hidden_dim: int | None = None,  # see DINOFrameAdapterCrossV2; defaults to input_dim
        temporal_mlp_depth: int = 2,  # see DINOFrameAdapterCrossV2
        patch_score_hidden_dim: int | None = None,  # see DINOFrameAdapterCrossV2; defaults to input_dim
        patch_score_depth: int = 2,  # see DINOFrameAdapterCrossV2
        temperature: float = 0.1,  # see DINOFrameAdapterCrossV2
        temporal_gate_init: float = -2.0,  # see DINOFrameAdapterCrossV2
        spatial_window_radius: int | None = 3,  # see DINOFrameAdapterCrossV2
        spatial_grid_size: Sequence[int] | None = None,  # see DINOFrameAdapterCrossV2
        temporal_scale_factor: int = 2,  # how many frame-pair tokens each TemporalShuffleAdapter compresses into one; must be >= 2 and divide every video's frame count
        motion_hidden_dim: int | None = None,  # shared with both cls_/patch_temporal_shuffle; see TemporalShuffleAdapter; defaults to output_dim
        proj_dropout: float = 0.1,  # see DINOFrameAdapterCrossV2
        motion_dropout: float = 0.1,  # shared with both cls_/patch_temporal_shuffle; see TemporalShuffleAdapter
        use_short_temporal_conv: bool = False,  # if True, add an extra PackedShortTemporalConv after each shuffle stream
        short_temporal_kernel_size: int = 3,  # PackedShortTemporalConv kernel size, only used when use_short_temporal_conv=True
        short_temporal_gate_init: float = -2.0,  # PackedShortTemporalConv residual gate init, only used when use_short_temporal_conv=True
    ) -> None:
        super().__init__()
        if temporal_scale_factor < 2:
            raise ValueError(
                "temporal_scale_factor must be at least 2 for temporal compression"
            )

        self.temporal_scale_factor = temporal_scale_factor
        self.use_short_temporal_conv = use_short_temporal_conv
        self.frame_adapter = DINOFrameAdapterCrossV2(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cls_input_dim=cls_input_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            temporal_mlp_depth=temporal_mlp_depth,
            patch_score_hidden_dim=patch_score_hidden_dim,
            patch_score_depth=patch_score_depth,
            temperature=temperature,
            temporal_gate_init=temporal_gate_init,
            spatial_window_radius=spatial_window_radius,
            spatial_grid_size=spatial_grid_size,
            proj_dropout=proj_dropout,
        )
        # CLS and pooled-patch tokens have different distributions, so the
        # compression branches intentionally do not share parameters.
        self.cls_temporal_shuffle = TemporalShuffleAdapter(
            input_hidden_size=output_dim,
            output_hidden_size=output_dim,
            scale_factor=temporal_scale_factor,
            motion_hidden_dim=motion_hidden_dim,
            dropout=motion_dropout,
        )
        self.patch_temporal_shuffle = TemporalShuffleAdapter(
            input_hidden_size=output_dim,
            output_hidden_size=output_dim,
            scale_factor=temporal_scale_factor,
            motion_hidden_dim=motion_hidden_dim,
            dropout=motion_dropout,
        )
        # Keep the semantic CLS stream and motion-oriented PATCH stream
        # independent when the optional post-shuffle temporal bias is enabled.
        # Do not construct these modules in the default path, keeping its
        # parameters and primary output identical to commit 5f86588.
        if use_short_temporal_conv:
            self.cls_short_temporal_conv = PackedShortTemporalConv(
                hidden_size=output_dim,
                kernel_size=short_temporal_kernel_size,
                gate_init=short_temporal_gate_init,
            )
            self.patch_short_temporal_conv = PackedShortTemporalConv(
                hidden_size=output_dim,
                kernel_size=short_temporal_kernel_size,
                gate_init=short_temporal_gate_init,
            )
        else:
            self.cls_short_temporal_conv = None
            self.patch_short_temporal_conv = None

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        return_weights: bool = True,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        frame_length = visual_backbone_output.visual_length
        if frame_length is None:
            raise ValueError(
                "visual_length must be provided for DINOFrameAdapterCrossV2Shuffle"
            )

        frame_output = self.frame_adapter(
            visual_backbone_output,
            return_weights=return_weights,
            permute_video_tokens=permute_video_tokens,
        )
        expected_frame_tokens = frame_length * 2
        if not torch.equal(frame_output.visual_length, expected_frame_tokens):
            raise RuntimeError(
                "wrapped DINOFrameAdapterCrossV2 returned unexpected lengths"
            )

        interleaved_tokens = frame_output.visual_features
        if interleaved_tokens.shape[0] != int(expected_frame_tokens.sum().item()):
            raise RuntimeError(
                "wrapped DINOFrameAdapterCrossV2 returned unexpected token count"
            )

        cls_tokens = interleaved_tokens[0::2]
        patch_tokens = interleaved_tokens[1::2]
        cls_tokens, compressed_frame_length = self.cls_temporal_shuffle(
            cls_tokens, frame_length
        )
        patch_tokens, patch_frame_length = self.patch_temporal_shuffle(
            patch_tokens, frame_length
        )
        if not torch.equal(compressed_frame_length, patch_frame_length):
            raise RuntimeError("CLS and pooled-patch temporal lengths diverged")

        if self.use_short_temporal_conv:
            if (
                self.cls_short_temporal_conv is None
                or self.patch_short_temporal_conv is None
            ):
                raise RuntimeError("short temporal convolution modules are missing")
            cls_tokens = self.cls_short_temporal_conv(
                cls_tokens, compressed_frame_length
            )
            patch_tokens = self.patch_short_temporal_conv(
                patch_tokens, compressed_frame_length
            )
        visual_features = torch.stack((cls_tokens, patch_tokens), dim=1).flatten(0, 1)
        visual_length = compressed_frame_length * 2
        if visual_features.shape[0] != int(visual_length.sum().item()):
            raise RuntimeError(
                "compressed visual token count does not match visual_length"
            )

        position_ids = torch.cat(
            [
                torch.arange(length, device=visual_features.device).repeat_interleave(2)
                for length in compressed_frame_length
            ]
        )
        extras = dict(frame_output.extras or {})
        if self.use_short_temporal_conv:
            extras.update(
                {
                    "cls_short_temporal_gate": torch.sigmoid(
                        self.cls_short_temporal_conv.residual_gate
                    ),
                    "patch_short_temporal_gate": torch.sigmoid(
                        self.patch_short_temporal_conv.residual_gate
                    ),
                }
            )

        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=visual_length,
            position_ids=position_ids,
            extras=extras,
        )


if __name__ == "__main__":
    import torch
    from torch import Tensor

    # Original adapter test
    B, N, D_PATCH, D_CLS = 8, 16, 768, 1024
    cls_token = torch.randn(B, D_CLS).cuda()
    patch_features = torch.randn(B, N, D_PATCH).cuda()
    visual_length = torch.tensor([2, 6]).cuda()

    visual_backbone_output = VisualBackboneOutput(
        visual_features=patch_features,
        pooled_visual_features=cls_token,
        visual_length=visual_length,
    )

    adapter = DINOFrameAdapterCrossV2Shuffle(
        input_dim=D_PATCH,
        cls_input_dim=D_CLS,
        output_dim=512,
    ).cuda()
    adapter.eval()
    with torch.no_grad():
        output = adapter(visual_backbone_output)
        print("Output shape:", output.visual_features.shape)
        print("Patch weights shape:", output.extras["patch_weights"].shape)
        print("Visual length:", output.visual_length)
