"""Attention-guided patch filtering adapter.

Data flow::

    PATCH PATH                              ATTENTION PATH

    patch features [F,P,D]                  CLS attention [F,H,P]
             |                                        |
             v                                        v
    +----------------------+               +-----------------------+
    | LayerNorm            |               | average over heads    |
    | spatial DW Conv2d    |               | zero four corners     |
    | optional temporal DW |               | spatial 3x3 smoothing |
    | Conv1d               |               | zero corners again    |
    | Linear + tanh gate   |               | normalize + Top-K     |
    +----------+-----------+               +-----------+-----------+
               |                                       |
               v                                       v
    contextual patches [F,P,D]             scores [F,P] + mask [F,P]
               |                                       |
               +-------------------+-------------------+
                                   |
                    HARD MASK APPLIES ONLY BELOW
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
          masked uniform mean          masked attention-weighted
               pool [F,D]                    pool [F,D]
                    |                             |
                    +------ sigmoid pool gate ----+
                                   |
                                   v
                    boundary-safe temporal mean
                                   |
                                   v
                       LayerNorm + projection
                                   |
                                   v
                       visual tokens [L,D_out]

All patches participate in the spatial-temporal convolution. The Top-K mask
removes patches only from the two final spatial pooling branches. F is the sum
of all frame counts in the packed video batch.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    packed_to_padded,
    padded_to_packed,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput


class SpatiotemporalSeparableConv(nn.Module):
    """Enrich packed patch features with gated spatial-then-temporal context."""

    def __init__(
        self,
        hidden_dim: int,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        use_temporal_conv: bool = True,
        grid_size: tuple[int, int] | None = None,
        residual_gate_init: float = 0.0,
        debug_validation: bool = False,
    ) -> None:
        super().__init__()
        self._validate_positive_integer("hidden_dim", hidden_dim)
        self._validate_odd_kernel("spatial_kernel_size", spatial_kernel_size)
        self._validate_odd_kernel("temporal_kernel_size", temporal_kernel_size)
        if not isinstance(use_temporal_conv, bool):
            raise TypeError("use_temporal_conv must be a boolean")
        if isinstance(residual_gate_init, bool) or not isinstance(
            residual_gate_init, (int, float)
        ):
            raise TypeError("residual_gate_init must be a real number")
        if not math.isfinite(float(residual_gate_init)):
            raise ValueError("residual_gate_init must be finite")
        if not isinstance(debug_validation, bool):
            raise TypeError("debug_validation must be a boolean")
        self.grid_size = self._validate_grid_size(grid_size)
        self.hidden_dim = hidden_dim
        self.use_temporal_conv = use_temporal_conv
        self.debug_validation = debug_validation

        self.norm = nn.LayerNorm(hidden_dim)
        self.spatial_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            groups=hidden_dim,
            bias=False,
        )
        self.temporal_conv = (
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=temporal_kernel_size,
                padding=temporal_kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            )
            if use_temporal_conv
            else None
        )
        # forward() feeds spatial_conv a permuted NHWC view, which is already
        # channels_last.  Matching the weight layout to it keeps cuDNN on the
        # fast depthwise kernel *and* makes the weight gradient come back with
        # the parameter's strides, which is what DDP's gradient layout contract
        # requires; otherwise DDP falls back to a strided bucket copy and warns.
        # Forcing the activations contiguous instead would satisfy DDP too, but
        # costs a full copy of the [F, D, H, W] tensor and measured 2x slower.
        self.spatial_conv = self.spatial_conv.to(memory_format=torch.channels_last)
        self.activation = nn.GELU()
        self.channel_projection = nn.Linear(hidden_dim, hidden_dim)
        self.residual_gate = nn.Parameter(
            torch.full((1,), float(residual_gate_init))
        )

    def forward(self, patch_features: Tensor, visual_length: Tensor) -> Tensor:
        """Process ``[sum(T), P, D]`` without crossing packed-video boundaries."""
        self._validate_inputs(patch_features, visual_length)
        frame_count, patch_count, hidden_dim = patch_features.shape
        grid_height, grid_width = self._resolve_grid_size(patch_count)

        residual = patch_features
        features = self.norm(patch_features)
        features = features.reshape(
            frame_count, grid_height, grid_width, hidden_dim
        ).permute(0, 3, 1, 2)
        features = self.activation(self.spatial_conv(features))

        features = features.permute(0, 2, 3, 1)
        if self.temporal_conv is not None:
            padded, valid_mask = packed_to_padded(features, visual_length)
            batch_size, max_length = padded.shape[:2]
            temporal_features = padded.permute(0, 2, 3, 4, 1).reshape(
                batch_size * patch_count, hidden_dim, max_length
            )
            temporal_features = self.activation(
                self.temporal_conv(temporal_features)
            )
            padded = temporal_features.reshape(
                batch_size, grid_height, grid_width, hidden_dim, max_length
            ).permute(0, 4, 1, 2, 3)
            features, _ = padded_to_packed(padded, valid_mask)

        features = features.reshape(frame_count, patch_count, hidden_dim)
        features = self.channel_projection(features)
        residual_weight = torch.tanh(self.residual_gate).to(dtype=features.dtype)
        return residual + residual_weight * features

    def _resolve_grid_size(self, patch_count: int) -> tuple[int, int]:
        if self.grid_size is not None:
            grid_height, grid_width = self.grid_size
            if grid_height * grid_width != patch_count:
                raise ValueError(
                    f"grid_size {self.grid_size} contains "
                    f"{grid_height * grid_width} cells, but features contain "
                    f"{patch_count} patches"
                )
            return grid_height, grid_width

        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(
                f"cannot infer a square grid from {patch_count} patches; "
                "provide grid_size explicitly"
            )
        return side, side

    def _validate_inputs(
        self, patch_features: Tensor, visual_length: Tensor
    ) -> None:
        if not isinstance(patch_features, Tensor) or patch_features.ndim != 3:
            raise ValueError("patch_features must have shape [sum(T), P, D]")
        if patch_features.shape[0] == 0 or patch_features.shape[1] == 0:
            raise ValueError("patch_features must contain frames and patches")
        if patch_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"patch feature dimension must be {self.hidden_dim}, got "
                f"{patch_features.shape[-1]}"
            )
        if not isinstance(visual_length, Tensor):
            raise TypeError("visual_length must be a torch.Tensor")
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if visual_length.is_floating_point() or visual_length.is_complex():
            raise TypeError("visual_length must use an integer dtype")
        # Value checks read tensor contents and therefore synchronize the
        # device; they stay behind debug_validation so training does not stall.
        if not self.debug_validation:
            return
        if bool((visual_length <= 0).any()):
            raise ValueError("all visual lengths must be positive")
        if int(visual_length.sum().item()) != patch_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed frames"
            )

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def _validate_odd_kernel(cls, name: str, value: int) -> None:
        cls._validate_positive_integer(name, value)
        if value % 2 == 0:
            raise ValueError(f"{name} must be odd")

    @staticmethod
    def _validate_grid_size(
        grid_size: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if grid_size is None:
            return None
        if (
            not isinstance(grid_size, tuple)
            or len(grid_size) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in grid_size
            )
        ):
            raise ValueError("grid_size must be a tuple of two positive integers")
        return grid_size


class AttentionSelectionOutput(NamedTuple):
    """Normalized patch scores and their binary Top-K selection mask."""

    scores: Tensor  # [F, P], sums to one over P for every frame
    mask: Tensor  # [F, P], bool


class ClsAttentionTopKSelector(nn.Module):
    """Turn per-frame CLS attention into smoothed scores and a Top-K mask."""

    def __init__(
        self,
        top_k: int,
        spatial_smooth_kernel: int = 3,
        grid_size: tuple[int, int] | None = None,
        debug_validation: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k == 0:
            raise ValueError("top_k must be -1 or a positive integer")
        if top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")
        if (
            isinstance(spatial_smooth_kernel, bool)
            or not isinstance(spatial_smooth_kernel, int)
            or spatial_smooth_kernel < 1
            or spatial_smooth_kernel % 2 == 0
        ):
            raise ValueError("spatial_smooth_kernel must be a positive odd integer")
        if grid_size is not None:
            if (
                len(grid_size) != 2
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in grid_size
                )
            ):
                raise ValueError("grid_size must contain two positive integers")
        if not isinstance(debug_validation, bool):
            raise TypeError("debug_validation must be a boolean")

        self.top_k = top_k
        self.spatial_smooth_kernel = spatial_smooth_kernel
        self.grid_size = grid_size
        self.debug_validation = debug_validation

    def forward(self, cls_attention: Tensor) -> AttentionSelectionOutput:
        """Score patches from ``[F, H, P]`` or already averaged ``[F, P]`` input."""
        if not isinstance(cls_attention, Tensor):
            raise TypeError("cls_attention must be a torch.Tensor")
        if cls_attention.ndim == 3:
            scores = cls_attention.float().mean(dim=1)
        elif cls_attention.ndim == 2:
            scores = cls_attention.float()
        else:
            raise ValueError(
                "cls_attention must have shape [F, heads, patches] or [F, patches], "
                f"got {tuple(cls_attention.shape)}"
            )
        if scores.shape[0] == 0 or scores.shape[1] == 0:
            raise ValueError("cls_attention must contain frames and patch tokens")
        # Reading tensor contents synchronizes the device, so the value checks
        # here and below only run under debug_validation.
        if self.debug_validation:
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("cls_attention must contain only finite values")
            if bool((scores < 0).any()):
                raise ValueError("cls_attention weights must be non-negative")

        frame_count, patch_count = scores.shape
        grid_height, grid_width = self._resolve_grid_size(patch_count)
        eligible_grid = torch.ones(
            (grid_height, grid_width), dtype=torch.bool, device=scores.device
        )
        eligible_grid[0, 0] = False
        eligible_grid[0, -1] = False
        eligible_grid[-1, 0] = False
        eligible_grid[-1, -1] = False
        eligible_mask = eligible_grid.flatten().expand(frame_count, -1)
        # Counted in Python rather than with eligible_grid.sum().item(), which
        # would synchronize the device on every forward.
        eligible_count = grid_height * grid_width - 4
        if self.top_k > eligible_count:
            raise ValueError(
                f"top_k={self.top_k} exceeds the {eligible_count} non-corner patches"
            )

        scores = scores.reshape(frame_count, 1, grid_height, grid_width)
        scores = self._zero_corners(scores)
        scores = self._smooth(scores)
        scores = self._zero_corners(scores)
        scores = scores.flatten(start_dim=1)

        score_sums = scores.sum(dim=1, keepdim=True)
        if self.debug_validation and bool((score_sums <= 0).any()):
            raise ValueError("corner masking and smoothing removed all attention mass")
        scores = scores / score_sums

        if self.top_k == -1:
            mask = eligible_mask.clone()
        else:
            selection_scores = scores.masked_fill(~eligible_mask, float("-inf"))
            top_indices = selection_scores.topk(self.top_k, dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
        return AttentionSelectionOutput(scores=scores, mask=mask)

    def _resolve_grid_size(self, patch_count: int) -> tuple[int, int]:
        if self.grid_size is not None:
            grid_height, grid_width = self.grid_size
            if grid_height * grid_width != patch_count:
                raise ValueError(
                    f"grid_size {self.grid_size} contains "
                    f"{grid_height * grid_width} cells, but attention has "
                    f"{patch_count} patches"
                )
            return grid_height, grid_width

        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(
                f"cannot infer a square grid from {patch_count} patches; "
                "provide grid_size explicitly"
            )
        return side, side

    def _smooth(self, scores: Tensor) -> Tensor:
        if self.spatial_smooth_kernel == 1:
            return scores
        padding = self.spatial_smooth_kernel // 2
        smoothed_sum = F.avg_pool2d(
            scores,
            kernel_size=self.spatial_smooth_kernel,
            stride=1,
            padding=padding,
            divisor_override=1,
        )
        valid_counts = F.avg_pool2d(
            torch.ones_like(scores),
            kernel_size=self.spatial_smooth_kernel,
            stride=1,
            padding=padding,
            divisor_override=1,
        )
        return smoothed_sum / valid_counts

    @staticmethod
    def _zero_corners(scores: Tensor) -> Tensor:
        scores = scores.clone()
        scores[..., 0, 0] = 0
        scores[..., 0, -1] = 0
        scores[..., -1, 0] = 0
        scores[..., -1, -1] = 0
        return scores


class MaskedAttentionPool(nn.Module):
    """Pool one feature per frame using renormalized selected attention scores."""

    def __init__(self, debug_validation: bool = False) -> None:
        super().__init__()
        if not isinstance(debug_validation, bool):
            raise TypeError("debug_validation must be a boolean")
        self.debug_validation = debug_validation

    def forward(
        self,
        patch_features: Tensor,
        attention_scores: Tensor,
        selection_mask: Tensor,
    ) -> Tensor:
        """Pool [F, P, D] features into [F, D] frame features."""
        self._validate_inputs(patch_features, attention_scores, selection_mask)

        # Normalize in float32 for stability. Masked patches receive exactly
        # zero weight before the weighted feature reduction.
        selected_scores = attention_scores.float().masked_fill(~selection_mask, 0)
        selected_mass = selected_scores.sum(dim=1, keepdim=True)
        if self.debug_validation and bool((selected_mass <= 0).any()):
            invalid_frames = (selected_mass.squeeze(1) <= 0).nonzero().flatten()
            raise ValueError(
                "every frame must retain positive attention mass; invalid frame "
                f"indices: {invalid_frames.tolist()}"
            )
        selected_weights = selected_scores / selected_mass
        selected_weights = selected_weights.to(dtype=patch_features.dtype)
        return (patch_features * selected_weights.unsqueeze(-1)).sum(dim=1)

    def _validate_inputs(
        self,
        patch_features: Tensor,
        attention_scores: Tensor,
        selection_mask: Tensor,
    ) -> None:
        if not isinstance(patch_features, Tensor) or patch_features.ndim != 3:
            raise ValueError("patch_features must have shape [F, P, D]")
        expected_shape = patch_features.shape[:2]
        if not isinstance(attention_scores, Tensor) or attention_scores.ndim != 2:
            raise ValueError("attention_scores must have shape [F, P]")
        if tuple(attention_scores.shape) != tuple(expected_shape):
            raise ValueError(
                "attention_scores shape must match patch_features.shape[:2]"
            )
        if not isinstance(selection_mask, Tensor) or selection_mask.ndim != 2:
            raise ValueError("selection_mask must have shape [F, P]")
        if tuple(selection_mask.shape) != tuple(expected_shape):
            raise ValueError(
                "selection_mask shape must match patch_features.shape[:2]"
            )
        if selection_mask.dtype != torch.bool:
            raise TypeError("selection_mask must use torch.bool dtype")
        if attention_scores.device != patch_features.device:
            raise ValueError("attention_scores and patch_features must share a device")
        if selection_mask.device != patch_features.device:
            raise ValueError("selection_mask and patch_features must share a device")
        # Device-synchronizing value checks; see debug_validation above.
        if not self.debug_validation:
            return
        if not bool(torch.isfinite(attention_scores).all()):
            raise ValueError("attention_scores must contain only finite values")
        if bool((attention_scores < 0).any()):
            raise ValueError("attention_scores must be non-negative")
        if not bool(selection_mask.any(dim=1).all()):
            raise ValueError("selection_mask must select at least one patch per frame")


class AttnPoolAdapter(nn.Module):
    """Filter contextualized patches with CLS attention, then pool over time."""

    requires_visual_backbone_attention = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_rank: int | None = None,
        use_layer_norm: bool = True,
        temporal_scale_factor: int = 2,
        top_k: int = 48,
        attention_smooth_kernel_size: int = 3,
        patch_grid_size: tuple[int, int] | None = None,
        spatial_conv_kernel_size: int = 3,
        use_temporal_conv: bool = True,
        temporal_conv_kernel_size: int = 3,
        residual_gate_init: float = 0.0,
        pooling_gate_init: float = -2.0,
        debug_validation: bool = False,
    ) -> None:
        super().__init__()
        self._validate_dimension("input_dim", input_dim)
        self._validate_dimension("output_dim", output_dim)
        if projection_rank is not None:
            self._validate_dimension("projection_rank", projection_rank)
        self._validate_dimension("temporal_scale_factor", temporal_scale_factor)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_rank = (
            output_dim if projection_rank is None else projection_rank
        )
        self.temporal_scale_factor = temporal_scale_factor
        # Off by default: every check it enables reads tensor contents and so
        # forces a device synchronization in the training hot path.
        if not isinstance(debug_validation, bool):
            raise TypeError("debug_validation must be a boolean")
        self.debug_validation = debug_validation
        self.norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        if isinstance(pooling_gate_init, bool) or not isinstance(
            pooling_gate_init, (int, float)
        ):
            raise TypeError("pooling_gate_init must be a real number")
        if not math.isfinite(float(pooling_gate_init)):
            raise ValueError("pooling_gate_init must be finite")
        self.pooling_gate = nn.Parameter(
            torch.full((1,), float(pooling_gate_init))
        )
        self.patch_context = SpatiotemporalSeparableConv(
            hidden_dim=input_dim,
            spatial_kernel_size=spatial_conv_kernel_size,
            temporal_kernel_size=temporal_conv_kernel_size,
            use_temporal_conv=use_temporal_conv,
            grid_size=patch_grid_size,
            residual_gate_init=residual_gate_init,
            debug_validation=debug_validation,
        )
        self.attention_selector = ClsAttentionTopKSelector(
            top_k=top_k,
            spatial_smooth_kernel=attention_smooth_kernel_size,
            grid_size=patch_grid_size,
            debug_validation=debug_validation,
        )
        self.attention_pool = MaskedAttentionPool(debug_validation=debug_validation)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.projection_rank),
            nn.GELU(),
            nn.Linear(self.projection_rank, output_dim),
        )

        self._reset_projection_parameters()
        mark_module_tree_as_initialized(self)

    @staticmethod
    def _validate_dimension(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def _reset_projection_parameters(self) -> None:
        input_projection, output_projection = self.projection[0], self.projection[2]
        for projection in (input_projection, output_projection):
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
            nn.init.zeros_(projection.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        patch_features = visual_backbone_output.visual_features
        visual_length = visual_backbone_output.visual_length
        self._validate_inputs(patch_features, visual_length)

        extras = visual_backbone_output.extras
        if extras is None or extras.get("attention_maps") is None:
            raise ValueError(
                "AttnPoolAdapter requires "
                "visual_backbone_output.extras['attention_maps']"
            )
        contextualized_patches = self.patch_context(
            patch_features, visual_length
        )
        selection = self.attention_selector(extras["attention_maps"])
        feature_mask = selection.mask.unsqueeze(-1).to(
            dtype=contextualized_patches.dtype
        )
        selected_counts = selection.mask.sum(dim=1, keepdim=True).to(
            dtype=contextualized_patches.dtype
        )
        mean_pooled_features = (
            contextualized_patches * feature_mask
        ).sum(dim=1) / selected_counts
        attention_pooled_features = self.attention_pool(
            contextualized_patches,
            selection.scores,
            selection.mask,
        )
        pooling_weight = torch.sigmoid(self.pooling_gate).to(
            dtype=contextualized_patches.dtype
        )
        frame_features = torch.lerp(
            mean_pooled_features,
            attention_pooled_features,
            pooling_weight,
        )

        video_features = torch.split(frame_features, visual_length.tolist(), dim=0)
        pooled_features = torch.cat(
            [
                features.unflatten(0, (-1, self.temporal_scale_factor)).mean(dim=1)
                for features in video_features
            ],
            dim=0,
        )
        pooled_length = visual_length // self.temporal_scale_factor
        visual_features = self.projection(self.norm(pooled_features))

        if permute_video_tokens:
            permutation = random_derangement(
                pooled_length, device=visual_features.device
            )
            visual_features = visual_features[permutation]

        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=pooled_length,
        )

    def _validate_inputs(
        self,
        patch_features: Tensor | None,
        visual_length: Tensor | None,
    ) -> None:
        if patch_features is None:
            raise ValueError("visual_features must contain patch features")
        if patch_features.ndim != 3:
            raise ValueError(
                "visual_features must have shape [sum(T), P, input_dim], got "
                f"{tuple(patch_features.shape)}"
            )
        if patch_features.shape[1] == 0:
            raise ValueError("visual_features must contain at least one patch")
        if patch_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"patch feature dimension must be {self.input_dim}, got "
                f"{patch_features.shape[-1]}"
            )
        if visual_length is None:
            raise ValueError("visual_length must be provided")
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if visual_length.is_floating_point() or visual_length.is_complex():
            raise TypeError("visual_length must use an integer dtype")
        # Device-synchronizing value checks; see debug_validation above.
        if not self.debug_validation:
            return
        if bool((visual_length <= 0).any()):
            raise ValueError("all entries in visual_length must be positive")
        if int(visual_length.sum().item()) != patch_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed frames"
            )
        if bool(visual_length.remainder(self.temporal_scale_factor).ne(0).any()):
            raise ValueError(
                "every visual length must be divisible by temporal_scale_factor "
                f"{self.temporal_scale_factor}"
            )
