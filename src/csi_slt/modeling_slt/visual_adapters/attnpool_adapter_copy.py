"""Attention-selected patch pooling adapter.

Data flow::

    PATCH PATH                              ATTENTION PATH

    patch features [F,P,D]                  CLS attention [F,H,P]
             |                                        |
             v                                        v
    +--------------------------+           +-----------------------+
    | x + Wsp(GELU(DWConv2d(   |           | average over heads    |
    |         LN(x))))         |           | zero four corners     |
    |                          |           | spatial 3x3 smoothing |
    | optionally, then         |           | zero corners again    |
    | x + Wtp(GELU(DWConv1d(   |           | normalize + Top-K     |
    |         LN(x))))         |           |                       |
    +------------+-------------+           +-----------+-----------+
                 |                                     |
                 v                                     v
      contextual patches [F,P,D]              mask [F,P] (bool)
                 |                                     |
                 +------------------+------------------+
                                    |
                                    v
                        mean over the selected
                          patches only [F,D]
                                    |
                                    v
                     boundary-safe temporal mean
                                    |
                                    v
                        LayerNorm + projection
                                    |
                                    v
                        visual tokens [L,D_out]

Both context blocks are pre-norm residuals whose output projection is
zero-initialized, so the whole patch path starts as an exact identity while
every parameter still receives gradient from the first step. All patches
participate in those convolutions; the Top-K mask removes patches only from
the spatial mean. F is the sum of all frame counts in the packed video batch.
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


class NextFramePatchFusion(nn.Module):
    """Align every patch with the next frame and fuse its temporal delta.

    For each current patch, cosine similarities select a soft correspondence
    over patches in the next frame. The aligned difference is transformed and
    added back through a learnable residual gate. Packed-video boundaries are
    respected: the final frame of each video receives an exact zero residual.

    The branch output is normalized before the gate.  ``sigmoid(g) * f(x)``
    alone is not identifiable -- ``f`` can grow its own weights to offset a
    small gate, so the gate value says nothing about how much motion the model
    uses.  Normalizing first pins the branch scale, which makes
    ``motion_weight`` a readable diagnostic that is comparable across runs.

    Current content, matched feature change and spatial displacement are kept
    as distinct inputs. Each is independently projected into a configurable
    fusion space, where the three representations are scale-stably combined::

        h = (W_content LN(x) + W_delta LN(delta) + W_disp displacement) / sqrt(3)
        update = W_out GELU(h)

    The independent projections prevent the two displacement coordinates from
    being numerically drowned by the D-dimensional content and delta streams.
    Their hidden representations are added rather than concatenated to keep
    the output projection compact for small datasets. The update remains a
    gated residual, so the original backbone patch always has a stable path.
    """

    def __init__(
        self,
        hidden_dim: int,
        fusion_hidden_dim: int | None = None,
        temperature: float = 0.1,
        matching_top_k: int = 1,
        gate_init: float = -2.0,
        spatial_window_radius: int | None = 3,
        grid_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if fusion_hidden_dim is not None and (
            isinstance(fusion_hidden_dim, bool)
            or not isinstance(fusion_hidden_dim, int)
            or fusion_hidden_dim <= 0
        ):
            raise ValueError("fusion_hidden_dim must be a positive integer")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if (
            isinstance(matching_top_k, bool)
            or not isinstance(matching_top_k, int)
            or matching_top_k <= 0
        ):
            raise ValueError("matching_top_k must be a positive integer")
        if spatial_window_radius is not None and (
            isinstance(spatial_window_radius, bool)
            or not isinstance(spatial_window_radius, int)
            or spatial_window_radius < 0
        ):
            raise ValueError("spatial_window_radius must be a non-negative integer")
        if grid_size is not None and (
            not isinstance(grid_size, tuple)
            or len(grid_size) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in grid_size
            )
        ):
            raise ValueError("grid_size must be a tuple of two positive integers")

        fusion_hidden_dim = fusion_hidden_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.temperature = temperature
        self.matching_top_k = matching_top_k
        self.spatial_window_radius = spatial_window_radius
        self.grid_size = grid_size
        self.content_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, fusion_hidden_dim),
        )
        self.delta_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, fusion_hidden_dim),
        )
        self.displacement_projection = nn.Linear(2, fusion_hidden_dim, bias=False)
        self.fusion_output = nn.Sequential(
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )
        # Fan-in init, not zeros. Zeros never blocked this branch's gradient --
        # a Linear's weight gradient depends on its input, not on its weight --
        # but it did leave it two orders of magnitude behind the others, and it
        # could not close that gap under a shared learning rate: after 42k steps
        # of one run it still contributed 0.9% of the fusion hidden vector
        # against content's 69% and delta's 74%. The handicap is structural, not
        # temporary: displacement enters through 2 dimensions of roughly unit
        # scale while content and delta enter through 1152 LayerNormed ones, so
        # starting it at zero on top of that is a second penalty for the same
        # thing.
        nn.init.kaiming_uniform_(self.displacement_projection.weight, a=math.sqrt(5))
        # Keep this one-dimensional for FSDP2 compatibility.
        self.fusion_gate = nn.Parameter(torch.tensor([float(gate_init)]))
        # Built on first use: these depend on the runtime patch count, and are
        # derived rather than learned, so they stay out of the state dict.
        self.register_buffer("_neighbourhood_mask", None, persistent=False)
        self.register_buffer("_patch_positions", None, persistent=False)
        self._displacement_scale = 1.0
        # Mean absolute displacement of the last forward, kept as a tensor so
        # reading it never synchronizes the device in the training loop.
        self.register_buffer("_last_displacement", None, persistent=False)

    @property
    def motion_weight(self) -> float:
        """Current gate value: how much aligned motion rides on the patches."""
        return float(torch.sigmoid(self.fusion_gate).item())

    @property
    def mean_displacement(self) -> float | None:
        """Mean |displacement| in patch units over the last forward pass.

        This measures coordinate movement, not matching confidence. In
        particular, an asymmetric boundary window can produce displacement
        even for an ambiguous distribution.
        """
        if self._last_displacement is None:
            return None
        return float(self._last_displacement.item())

    def forward(self, patch_features: Tensor, visual_length: Tensor) -> Tensor:
        """Fuse packed patch features shaped ``[sum(T), P, D]``."""
        self._validate_inputs(patch_features, visual_length)
        source_indices, has_next = self._next_frame_shift(patch_features, visual_length)
        # L2 normalization is row-wise, so normalize(x[idx]) == normalize(x)[idx].
        # Normalizing once and gathering avoids running it twice over the same
        # rows, which is what normalizing the already-gathered tensor did.
        normalized = F.normalize(patch_features, dim=-1)
        aligned_next, displacement = self.similarity_aggregate(
            normalized, normalized[source_indices], patch_features[source_indices]
        )
        valid = has_next[:, None, None]
        delta = (aligned_next - patch_features) * valid.to(dtype=patch_features.dtype)
        content_hidden = self.content_encoder(patch_features)
        delta_hidden = self.delta_encoder(delta)
        displacement_hidden = self.displacement_projection(
            displacement.to(dtype=delta_hidden.dtype)
        )
        fusion_hidden = (
            content_hidden + delta_hidden + displacement_hidden
        ) / math.sqrt(3.0)
        fusion_update = self.fusion_output(fusion_hidden)
        # Mask last: the projections and normalization carry biases, so a final
        # frame would otherwise pick up a residual out of thin air rather than
        # staying exactly unchanged.
        fusion_update = fusion_update * valid.to(dtype=fusion_update.dtype)
        return patch_features + torch.sigmoid(self.fusion_gate) * fusion_update

    def similarity_aggregate(
        self, base_norm: Tensor, shifted_norm: Tensor, shifted: Tensor
    ) -> Tensor:
        """Soft-align next-frame patches to current patches by cosine similarity.

        ``base_norm`` and ``shifted_norm`` are the already L2-normalized
        descriptors that decide the correspondence; ``shifted`` holds the raw
        next-frame features the weights are applied to.

        Returns the aligned next-frame features and, in patch units scaled to
        roughly [-1, 1], the expected displacement of every patch.
        """
        # The contraction stays in the input dtype -- it is the expensive part
        # and is well conditioned -- but the softmax does not.  Dividing cosine
        # similarities by a temperature of 0.1 spreads the logits over roughly
        # [-10, 10], and bf16 carries about three decimal digits, so a peaked
        # softmax there loses real resolution between close candidates.
        similarity = torch.einsum("bnd,btd->bnt", base_norm, shifted_norm).float()
        if self.spatial_window_radius is not None:
            spatial_mask = self._spatial_neighbourhood_mask(
                base_norm.shape[1], base_norm.device
            )
            similarity = similarity.masked_fill(~spatial_mask, float("-inf"))
        # Retain only the strongest correspondences before normalization. This
        # removes the long low-probability tail that otherwise turns aligned
        # features and expected displacement into a local spatial average.
        matching_top_k = min(self.matching_top_k, similarity.shape[-1])
        top_values, top_indices = similarity.topk(matching_top_k, dim=-1)
        top_weights = F.softmax(top_values / self.temperature, dim=-1)
        weights = (
            torch.zeros_like(similarity)
            .scatter(-1, top_indices, top_weights)
            .to(dtype=shifted.dtype)
        )
        aligned = torch.einsum("bnt,btd->bnd", weights, shifted)

        positions, scale = self._patch_grid_positions(
            base_norm.shape[1], base_norm.device
        )
        # sum_t w[n,t] * pos[t] - pos[n]: the weights sum to one, so the
        # expected displacement is just the expected coordinate minus the
        # current one, which is a single matmul against a constant table.
        displacement = (weights.float() @ positions - positions) / scale
        self._last_displacement = displacement.detach().abs().mean()
        return aligned, displacement

    def _resolve_grid(self, num_patches: int) -> tuple[int, int]:
        """Grid shape of a frame's patches, from grid_size or a square guess."""
        if self.grid_size is not None:
            grid_height, grid_width = self.grid_size
            if grid_height * grid_width != num_patches:
                raise ValueError(
                    f"grid_size {self.grid_size} contains "
                    f"{grid_height * grid_width} cells, but features contain "
                    f"{num_patches} patches"
                )
            return grid_height, grid_width
        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(
                f"cannot infer a square grid from {num_patches} patches; "
                "provide grid_size explicitly"
            )
        return side, side

    def _patch_grid_positions(
        self, num_patches: int, device: torch.device
    ) -> tuple[Tensor, float]:
        """Row/column coordinate of every patch, plus the displacement scale.

        The scale comes from the window radius, which bounds how far a patch
        can be matched, so the embedding's input stays near [-1, 1] whatever
        the window is set to. It is derived from Python ints rather than read
        off the tensor, which would synchronize the device every forward.
        """
        cached = self._patch_positions
        if cached is None or cached.shape[0] != num_patches or cached.device != device:
            grid_height, grid_width = self._resolve_grid(num_patches)
            indices = torch.arange(num_patches, device=device)
            rows = torch.div(indices, grid_width, rounding_mode="floor")
            columns = indices.remainder(grid_width)
            self._patch_positions = torch.stack((rows, columns), dim=-1).float()
            self._displacement_scale = (
                float(
                    self.spatial_window_radius
                    if self.spatial_window_radius
                    else max(grid_height, grid_width) - 1
                )
                or 1.0
            )
            cached = self._patch_positions
        return cached, self._displacement_scale

    def _spatial_neighbourhood_mask(
        self, num_patches: int, device: torch.device
    ) -> Tensor:
        # Constant for a given grid, so it is built once instead of rebuilt
        # from six small kernels on every forward.
        cached = self._neighbourhood_mask
        if (
            cached is not None
            and cached.shape[0] == num_patches
            and cached.device == device
        ):
            return cached
        grid_height, grid_width = self._resolve_grid(num_patches)
        indices = torch.arange(num_patches, device=device)
        rows = torch.div(indices, grid_width, rounding_mode="floor")
        columns = indices.remainder(grid_width)
        row_distance = (rows[:, None] - rows[None, :]).abs()
        column_distance = (columns[:, None] - columns[None, :]).abs()
        mask = torch.maximum(row_distance, column_distance).le(
            self.spatial_window_radius
        )
        self._neighbourhood_mask = mask
        return mask

    @staticmethod
    def _next_frame_shift(
        patch_features: Tensor, visual_length: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the next-frame index of every frame, and which frames have one.

        The last frame of each packed video points at itself; ``has_next`` marks
        it so the caller can zero its residual.
        """
        total_frames = patch_features.shape[0]
        device = patch_features.device
        ends = torch.cumsum(visual_length.to(device=device), dim=0) - 1
        source_indices = torch.arange(total_frames, device=device)
        has_next = torch.ones(total_frames, dtype=torch.bool, device=device)
        has_next[ends] = False
        source_indices = source_indices + has_next.to(dtype=source_indices.dtype)
        return source_indices, has_next

    def _validate_inputs(self, patch_features: Tensor, visual_length: Tensor) -> None:
        if patch_features.ndim != 3:
            raise ValueError("patch_features must have shape [sum(T), P, D]")
        if patch_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"patch feature dimension must be {self.hidden_dim}, got "
                f"{patch_features.shape[-1]}"
            )
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if visual_length.is_floating_point() or visual_length.is_complex():
            raise TypeError("visual_length must use an integer dtype")
        if bool((visual_length <= 0).any()):
            raise ValueError("all visual lengths must be positive")
        if int(visual_length.sum().item()) != patch_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed frames"
            )


class SpatiotemporalSeparableConv(nn.Module):
    """Enrich packed patch features with spatial and temporal context.

    Two independent pre-norm residual blocks rather than one residual wrapped
    around both convolutions::

        x = x + spatial_projection(GELU(spatial_conv(spatial_norm(x))))
        x = x + temporal_projection(GELU(temporal_conv(temporal_norm(x))))

    Each branch normalizes its own input, so neither activation ever sees an
    unnormalized distribution, and each output projection is zero-initialized
    so both blocks start as exact identities while still receiving gradient
    from the first step. The temporal block is optional; when it is disabled
    the module is a single spatial residual block.
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        use_temporal_conv: bool = True,
        grid_size: tuple[int, int] | None = None,
        debug_validation: bool = False,
    ) -> None:
        super().__init__()
        self._validate_positive_integer("hidden_dim", hidden_dim)
        self._validate_odd_kernel("spatial_kernel_size", spatial_kernel_size)
        self._validate_odd_kernel("temporal_kernel_size", temporal_kernel_size)
        if not isinstance(use_temporal_conv, bool):
            raise TypeError("use_temporal_conv must be a boolean")
        if not isinstance(debug_validation, bool):
            raise TypeError("debug_validation must be a boolean")
        self.grid_size = self._validate_grid_size(grid_size)
        self.hidden_dim = hidden_dim
        self.use_temporal_conv = use_temporal_conv
        self.debug_validation = debug_validation

        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            groups=hidden_dim,
            bias=False,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim) if use_temporal_conv else None
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
        self.spatial_projection = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_projection = (
            nn.Linear(hidden_dim, hidden_dim) if use_temporal_conv else None
        )
        # Zero-init each branch's output projection instead of gating the whole
        # branch with a scalar. A block still starts as an exact identity, but
        # its projection weight receives gradient from the first step
        # (dL/dW = dL/dout . h^T, which is nonzero at W=0), and the upstream
        # conv and norm start learning one step later. A tanh gate initialized
        # at zero instead multiplies the entire branch by zero, so every
        # parameter behind it gets exactly zero gradient and the branch has to
        # bootstrap through a single scalar -- measured at tanh=0.022 after
        # 12k steps, i.e. effectively never.
        for projection in (self.spatial_projection, self.temporal_projection):
            if projection is not None:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def forward(self, patch_features: Tensor, visual_length: Tensor) -> Tensor:
        """Process ``[sum(T), P, D]`` without crossing packed-video boundaries."""
        self._validate_inputs(patch_features, visual_length)
        grid_height, grid_width = self._resolve_grid_size(patch_features.shape[1])

        features = patch_features + self._spatial_block(
            self.spatial_norm(patch_features), grid_height, grid_width
        )
        if self.temporal_conv is not None:
            features = features + self._temporal_block(
                self.temporal_norm(features), visual_length
            )
        return features

    def _spatial_block(
        self, features: Tensor, grid_height: int, grid_width: int
    ) -> Tensor:
        """Depthwise 2D convolution over the patch grid of each frame."""
        frame_count, patch_count, hidden_dim = features.shape
        features = features.reshape(
            frame_count, grid_height, grid_width, hidden_dim
        ).permute(0, 3, 1, 2)
        features = self.activation(self.spatial_conv(features))
        features = features.permute(0, 2, 3, 1).reshape(
            frame_count, patch_count, hidden_dim
        )
        return self.spatial_projection(features)

    def _temporal_block(self, features: Tensor, visual_length: Tensor) -> Tensor:
        """Depthwise 1D convolution along time, per patch position.

        Padding each video separately is what keeps the kernel from reading
        across a packed-video boundary.
        """
        patch_count, hidden_dim = features.shape[1:]
        padded, valid_mask = packed_to_padded(features, visual_length)
        batch_size, max_length = padded.shape[:2]
        temporal_features = padded.permute(0, 2, 3, 1).reshape(
            batch_size * patch_count, hidden_dim, max_length
        )
        temporal_features = self.activation(self.temporal_conv(temporal_features))
        padded = temporal_features.reshape(
            batch_size, patch_count, hidden_dim, max_length
        ).permute(0, 3, 1, 2)
        features, _ = padded_to_packed(padded, valid_mask)
        return self.temporal_projection(features)

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

    def _validate_inputs(self, patch_features: Tensor, visual_length: Tensor) -> None:
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
    """Normalized patch scores and their binary Top-K selection mask.

    Only ``mask`` reaches the pooled feature: the adapter takes a uniform mean
    over the selected patches and never weights them by ``scores``. ``scores``
    is what Top-K ranks, and is returned so callers can visualize the ranking.
    """

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
            if len(grid_size) != 2 or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in grid_size
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


class AttnPoolAdapter(nn.Module):
    """Select patches with CLS attention, mean-pool them, then pool over time."""

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
        patch_fusion_hidden_dim: int | None = None,
        patch_fusion_temperature: float = 0.1,
        patch_fusion_matching_top_k: int = 1,
        patch_fusion_gate_init: float = -2.0,
        patch_fusion_window_radius: int | None = 3,
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
        self.next_frame_patch_fusion = NextFramePatchFusion(
            hidden_dim=input_dim,
            fusion_hidden_dim=patch_fusion_hidden_dim,
            temperature=patch_fusion_temperature,
            matching_top_k=patch_fusion_matching_top_k,
            gate_init=patch_fusion_gate_init,
            spatial_window_radius=patch_fusion_window_radius,
            grid_size=patch_grid_size,
        )
        self.patch_context = SpatiotemporalSeparableConv(
            hidden_dim=input_dim,
            spatial_kernel_size=spatial_conv_kernel_size,
            temporal_kernel_size=temporal_conv_kernel_size,
            use_temporal_conv=use_temporal_conv,
            grid_size=patch_grid_size,
            debug_validation=debug_validation,
        )
        self.attention_selector = ClsAttentionTopKSelector(
            top_k=top_k,
            spatial_smooth_kernel=attention_smooth_kernel_size,
            grid_size=patch_grid_size,
            debug_validation=debug_validation,
        )

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
        fused_patches = self.next_frame_patch_fusion(patch_features, visual_length)
        contextualized_patches = self.patch_context(fused_patches, visual_length)
        selection = self.attention_selector(extras["attention_maps"])
        # Uniform mean over the selected patches. The attention scores decide
        # *which* patches survive; they never weight the surviving ones.
        feature_mask = selection.mask.unsqueeze(-1).to(
            dtype=contextualized_patches.dtype
        )
        selected_counts = selection.mask.sum(dim=1, keepdim=True).to(
            dtype=contextualized_patches.dtype
        )
        frame_features = (contextualized_patches * feature_mask).sum(
            dim=1
        ) / selected_counts

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
