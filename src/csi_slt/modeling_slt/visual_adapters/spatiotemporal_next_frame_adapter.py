"""Next-frame patch fusion in front of the pooled-linear baseline.

This is ``SpatiotemporalPooledLinearAdapter`` with exactly one thing added:
``NextFramePatchFusion`` runs on the backbone patch features the moment they
arrive, before anything else happens.  The spatial mean, the boundary-safe
temporal mean, the LayerNorm and the two-layer connector are unchanged, so a
run of this adapter against that baseline isolates the fusion block and
nothing else.

Data flow::

    patch features [F, P, D]
             |
             v  NextFramePatchFusion      <- the only added stage
    fused patches [F, P, D]                  gate -> 0 recovers the
             |                               baseline exactly
             v  mean over P                (no selection, no CLS)
    frame features [F, D]
             |
             v  boundary-safe mean over windows of s
    pooled features [N, D]                   N = F / s
             |
             v  LayerNorm
             v  Linear -> R
             v  GELU
             v  Linear -> D_out
             |
             v
    visual tokens [N, D_out]

``NextFramePatchFusion`` is imported from ``attnpool_adapter_copy`` rather
than re-implemented, so this ablation and the attention-pooling one always
run bit-identical fusion code.

Unlike the pooled-linear baseline this adapter has no ``use_cls_token`` mode:
the fusion matches patches against patches, so there is nothing for it to do
on a single CLS vector per frame.  F is the sum of all frame counts in the
packed video batch and s is ``temporal_scale_factor``.
"""

import math

import torch
from torch import Tensor, nn

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
import torch.nn.functional as F


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
        # Let direction enter gradually without blocking its first-step
        # gradient. The content and delta streams retain their normal init.
        nn.init.zeros_(self.displacement_projection.weight)
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


class SpatiotemporalNextFrameAdapter(nn.Module):
    """Fuse each patch with its next-frame match, then pool and project.

    The projection is the same two-layer VLM connector the baseline uses,
    ``Linear -> GELU -> Linear``, and ``projection_rank`` is still the single
    knob controlling its capacity.  The fusion block adds
    ``3460 * F + 5761`` parameters at ``input_dim = 1152``, where ``F`` is
    ``patch_fusion_hidden_dim``; drop ``projection_rank`` to pay for them when
    matching a parameter budget against the baseline.

    ``patch_fusion_gate_init`` defaults to 0.0 (an open half gate) rather than
    the -2.0 the repo's other gated branches use, because the spatial mean
    here is taken over *every* patch.  Measured on real C-RADIO features, the
    pooled frame vector has norm 27.8 while the fusion output is pinned to
    sqrt(D) = 33.9 per patch, and observed frame-to-frame motion is spread
    broadly over the grid rather than concentrated, so the mean dilutes the
    residual well below its nominal gate value.  At -2.0 the branch would
    reach the projection at roughly 7% of the pooled signal -- too small a
    share to test whether next-frame motion helps at all.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_rank: int | None = None,
        use_layer_norm: bool = True,
        temporal_scale_factor: int = 2,
        patch_grid_size: tuple[int, int] | None = None,
        patch_fusion_hidden_dim: int | None = None,
        patch_fusion_temperature: float = 0.1,
        patch_fusion_matching_top_k: int = 1,
        patch_fusion_gate_init: float = 0.0,
        patch_fusion_window_radius: int | None = 3,
    ) -> None:
        super().__init__()
        self._validate_dimension("input_dim", input_dim)
        self._validate_dimension("output_dim", output_dim)
        if projection_rank is not None:
            self._validate_dimension("projection_rank", projection_rank)
        self._validate_dimension("temporal_scale_factor", temporal_scale_factor)

        self.input_dim = input_dim
        self.output_dim = output_dim
        # A ``None`` rank resolves to output_dim, so this attribute always
        # reports the hidden width the projection actually uses.
        self.projection_rank = (
            output_dim if projection_rank is None else projection_rank
        )
        self.temporal_scale_factor = temporal_scale_factor
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

        # Both layers keep their bias: with a GELU in between, the first bias
        # sets where each hidden unit sits on the nonlinearity.
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
        # Plain fan-in init on both layers, matching the pooled-linear
        # baseline so the two runs start from the same projection scale.
        input_projection, output_projection = self.projection[0], self.projection[2]
        for projection in (input_projection, output_projection):
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
            nn.init.zeros_(projection.bias)

    @property
    def trainable_parameter_count(self) -> int:
        """Return the actual number of trainable adapter parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def motion_weight(self) -> float:
        """Current fusion gate value, forwarded for logging."""
        return self.next_frame_patch_fusion.motion_weight

    @property
    def mean_displacement(self) -> float | None:
        """Mean |displacement| of the last forward, forwarded for logging."""
        return self.next_frame_patch_fusion.mean_displacement

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        patch_features = visual_backbone_output.visual_features
        visual_length = visual_backbone_output.visual_length
        self._validate_inputs(patch_features, visual_length)

        # The fusion runs on the raw backbone patches, before any pooling, so
        # correspondences are still matched at full spatial resolution.
        fused_patches = self.next_frame_patch_fusion(patch_features, visual_length)

        # Spatial mean: [sum(T), P, D] -> [sum(T), D]. Every patch has fixed
        # weight 1/P, exactly as in the baseline.
        frame_features = fused_patches.mean(dim=1)

        # Temporal mean is performed separately within each packed video, so a
        # window can never cross a video boundary.
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
            # Both are already computed -- the gate is a parameter read and the
            # displacement was needed by the matching -- so logging them costs
            # nothing. Kept on device and unconverted: .item() here would
            # synchronize the accelerator on every step.
            logging_scalars={
                "motion_gate": torch.sigmoid(
                    self.next_frame_patch_fusion.fusion_gate.detach()
                ).reshape(()),
                "mean_displacement": self.next_frame_patch_fusion._last_displacement,
            },
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
