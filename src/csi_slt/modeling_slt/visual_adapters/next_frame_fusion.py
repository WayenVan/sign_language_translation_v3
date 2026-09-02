"""Next-frame patch fusion, shared by every adapter that uses it.

Main data flow::

    x [F, P, D] + packed lengths
              |
              v
    boundary-safe next-frame shift -> x_next, has_next
              |
              v
    cosine similarity -> local mask -> Top-K softmax -> weights
              |
       +------+------------------+
       |                         |
       v                         v
    aligned = weights @ x_next   displacement = weights @ pos - pos
       |                         |
       v                         v
    delta = aligned - x       Linear(2 -> H)
       |                         |
    LN -> Linear(D -> H)          |
       |       x -> LN -> Linear(D -> H)
       +-------------------------+
                    |
                    v
       sum / sqrt(3) -> GELU -> Linear -> LayerNorm
                    |
              mask by has_next
                    |
                    v
       output = x + sigmoid(gate) * update [F, P, D]

``F`` is the total number of packed frames, ``P`` the patches per frame and
``D`` the backbone feature width. At each video's final frame, ``has_next`` is
false and masks the update to exactly zero, so that frame passes through
unchanged.

Extracted so the adapters cannot drift apart. They had already started to:
three byte-identical copies existed until an initialization change landed in
one of them, leaving a run configured against a module that no longer matched
what it was measured on -- silently, since nothing imports across the copies.

The fusion aligns every patch with its match in the next frame, transforms the
resulting temporal delta and adds it back through a learnable residual gate.
Packed-video boundaries are respected: the final frame of each video receives
an exact zero residual.

``spatiotemporal_next_frame_motion_adapter`` keeps its own copy on purpose --
its docstring says so and a test asserts it -- so it is deliberately not a
consumer of this module.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
        gate_init: float = 1.0,
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
        # Fan-in init rather than zeros. Zeros never blocked this stream's
        # gradient -- a Linear's weight gradient depends on its input, not on
        # its own value -- but it did leave it two orders of magnitude behind
        # the other two, and a shared learning rate could not close the gap:
        # 42k steps into the wr3/hardmatch run it still contributed 0.9% of the
        # fusion hidden vector against content's 69% and delta's 74%, and its
        # weight norm was still climbing rather than settling.
        #
        # The handicap is structural, not a slow start: displacement enters
        # through 2 dimensions of roughly unit scale while content and delta
        # enter through 1152 LayerNormed ones, so beginning at zero on top of
        # that is a second penalty for the same thing. Fan-in measures out at
        # 15.2% of the hidden vector at init, a defensible share for a
        # 2-dimensional signal.
        #
        # Untested end to end: the configuration that produced test BLEU-4
        # 0.1075 used zeros, so this needs its own run rather than riding along
        # with another change.
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
