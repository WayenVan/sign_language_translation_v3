"""Spatial mean pooling with a gated motion residual over the temporal window.

This is ``SpatiotemporalPooledLinearAdapter`` with exactly one thing changed:
the fixed temporal mean becomes ``MotionTemporalFusion``.  Everything else --
the spatial mean, the LayerNorm, the two-layer connector, ``projection_rank``
-- is identical, so a run of this adapter against that baseline isolates the
temporal operator and nothing else.

Data flow::

    patch features [F, P, D]
             |
             v  mean over P            (no selection, no CLS)
    frame features [F, D]
             |
             v  unflatten into windows of s
    windows [N, s, D]                          N = F / s
             |
      +------+------------------------------+
      |                                      |
      v  mean over s                         v  adjacent differences
    STATIC [N, D]                          d = [x_1-x_0, ... ] [N, (s-1)D]
      |                                      |
      |                                      v  LayerNorm
      |                                      v  Linear -> 2H
      |                                      v  SwiGLU: silu(a) * b -> [N, H]
      |                                      v  Linear -> D
      |                                    MOTION [N, D]
      |                                      |
      |                                      v  LayerNorm   <- pins the branch
      |                                      |                 scale so the gate
      |                                      v  * sigmoid(g)   stays readable
      |                                      |
      +------------------> + <---------------+
                           |
                           v
             fused frame features [N, D]      gate -> 0 recovers the
                           |                  baseline temporal mean exactly
                           v  LayerNorm
                           v  Linear -> R
                           v  GELU
                           v  Linear -> D_out
                           |
                           v
              visual tokens [N, D_out]

The left branch is the baseline; the right branch is everything this adapter
adds.  F is the sum of all frame counts in the packed video batch, s is
``temporal_scale_factor``, H is ``motion_hidden_dim`` and R is
``projection_rank``.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput


class MotionTemporalFusion(nn.Module):
    """Fuse ``s`` consecutive frame features into one, mean plus motion.

    ::

        m = mean(x_t .. x_{t+s-1})                     # static content
        d = [x_{t+1} - x_t, ..., x_{t+s-1} - x_{t+s-2}]  # motion
        y = m + sigmoid(g) * LayerNorm(SwiGLU(LayerNorm(d)))

    Two deliberate departures from the older ``TemporalShuffleAdapter``:

    - **The motion path sees only the differences.**  That module fed it
      ``[x_t, ..., x_{t+s-1}, d]``, which is rank-deficient: ``d`` is a linear
      function of the frames beside it, so a linear layer on the concatenation
      spans exactly the space that ``[x_t, ..., x_{t+s-1}]`` already spans.  The
      redundant block cost a third of the largest weight matrix and bought no
      expressive power.  Splitting the window into ``m`` (handled by the
      residual) and ``d`` (handled here) is a change of basis over the same
      space, so nothing is lost and the input width drops from ``D*(2s-1)`` to
      ``D*(s-1)``.
    - **The branch output is normalized before the gate.**  ``sigmoid(g) * f(x)``
      alone is not identifiable -- ``f`` can grow its own weights to offset a
      small gate, which is exactly what happened in the earlier runs: their
      gates sat at their 0.12 initialization for 80k steps while the branch's
      input projection grew fivefold.  Normalizing first pins the branch scale,
      so ``sigmoid(g)`` becomes a readable measure of how much motion the model
      actually uses, comparable across runs.

    ``gate_init`` is deliberately not zero.  A gate that starts at exactly zero
    multiplies the whole branch by zero, so every parameter behind it receives
    zero gradient and has to bootstrap through a single scalar; sigmoid(-2) is
    about 0.12, small enough to start near the mean-pooling baseline but large
    enough that the branch trains from the first step.

    The residual ``m`` carries whatever scale the backbone's features have,
    while the normalized branch is unit-scale, so ``sigmoid(g)`` is a ratio
    against that scale rather than an absolute fraction.  The adapter's own
    LayerNorm runs after this module and absorbs the combined scale.
    """

    def __init__(
        self,
        hidden_dim: int,
        scale_factor: int = 2,
        motion_hidden_dim: int | None = None,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        _validate_dimension("hidden_dim", hidden_dim)
        _validate_dimension("scale_factor", scale_factor)
        if scale_factor < 2:
            raise ValueError(
                "scale_factor must be at least 2 for a temporal difference"
            )
        if motion_hidden_dim is not None:
            _validate_dimension("motion_hidden_dim", motion_hidden_dim)
        if isinstance(gate_init, bool) or not isinstance(gate_init, (int, float)):
            raise TypeError("gate_init must be a real number")
        if not math.isfinite(float(gate_init)):
            raise ValueError("gate_init must be finite")

        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        self.motion_hidden_dim = (
            hidden_dim if motion_hidden_dim is None else motion_hidden_dim
        )

        difference_dim = hidden_dim * (scale_factor - 1)
        self.motion_norm = nn.LayerNorm(difference_dim)
        # SwiGLU: one projection produces both the gate and the value half.
        self.motion_in = nn.Linear(difference_dim, self.motion_hidden_dim * 2)
        self.motion_out = nn.Linear(self.motion_hidden_dim, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for projection in (self.motion_in, self.motion_out):
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
            nn.init.zeros_(projection.bias)

    @property
    def motion_weight(self) -> float:
        """Current gate value: how much motion rides on the mean."""
        return float(torch.sigmoid(self.gate).item())

    def forward(self, frame_features: Tensor) -> Tensor:
        """Fuse ``[sum(T), D]`` frame features into ``[sum(T) / s, D]``.

        Every video length is a multiple of ``scale_factor``, so video
        boundaries in the packed batch land on window boundaries and a window
        can never span two videos.  Windowing the packed tensor directly is
        therefore equivalent to splitting per video first, and it avoids the
        device synchronization that reading the lengths would cost.
        """
        if frame_features.ndim != 2:
            raise ValueError(
                "frame_features must have shape [sum(T), D], got "
                f"{tuple(frame_features.shape)}"
            )
        if frame_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"frame feature dimension must be {self.hidden_dim}, got "
                f"{frame_features.shape[-1]}"
            )

        windows = frame_features.unflatten(0, (-1, self.scale_factor))
        static = windows.mean(dim=1)
        differences = (windows[:, 1:] - windows[:, :-1]).flatten(start_dim=1)

        hidden = self.motion_in(self.motion_norm(differences))
        gate_half, value_half = hidden.chunk(2, dim=-1)
        motion = self.motion_out(F.silu(gate_half) * value_half)

        motion_weight = torch.sigmoid(self.gate).to(dtype=static.dtype)
        return static + motion_weight * self.gate_norm(motion)


class SpatiotemporalMotionAdapter(nn.Module):
    """Spatially mean-pool patches, fuse frames with motion, then project.

    Identical to ``SpatiotemporalPooledLinearAdapter`` except that the fixed
    temporal mean is replaced by :class:`MotionTemporalFusion`.  The backbone's
    pooled/CLS feature is still ignored, there is still no learned patch
    selection or positional embedding, and the connector is still the standard
    ``Linear -> GELU -> Linear`` with ``projection_rank`` as its hidden width.

    With affine LayerNorm enabled, the exact parameter count is the baseline's
    ``2 * input_dim + R * (input_dim + output_dim + 1) + output_dim`` plus the
    fusion module's ``2 * D * (s - 1) + D * (s - 1) * 2H + 2H + H * D + D
    + 2 * D + 1``, where ``D`` is ``input_dim``, ``s`` is
    ``temporal_scale_factor`` and ``H`` is ``motion_hidden_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_rank: int | None = None,
        use_layer_norm: bool = True,
        temporal_scale_factor: int = 2,
        motion_hidden_dim: int | None = None,
        motion_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        _validate_dimension("input_dim", input_dim)
        _validate_dimension("output_dim", output_dim)
        if projection_rank is not None:
            _validate_dimension("projection_rank", projection_rank)
        _validate_dimension("temporal_scale_factor", temporal_scale_factor)

        self.input_dim = input_dim
        self.output_dim = output_dim
        # A ``None`` rank resolves to output_dim, so this attribute always
        # reports the hidden width the projection actually uses.
        self.projection_rank = (
            output_dim if projection_rank is None else projection_rank
        )
        self.temporal_scale_factor = temporal_scale_factor
        self.norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()

        self.temporal_fusion = MotionTemporalFusion(
            hidden_dim=input_dim,
            scale_factor=temporal_scale_factor,
            motion_hidden_dim=motion_hidden_dim,
            gate_init=motion_gate_init,
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

    def _reset_projection_parameters(self) -> None:
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
        """Gate value of the temporal fusion, for logging across runs."""
        return self.temporal_fusion.motion_weight

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        patch_features = visual_backbone_output.visual_features
        visual_length = visual_backbone_output.visual_length
        self._validate_inputs(patch_features, visual_length)

        # Spatial mean: [sum(T), P, D] -> [sum(T), D].  Every patch has fixed
        # weight 1/P; the backbone's pooled/CLS feature is never consumed.
        frame_features = patch_features.mean(dim=1)

        # Temporal fusion replaces the baseline's fixed mean.  Windows never
        # span two videos because every visual length divides the window size.
        pooled_features = self.temporal_fusion(frame_features)
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


def _validate_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
