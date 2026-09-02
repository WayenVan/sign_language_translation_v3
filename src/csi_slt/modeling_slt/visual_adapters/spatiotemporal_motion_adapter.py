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

from torch import Tensor, nn

from csi_slt.modeling_slt.misc import (
    SpatialDropoutMean,
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.motion_temporal_fusion import (
    MotionTemporalFusion,
)


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
        spatial_dropout: float = 0.0,
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
        self.spatial_pool = SpatialDropoutMean(spatial_dropout)
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
        frame_features = self.spatial_pool(patch_features)

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
