"""Minimal spatial-temporal mean-pooling adapter for capacity controls."""

import math

import torch
from torch import Tensor, nn

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput


class SpatiotemporalPooledLinearAdapter(nn.Module):
    """Spatially and temporally mean-pool patches, then linearly project.

    This adapter deliberately ignores the backbone's pooled/CLS feature and
    contains no learned patch selection, positional embedding, nonlinear
    activation, or learned spatial/temporal interaction.  Temporal processing
    is a fixed mean over non-overlapping windows inside each video.  It is meant
    to be the smallest stable temporal baseline against which learned
    mechanisms can be ablated.

    ``projection_rank`` controls the number of trainable projection parameters:

    - ``None``: one dense ``input_dim -> output_dim`` projection.
    - positive integer ``R``: a bias-free ``input_dim -> R`` projection followed
      by an ``R -> output_dim`` projection.  With no activation between them,
      the composed operation remains linear.  ``R`` may exceed either endpoint
      dimension when an over-parameterized but still linear capacity control is
      needed.

    With affine LayerNorm enabled, the exact parameter counts are:

    - dense: ``2 * input_dim + input_dim * output_dim + output_dim``;
    - rank R: ``2 * input_dim + R * (input_dim + output_dim) + output_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_rank: int | None = None,
        use_layer_norm: bool = True,
        temporal_scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self._validate_dimension("input_dim", input_dim)
        self._validate_dimension("output_dim", output_dim)
        if projection_rank is not None:
            self._validate_dimension("projection_rank", projection_rank)
        self._validate_dimension("temporal_scale_factor", temporal_scale_factor)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_rank = projection_rank
        self.temporal_scale_factor = temporal_scale_factor
        self.norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()

        if projection_rank is None:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, projection_rank, bias=False),
                nn.Linear(projection_rank, output_dim),
            )

        # Use fan-in initialization rather than allowing the outer HF model to
        # initialize both factor matrices with the same fixed standard
        # deviation.  The latter makes output variance grow with R and would
        # confound parameter-budget comparisons.  The second factor is scaled
        # so the dense and factorized forms have similar initial output scale.
        self._reset_projection_parameters()
        mark_module_tree_as_initialized(self)

    @staticmethod
    def _validate_dimension(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def _reset_projection_parameters(self) -> None:
        if self.projection_rank is None:
            nn.init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))
            nn.init.zeros_(self.projection.bias)
            return

        input_projection, output_projection = self.projection
        nn.init.kaiming_uniform_(input_projection.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(output_projection.weight, a=math.sqrt(5))
        with torch.no_grad():
            output_projection.weight.mul_(math.sqrt(3.0))
        nn.init.zeros_(output_projection.bias)

    @property
    def trainable_parameter_count(self) -> int:
        """Return the actual number of trainable adapter parameters."""
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

        # Spatial mean: [sum(T), P, D] -> [sum(T), D].  Every patch has fixed
        # weight 1/P; the backbone's pooled/CLS feature is never consumed.
        frame_features = patch_features.mean(dim=1)

        # Temporal mean is performed separately within each packed video, so a
        # window can never cross a video boundary.  Projection happens after
        # temporal pooling to keep the baseline as small and cheap as possible.
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
