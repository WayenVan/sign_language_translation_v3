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

``NextFramePatchFusion`` lives in ``next_frame_fusion`` rather than being
re-implemented here, so this ablation and every other consumer always run
bit-identical fusion code.

Unlike the pooled-linear baseline this adapter has no ``use_cls_token`` mode:
the fusion matches patches against patches, so there is nothing for it to do
on a single CLS vector per frame.  F is the sum of all frame counts in the
packed video batch and s is ``temporal_scale_factor``.
"""

import math

import torch
from torch import Tensor, nn

from csi_slt.modeling_slt.misc import (
    SpatialDropoutMean,
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.next_frame_fusion import NextFramePatchFusion


class SpatiotemporalNextFrameAdapter(nn.Module):
    """Fuse each patch with its next-frame match, then pool and project.

    The projection is the same two-layer VLM connector the baseline uses,
    ``Linear -> GELU -> Linear``, and ``projection_rank`` is still the single
    knob controlling its capacity.  The fusion block adds
    ``3460 * F + 5761`` parameters at ``input_dim = 1152``, where ``F`` is
    ``patch_fusion_hidden_dim``; drop ``projection_rank`` to pay for them when
    matching a parameter budget against the baseline.

    ``patch_fusion_gate_init`` defaults well above the -2.0 the repo's other
    gated branches use, because the spatial mean here is taken over *every*
    patch.  Measured on real C-RADIO features, the pooled frame vector has norm
    27.8 while the fusion output is pinned to sqrt(D) = 33.9 per patch, and
    observed frame-to-frame motion is spread broadly over the grid rather than
    concentrated, so the mean dilutes the residual well below its nominal gate
    value.  At -2.0 the branch would reach the projection at roughly 7% of the
    pooled signal -- too small a share to test whether motion helps at all.

    sigmoid(1.0) = 0.73.  Measured against the 0.0 this used to default to,
    on the wr3/hardmatch configuration: dev BLEU-4 0.1135 by step 24k where 0.0
    reached 0.0982, and 0.0's own peak of 0.1075 took until step 42k.  The gate
    also stays put once training starts (0.976 -> 0.992 over 6k steps), so the
    model is not being dragged back toward a lower value.
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
        patch_fusion_gate_init: float = 1.0,
        patch_fusion_window_radius: int | None = 3,
        spatial_dropout: float = 0.0,
        projection_dropout: float = 0.0,
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
        # Applied after the fusion, never before: the fusion matches each patch
        # against a spatial neighbourhood in the next frame, so removing patches
        # first would break the correspondence rather than a shortcut.
        self.spatial_pool = SpatialDropoutMean(spatial_dropout)
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
        # Dropout only when asked for: nn.Sequential keys its state dict by
        # position, so inserting it unconditionally would renumber the second
        # Linear from 2 to 3 and stop every existing checkpoint from loading.
        projection_layers = [
            nn.Linear(input_dim, self.projection_rank),
            nn.GELU(),
        ]
        if projection_dropout > 0.0:
            projection_layers.append(nn.Dropout(projection_dropout))
        projection_layers.append(nn.Linear(self.projection_rank, output_dim))
        self.projection = nn.Sequential(*projection_layers)

        self._reset_projection_parameters()
        mark_module_tree_as_initialized(self)

    @staticmethod
    def _validate_dimension(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def _reset_projection_parameters(self) -> None:
        # Plain fan-in init on both layers, matching the pooled-linear
        # baseline so the two runs start from the same projection scale.
        # Selected by type rather than by index: the projection may carry a
        # Dropout between its layers, which would shift any fixed index.
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.zeros_(layer.bias)

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

    def optimization_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Expose the patch-motion residual gate to optimizer policies."""
        return {"gates": (self.next_frame_patch_fusion.fusion_gate,)}

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

        # Spatial mean: [sum(T), P, D] -> [sum(T), D]. Every surviving patch
        # has equal weight, exactly as in the baseline.
        frame_features = self.spatial_pool(fused_patches)

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
