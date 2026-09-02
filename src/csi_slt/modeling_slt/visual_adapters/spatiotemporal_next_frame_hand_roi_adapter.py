"""Next-frame patch fusion followed by global and hand-ROI pooling.

Data flow::

    raw patches [F, P, D] ---- frozen scorer ----> Top-K mask
             |
             v
    NextFramePatchFusion
             |
      fused patches [F, P, D]
             |
       +-----+------------------+
       |                        |
       v                        v
    global mean          ROI mean under raw mask
       |                        |
       +----- concat/gated -----+
                  |
          temporal mean over s
                  |
       LayerNorm + rank projection
                  |
          visual tokens [F/s, D_out]

The scorer always sees raw backbone patches, because its frozen coefficients
are valid only for that feature distribution. Both content branches pool the
next-frame-fused patches; :class:`TopKRoiPool` explicitly supports scoring one
tensor while pooling another.
"""

import torch

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.hand_roi_pooled_adapter import (
    HandRoiPooledAdapter,
)
from csi_slt.modeling_slt.visual_adapters.next_frame_fusion import (
    NextFramePatchFusion,
)


class SpatiotemporalNextFrameHandRoiAdapter(HandRoiPooledAdapter):
    """Fuse next-frame patches, then combine their global and hand-ROI means.

    ``fusion_mode`` retains the two behaviours of ``HandRoiPooledAdapter``:

    - ``concat`` jointly projects ``[global; roi]``.
    - ``gated`` adds a normalized ROI projection to the baseline-like global
      projection through a learnable sigmoid gate.

    Temporal pooling, output projection, scorer loading, input validation and
    token counts otherwise remain aligned with the pooled-linear baseline.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scorer_path: str | None = None,
        top_k: int = 24,
        projection_rank: int | None = None,
        use_layer_norm: bool = True,
        temporal_scale_factor: int = 2,
        freeze_scorer: bool = True,
        fusion_mode: str = "concat",
        roi_projection_rank: int | None = None,
        gate_init: float = -2.0,
        spatial_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        roi_projection_dropout: float | None = None,
        patch_grid_size: tuple[int, int] | None = None,
        patch_fusion_hidden_dim: int | None = None,
        patch_fusion_temperature: float = 0.1,
        patch_fusion_matching_top_k: int = 1,
        patch_fusion_gate_init: float = 1.0,
        patch_fusion_window_radius: int | None = 3,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            scorer_path=scorer_path,
            top_k=top_k,
            projection_rank=projection_rank,
            use_layer_norm=use_layer_norm,
            temporal_scale_factor=temporal_scale_factor,
            freeze_scorer=freeze_scorer,
            fusion_mode=fusion_mode,
            roi_projection_rank=roi_projection_rank,
            gate_init=gate_init,
            spatial_dropout=spatial_dropout,
            projection_dropout=projection_dropout,
            roi_projection_dropout=roi_projection_dropout,
        )
        self.next_frame_patch_fusion = NextFramePatchFusion(
            hidden_dim=input_dim,
            fusion_hidden_dim=patch_fusion_hidden_dim,
            temperature=patch_fusion_temperature,
            matching_top_k=patch_fusion_matching_top_k,
            gate_init=patch_fusion_gate_init,
            spatial_window_radius=patch_fusion_window_radius,
            grid_size=patch_grid_size,
        )
        # HandRoiPooledAdapter marks the modules created by super().__init__.
        # This fusion is attached afterwards and therefore needs its own mark.
        mark_module_tree_as_initialized(self.next_frame_patch_fusion)

    @property
    def patch_motion_weight(self) -> float:
        """Current gate value of the next-frame patch residual."""
        return self.next_frame_patch_fusion.motion_weight

    @property
    def mean_displacement(self) -> float | None:
        """Mean matched displacement from the most recent forward pass."""
        return self.next_frame_patch_fusion.mean_displacement

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        raw_patches = visual_backbone_output.visual_features
        visual_length = visual_backbone_output.visual_length
        self._validate_inputs(raw_patches, visual_length)

        fused_patches = self.next_frame_patch_fusion(raw_patches, visual_length)

        # Selection remains calibrated to raw backbone features, while both
        # branches carry the richer next-frame-fused content.
        roi_features = self.roi_pool(raw_patches, fused_patches)
        global_features = self.spatial_pool(fused_patches)
        frame_features = torch.cat([global_features, roi_features], dim=-1)

        # Identical boundary-safe temporal mean to HandRoiPooledAdapter and the
        # pooled-linear baseline.
        video_features = torch.split(frame_features, visual_length.tolist(), dim=0)
        pooled_features = torch.cat(
            [
                features.unflatten(0, (-1, self.temporal_scale_factor)).mean(dim=1)
                for features in video_features
            ],
            dim=0,
        )
        pooled_length = visual_length // self.temporal_scale_factor

        if self.fusion_mode == "concat":
            visual_features = self.projection(self.norm(pooled_features))
        else:
            pooled_global, pooled_roi = pooled_features.split(self.input_dim, dim=-1)
            visual_features = self.projection(self.norm(pooled_global)) + torch.sigmoid(
                self.fusion_gate
            ) * self.roi_projection(self.roi_norm(pooled_roi))

        if permute_video_tokens:
            permutation = random_derangement(
                pooled_length, device=visual_features.device
            )
            visual_features = visual_features[permutation]

        logging_scalars = {
            "motion_gate": torch.sigmoid(
                self.next_frame_patch_fusion.fusion_gate.detach()
            ).reshape(()),
            "mean_displacement": self.next_frame_patch_fusion._last_displacement,
            "roi_global_distance": (roi_features - global_features)
            .detach()
            .norm(dim=-1)
            .mean()
            .reshape(()),
            "selection_margin": self.roi_pool.score_margin(raw_patches),
        }
        if self.fusion_gate is not None:
            logging_scalars["roi_gate"] = torch.sigmoid(
                self.fusion_gate.detach()
            ).reshape(())

        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=pooled_length,
            logging_scalars=logging_scalars,
        )
