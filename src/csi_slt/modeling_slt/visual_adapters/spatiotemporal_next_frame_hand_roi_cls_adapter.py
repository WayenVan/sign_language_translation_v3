"""Next-frame + hand-ROI pooling with a third gated residual for the CLS token.

This is :class:`SpatiotemporalNextFrameHandRoiAdapter` with one addition: the
backbone's summary / CLS vector (``pooled_visual_features``) rides onto the
output token as its own gated residual, beside the hand-ROI one.

Data flow::

    VisualBackboneOutput
      raw patches [sum(T), P, D]              CLS / summary [sum(T), D_cls]
            |                                         |
            |  NextFramePatchFusion                   |
            |  (content + delta + displacement,       |
            |   gate init sigmoid(+1.0))              |
            v                                         |
      fused patches [sum(T), P, D]                    |
        |            \\                                |
        |             \\  frozen hand scorer          |
        |              \\ (scores RAW patches) --> top-k mask [sum(T), P]
        |               \\      |                     |
        v                v      v                     v
    spatial mean     ROI mean under mask       (no spatial axis)
     global [.,D]      roi [.,D]                      |
        |                |                            |
        +----- concat [., 2D] -----+                  |
                   |                                  |
      boundary-safe temporal mean over s frames  (same window mean)
                   |                                  |
             [N, 2D]  split                        [N, D_cls]
             /            \\                            |
      pooled_global    pooled_roi                  pooled_cls
            |               |                          |
        LN, proj        LN, roi_proj              LN_cls, cls_proj
     (Lin-GELU-Lin)   (..-Lin, tail LN*)        (..-Lin, tail LN*)
            |               |                          |
            |         x sigmoid(roi_gate)      x sigmoid(cls_gate)
            |               |                          |
            +-------------- + ------------------------ +
                            |
                    visual tokens [N = sum(T)/s, D_out]

    LN* = non-affine LayerNorm, pins the branch scale so the gate reads as a
    fraction. At sigmoid(cls_gate) = 0 the CLS branch drops out and this is
    bit-for-bit SpatiotemporalNextFrameHandRoiAdapter. Equivalently::

        out = projection(LN(global_mean))                       # 表 B 行 10, unchanged
            + sigmoid(roi_gate) * roi_projection(LN(roi))       # 表 B 行 10, unchanged
            + sigmoid(cls_gate) * cls_projection(LN_cls(CLS))   # new, 表 B 行 2

Why a residual and not a replacement. 表 B 行 2 measured CLS *instead of* the
spatial-mean global feature and gained +0.8 on the pooled-linear path. Here the
global slot is exactly where next-frame fusion (行 3, the table's strongest
module) delivers its gain, so CLS cannot take that slot without losing 行 3.
Adding it as a gate-from-zero residual keeps both: at ``sigmoid(cls_gate) = 0``
this adapter is bit-for-bit the next-frame + hand-ROI adapter, and the CLS half
has to earn its contribution -- the same discipline the ROI residual is under,
and the same shape .ai/visual_adapter_component_ablation_summary.md argues for.

Like the ROI branch, ``cls_projection`` ends in a non-affine LayerNorm so the
gate is an identifiable, readable fraction rather than something ``cls_projection``
can absorb into its own weights.

Token count, ``video_token_scale`` and the CTC head are untouched: still one
token per ``temporal_scale_factor`` frames. Only ``gated`` fusion is supported;
the residual has no meaning without the baseline-like global projection to sit
on.
"""

import math

import torch
from torch import Tensor, nn

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_hand_roi_adapter import (
    SpatiotemporalNextFrameHandRoiAdapter,
)


class SpatiotemporalNextFrameHandRoiClsAdapter(SpatiotemporalNextFrameHandRoiAdapter):
    """``SpatiotemporalNextFrameHandRoiAdapter`` plus a gated CLS residual.

    New keyword arguments:

    ``cls_input_dim``
        Width of the backbone summary vector in ``pooled_visual_features``
        (C-RADIOv4-SO400M: 2304). Required -- it differs from the patch width.
    ``cls_projection_rank``
        Hidden width of the CLS residual's ``Linear -> GELU -> Linear``. Defaults
        to ``projection_rank`` (i.e. the global branch's rank).
    ``cls_gate_init``
        Pre-sigmoid init of the CLS residual gate. Default ``-2.0`` ->
        ``sigmoid(-2) ~= 0.12``, matching the ROI residual: the CLS half is a
        whole vector beside the global one, not a residual diluted across a
        spatial mean, so it starts small and is selected on dev.
    ``cls_projection_dropout``
        Dropout after the CLS residual's GELU. Defaults to ``projection_dropout``.
        As in the ROI branch, the trailing non-affine LayerNorm cancels dropout's
        damping and leaves only direction noise.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cls_input_dim: int,
        scorer_path: str | None = None,
        top_k: int = 24,
        projection_rank: int | None = None,
        use_layer_norm: bool = True,
        temporal_scale_factor: int = 2,
        freeze_scorer: bool = True,
        fusion_mode: str = "gated",
        roi_projection_rank: int | None = None,
        gate_init: float = -2.0,
        spatial_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        roi_projection_dropout: float | None = None,
        cls_projection_rank: int | None = None,
        cls_gate_init: float = -2.0,
        cls_projection_dropout: float | None = None,
        patch_grid_size: tuple[int, int] | None = None,
        patch_fusion_hidden_dim: int | None = None,
        patch_fusion_temperature: float = 0.1,
        patch_fusion_matching_top_k: int = 1,
        patch_fusion_gate_init: float = 1.0,
        patch_fusion_window_radius: int | None = 3,
    ) -> None:
        if fusion_mode != "gated":
            raise ValueError(
                "SpatiotemporalNextFrameHandRoiClsAdapter only supports "
                f"fusion_mode='gated'; the CLS residual has no baseline "
                f"projection to sit on in 'concat' mode, got {fusion_mode!r}"
            )
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
            patch_grid_size=patch_grid_size,
            patch_fusion_hidden_dim=patch_fusion_hidden_dim,
            patch_fusion_temperature=patch_fusion_temperature,
            patch_fusion_matching_top_k=patch_fusion_matching_top_k,
            patch_fusion_gate_init=patch_fusion_gate_init,
            patch_fusion_window_radius=patch_fusion_window_radius,
        )
        self._validate_dimension("cls_input_dim", cls_input_dim)
        if cls_projection_rank is not None:
            self._validate_dimension("cls_projection_rank", cls_projection_rank)

        self.cls_input_dim = cls_input_dim
        self.cls_projection_rank = (
            self.projection_rank if cls_projection_rank is None else cls_projection_rank
        )
        self.cls_projection_dropout = (
            projection_dropout
            if cls_projection_dropout is None
            else cls_projection_dropout
        )

        self.cls_norm = (
            nn.LayerNorm(cls_input_dim) if use_layer_norm else nn.Identity()
        )
        # Same shape as the ROI residual: Linear -> GELU -> [Dropout] -> Linear,
        # pinned by a non-affine LayerNorm so sigmoid(cls_gate) is a readable
        # fraction rather than something cls_projection can rescale away.
        self.cls_projection = self._build_projection(
            cls_input_dim,
            self.cls_projection_rank,
            output_dim,
            self.cls_projection_dropout,
            tail=nn.LayerNorm(output_dim, elementwise_affine=False),
        )
        # One-dimensional for FSDP2 compatibility, matching self.fusion_gate.
        self.cls_gate = nn.Parameter(torch.tensor([float(cls_gate_init)]))

        # HandRoiPooledAdapter.__init__ already ran _reset_projection_parameters
        # and mark_module_tree_as_initialized(self) before this branch existed,
        # so the CLS modules need their own fan-in init and marking.
        for layer in self.cls_projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.zeros_(layer.bias)
        mark_module_tree_as_initialized(self)

    @property
    def cls_weight(self) -> float:
        """Current gate value: how much of the CLS residual rides on the token."""
        return float(torch.sigmoid(self.cls_gate).item())

    def optimization_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Every scalar gate in the adapter, as one ``"gates"`` group.

        The three sit on very different scales (patch-level motion residual,
        ROI residual, CLS residual) and an optimizer policy usually wants to
        treat them together and apart from the projections.
        """
        gates = [self.next_frame_patch_fusion.fusion_gate]
        if self.fusion_gate is not None:
            gates.append(self.fusion_gate)
        gates.append(self.cls_gate)
        return {"gates": tuple(gates)}

    def _temporal_window_mean(
        self, frame_features: Tensor, visual_length: Tensor
    ) -> Tensor:
        """Boundary-safe mean over each ``temporal_scale_factor``-frame window.

        The same idiom the parent inlines for the concatenated global/ROI
        features; kept here as a helper so the CLS stream is pooled identically
        without duplicating it a third time.
        """
        per_video = torch.split(frame_features, visual_length.tolist(), dim=0)
        return torch.cat(
            [
                features.unflatten(0, (-1, self.temporal_scale_factor)).mean(dim=1)
                for features in per_video
            ],
            dim=0,
        )

    def _validate_cls_features(
        self, cls_features: Tensor | None, visual_length: Tensor
    ) -> None:
        if cls_features is None:
            raise ValueError(
                "pooled_visual_features must carry the backbone summary / CLS "
                "vector for SpatiotemporalNextFrameHandRoiClsAdapter"
            )
        if cls_features.ndim != 2:
            raise ValueError(
                "pooled_visual_features must have shape [sum(T), cls_input_dim], "
                f"got {tuple(cls_features.shape)}"
            )
        if cls_features.shape[-1] != self.cls_input_dim:
            raise ValueError(
                f"CLS feature dimension must be {self.cls_input_dim}, got "
                f"{cls_features.shape[-1]}"
            )
        if cls_features.shape[0] != int(visual_length.sum().item()):
            raise ValueError(
                "pooled_visual_features must carry one summary vector per packed "
                "frame; its length must match visual_length.sum()"
            )

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        raw_patches = visual_backbone_output.visual_features
        visual_length = visual_backbone_output.visual_length
        self._validate_inputs(raw_patches, visual_length)
        cls_features = visual_backbone_output.pooled_visual_features
        self._validate_cls_features(cls_features, visual_length)

        fused_patches = self.next_frame_patch_fusion(raw_patches, visual_length)

        # Selection stays calibrated to raw backbone features, while both patch
        # branches carry the richer next-frame-fused content.
        roi_features = self.roi_pool(raw_patches, fused_patches)
        global_features = self.spatial_pool(fused_patches)
        frame_features = torch.cat([global_features, roi_features], dim=-1)

        pooled_features = self._temporal_window_mean(frame_features, visual_length)
        pooled_cls = self._temporal_window_mean(cls_features, visual_length)
        pooled_length = visual_length // self.temporal_scale_factor

        pooled_global, pooled_roi = pooled_features.split(self.input_dim, dim=-1)
        visual_features = (
            self.projection(self.norm(pooled_global))
            + torch.sigmoid(self.fusion_gate)
            * self.roi_projection(self.roi_norm(pooled_roi))
            + torch.sigmoid(self.cls_gate)
            * self.cls_projection(self.cls_norm(pooled_cls))
        )

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
            "cls_gate": torch.sigmoid(self.cls_gate.detach()).reshape(()),
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
