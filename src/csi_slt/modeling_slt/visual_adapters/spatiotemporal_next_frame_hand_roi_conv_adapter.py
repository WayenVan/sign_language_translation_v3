"""Next-frame patch fusion, hand-ROI pooling, and a learnable temporal filter.

Identical to :class:`SpatiotemporalNextFrameHandRoiAdapter` except for the
step that turns a window of ``temporal_scale_factor`` frames into one token::

    global mean + roi mean under raw mask   [F, 2*input_dim]
                    |
       fixed 1/s window mean  ---->   TemporalConvDownsample  (learnable, per-channel)
                    |                          |
                    |                 [optional] PackedTransformerContext
                    |                          |
       LayerNorm + rank projection  (unchanged)

``TemporalConvDownsample`` generalizes the boundary-safe temporal mean into a
learnable, per-channel temporal filter -- see that module's docstring for why
it is parameterized by a context ``radius`` rather than a raw kernel size, and
why its padding is per-video edge replication rather than zero. At
``radius=0`` it is initialized to reproduce the mean it replaces exactly, so
this adapter's first forward pass is numerically identical to the
mean-pooling baseline; everything downstream is free to learn away from that
starting point.

``PackedTransformerContext`` is optional (``num_transformer_layers=0``
disables it, and the module is not even constructed) and, when enabled, gives
each pooled token a way to attend across the whole video rather than only its
own local window -- something no amount of context radius on the convolution
alone can provide. It runs after the downsampling, on the same ``2*input_dim``
concatenated features the convolution produces, before the concat/gated
projection splits them apart -- so both the global and ROI halves gain the
same cross-video context, whichever fusion mode is selected.

``use_temporal_conv=False`` disables the convolution entirely and falls back
to the parent's exact fixed window mean -- no ``TemporalConvDownsample`` is
even constructed, so this is a structural switch, not just an initialization
that happens to agree with the mean at ``radius=0``. It exists so a config can
flip a single boolean to run the mean-pooling baseline and the learnable-filter
variant back to back with every other line identical, rather than relying on
``radius=0`` behaving like the mean only until the first optimizer step moves
it.
"""

import torch
from torch import nn

from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized, random_derangement
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.packed_transformer_context import (
    PackedTransformerContext,
)
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_hand_roi_adapter import (
    SpatiotemporalNextFrameHandRoiAdapter,
)
from csi_slt.modeling_slt.visual_adapters.temporal_conv_downsample import (
    TemporalConvDownsample,
)


class SpatiotemporalNextFrameHandRoiConvAdapter(SpatiotemporalNextFrameHandRoiAdapter):
    """``SpatiotemporalNextFrameHandRoiAdapter`` with a learnable temporal filter.

    Accepts every constructor argument of the parent adapter, plus:

    - ``use_temporal_conv``: ``True`` (the default) builds the learnable
      ``TemporalConvDownsample`` described above. ``False`` skips it entirely
      and pools with the parent's exact fixed window mean instead, so the
      adapter degrades to the mean-pooling baseline structurally, not merely
      numerically at step 0.
    - ``temporal_conv_radius``: context radius of the downsampling
      convolution: ``kernel_size = temporal_scale_factor + 2 * radius``. ``0``
      (the default) keeps the parent's exact non-overlapping window and
      reproduces its mean numerically at initialization. Ignored when
      ``use_temporal_conv=False``, and must be left at ``0`` there -- a
      non-zero radius with the convolution disabled has nothing to apply it
      to.
    - ``num_transformer_layers``: number of self-attention context blocks
      after the downsampling. ``0`` (the default) builds none, which makes
      this adapter identical in every way -- shape, parameter count,
      forward pass -- to the same class with only the temporal conv added.
    - ``transformer_num_heads``, ``transformer_mlp_ratio``,
      ``transformer_dropout``: forwarded to ``PackedTransformerContext``
      when it is built; unused and unvalidated otherwise.
    """

    def __init__(
        self,
        *args,
        use_temporal_conv: bool = True,
        temporal_conv_radius: int = 0,
        num_transformer_layers: int = 0,
        transformer_num_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(use_temporal_conv, bool):
            raise TypeError("use_temporal_conv must be a boolean")
        if not use_temporal_conv and temporal_conv_radius != 0:
            raise ValueError(
                "temporal_conv_radius has no effect when use_temporal_conv is "
                "False; leave it at its default of 0"
            )
        pooled_dim = 2 * self.input_dim

        self.temporal_conv_downsample = None
        if use_temporal_conv:
            self.temporal_conv_downsample = TemporalConvDownsample(
                hidden_dim=pooled_dim,
                scale_factor=self.temporal_scale_factor,
                radius=temporal_conv_radius,
            )
            mark_module_tree_as_initialized(self.temporal_conv_downsample)

        self.transformer_context = None
        if num_transformer_layers > 0:
            self.transformer_context = PackedTransformerContext(
                hidden_dim=pooled_dim,
                num_layers=num_transformer_layers,
                num_heads=transformer_num_heads,
                mlp_ratio=transformer_mlp_ratio,
                dropout=transformer_dropout,
            )
            mark_module_tree_as_initialized(self.transformer_context)

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

        # The one step that differs from SpatiotemporalNextFrameHandRoiAdapter:
        # a learnable per-channel temporal filter instead of a fixed window
        # mean, unless use_temporal_conv=False asked to fall back to that
        # exact mean. Boundary safety (no window crosses a video) is each
        # path's own responsibility.
        if self.temporal_conv_downsample is not None:
            pooled_features = self.temporal_conv_downsample(frame_features, visual_length)
        else:
            video_features = torch.split(frame_features, visual_length.tolist(), dim=0)
            pooled_features = torch.cat(
                [
                    features.unflatten(0, (-1, self.temporal_scale_factor)).mean(dim=1)
                    for features in video_features
                ],
                dim=0,
            )
        pooled_length = visual_length // self.temporal_scale_factor

        if self.transformer_context is not None:
            pooled_features = self.transformer_context(pooled_features, pooled_length)

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
