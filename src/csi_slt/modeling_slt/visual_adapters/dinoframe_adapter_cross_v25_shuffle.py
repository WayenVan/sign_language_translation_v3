"""Temporal-compression wrapper with gated CLS/PATCH fusion.

Model flow::

    VisualBackboneOutput
    (frame CLS + patch features, lengths=[T1, T2, ...])
                         |
                         v
              DINOFrameAdapterCrossV2
                         |
                         v
       [CLS_0, PATCH_0, CLS_1, PATCH_1, ...]
                         |
                  de-interleave
                  /             \
                 v               v
          CLS token stream   PATCH token stream
          [CLS_0, CLS_1, ...] [PATCH_0, PATCH_1, ...]
                 |               |
                 v               v
          CLS TemporalShuffle PATCH TemporalShuffle
          (scale_factor=S)    (scale_factor=S)
                 |               |
                 v               v
          compressed CLS     compressed PATCH
                  \             /
                   \           /
                    v         v
            g = sigmoid(fusion_gate)
            fused = g * CLS + (1 - g) * PATCH
                         |
                         v
                    LayerNorm
                         |
                         v
            optional PackedShortTemporalConv
                         |
                         v
               VisualAdapterOutput
       (one token per temporal group, lengths=T/S)
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2 import (
    DINOFrameAdapterCrossV2,
)
from csi_slt.modeling_slt.visual_adapters.patch_shuffle import TemporalShuffleAdapter


class PackedShortTemporalConv(nn.Module):
    """Add a local temporal residual to packed variable-length sequences."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.norm = nn.LayerNorm(hidden_size)
        self.temporal_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
            bias=False,
        )
        self.activation = nn.GELU()
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.zeros_(self.temporal_conv.weight)
        mark_module_tree_as_initialized(self)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [sum(T), D], got {tuple(features.shape)}"
            )
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty 1D tensor")
        if lengths.is_floating_point() or lengths.is_complex():
            raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
        if bool((lengths <= 0).any()):
            raise ValueError("all temporal lengths must be positive")
        if int(lengths.sum().item()) != features.shape[0]:
            raise ValueError("lengths.sum() must equal the packed token count")

        sequences = torch.split(features, lengths.tolist(), dim=0)
        padded = pad_sequence(
            [self.norm(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=0.0,
        )
        residual = self.temporal_conv(padded.transpose(1, 2)).transpose(1, 2)
        residual = self.activation(residual)
        packed_residual = torch.cat(
            [residual[index, :length] for index, length in enumerate(lengths.tolist())],
            dim=0,
        )
        return features + torch.sigmoid(self.residual_gate) * packed_residual


class DINOFrameAdapterCrossV25Shuffle(nn.Module):
    """Compress CLS/PATCH streams independently, then fuse each aligned pair."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        cls_input_dim: int | None = None,
        temporal_hidden_dim: int | None = None,
        temperature: float = 0.1,
        temporal_gate_init: float = -2.0,
        spatial_window_radius: int | None = 3,
        spatial_grid_size: Sequence[int] | None = None,
        temporal_scale_factor: int = 2,
        use_short_temporal_conv: bool = False,
        short_temporal_kernel_size: int = 3,
        short_temporal_gate_init: float = -2.0,
        fusion_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if temporal_scale_factor < 2:
            raise ValueError(
                "temporal_scale_factor must be at least 2 for temporal compression"
            )

        self.temporal_scale_factor = temporal_scale_factor
        self.use_short_temporal_conv = use_short_temporal_conv
        self.frame_adapter = DINOFrameAdapterCrossV2(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cls_input_dim=cls_input_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            temperature=temperature,
            temporal_gate_init=temporal_gate_init,
            spatial_window_radius=spatial_window_radius,
            spatial_grid_size=spatial_grid_size,
        )
        self.cls_temporal_shuffle = TemporalShuffleAdapter(
            input_hidden_size=output_dim,
            output_hidden_size=output_dim,
            scale_factor=temporal_scale_factor,
        )
        self.patch_temporal_shuffle = TemporalShuffleAdapter(
            input_hidden_size=output_dim,
            output_hidden_size=output_dim,
            scale_factor=temporal_scale_factor,
        )
        self.fusion_gate = nn.Parameter(torch.tensor(float(fusion_gate_init)))
        self.fusion_norm = nn.LayerNorm(output_dim)
        self.short_temporal_conv = (
            PackedShortTemporalConv(
                hidden_size=output_dim,
                kernel_size=short_temporal_kernel_size,
                gate_init=short_temporal_gate_init,
            )
            if use_short_temporal_conv
            else None
        )

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        return_weights: bool = True,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        frame_length = visual_backbone_output.visual_length
        if frame_length is None:
            raise ValueError(
                "visual_length must be provided for DINOFrameAdapterCrossV25Shuffle"
            )

        frame_output = self.frame_adapter(
            visual_backbone_output,
            return_weights=return_weights,
            permute_video_tokens=permute_video_tokens,
        )
        expected_frame_tokens = frame_length * 2
        if not torch.equal(frame_output.visual_length, expected_frame_tokens):
            raise RuntimeError(
                "wrapped DINOFrameAdapterCrossV2 returned unexpected lengths"
            )
        interleaved_tokens = frame_output.visual_features
        if interleaved_tokens.shape[0] != int(expected_frame_tokens.sum().item()):
            raise RuntimeError(
                "wrapped DINOFrameAdapterCrossV2 returned unexpected token count"
            )

        cls_tokens, compressed_length = self.cls_temporal_shuffle(
            interleaved_tokens[0::2], frame_length
        )
        patch_tokens, patch_length = self.patch_temporal_shuffle(
            interleaved_tokens[1::2], frame_length
        )
        if not torch.equal(compressed_length, patch_length):
            raise RuntimeError("CLS and pooled-patch temporal lengths diverged")

        fusion_gate = torch.sigmoid(self.fusion_gate)
        visual_features = self.fusion_norm(
            fusion_gate * cls_tokens + (1.0 - fusion_gate) * patch_tokens
        )
        if self.use_short_temporal_conv:
            if self.short_temporal_conv is None:
                raise RuntimeError("short temporal convolution module is missing")
            visual_features = self.short_temporal_conv(
                visual_features, compressed_length
            )

        if visual_features.shape[0] != int(compressed_length.sum().item()):
            raise RuntimeError(
                "compressed visual token count does not match visual_length"
            )
        position_ids = torch.cat(
            [
                torch.arange(length, device=visual_features.device)
                for length in compressed_length
            ]
        )
        extras = dict(frame_output.extras or {})
        extras["cls_patch_fusion_gate"] = fusion_gate
        if self.use_short_temporal_conv:
            extras["short_temporal_gate"] = torch.sigmoid(
                self.short_temporal_conv.residual_gate
            )
        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=compressed_length,
            position_ids=position_ids,
            extras=extras,
        )
