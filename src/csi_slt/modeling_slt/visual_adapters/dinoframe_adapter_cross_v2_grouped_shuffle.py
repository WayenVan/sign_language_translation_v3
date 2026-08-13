r"""Overlapping temporal-window wrapper for :class:`DINOFrameAdapterCrossV2`.

Approximate data flow (the default example uses window=3 and stride=2)::

    packed DINO features from independent videos
                         |
                         v
             DINOFrameAdapterCrossV2
       (next-frame patch alignment and pooling)
                         |
                         v
       [C0, P0, C1, P1, C2, P2, ...]
                  /             \
                 v               v
       CLS stream [C0,C1,...]   PATCH stream [P0,P1,...]
                 |               |
                 |  split at video boundaries
                 |               |
                 v               v
       centred replicate-padded temporal windows

       input:       x0       x1       x2       x3       x4
       centres:     ^                 ^                 ^
       windows:  [x0,x0,x1]       [x1,x2,x3]       [x3,x4,x4]
                       (one output per window)

                 |               |
                 v               v
       independent CLS/PATCH window fusion modules
       mean-content path + gated ordered-motion path
                  \             /
                   v           v
          [Cw0, Pw0, Cw1, Pw1, Cw2, Pw2, ...]

Every input length must be divisible by ``stride``. For each video,
``T_out = T / stride`` and the final visual token length is ``2 * T_out``. A
window is always built inside one video, so temporal context can never leak
across packed-video boundaries.

"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2 import (
    DINOFrameAdapterCrossV2,
)


class PackedTemporalWindowAdapter(nn.Module):
    """Fuse overlapping, boundary-aware windows in packed video sequences.

    Windows are centred on frame indices ``0, stride, 2 * stride, ...``.
    Indices outside an individual video are clamped to its first or last frame,
    which is equivalent to replicate padding and never crosses video boundaries.
    Every video length must be divisible by ``stride``, and a video of length
    ``T`` emits exactly ``T / stride`` tokens.
    """

    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        window_size: int = 3,
        stride: int = 2,
        motion_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if input_hidden_size <= 0 or output_hidden_size <= 0:
            raise ValueError("hidden sizes must be positive")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer of at least 3")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.window_size = window_size
        self.stride = stride

        self.base_norm = nn.LayerNorm(input_hidden_size)
        self.base_projection = nn.Linear(input_hidden_size, output_hidden_size)

        fusion_input_size = input_hidden_size * (2 * window_size - 1)
        self.fusion_norm = nn.LayerNorm(fusion_input_size)
        self.fusion_in_projection = nn.Linear(
            fusion_input_size, output_hidden_size * 2
        )
        self.fusion_out_projection = nn.Linear(
            output_hidden_size, output_hidden_size
        )
        self.motion_gate = nn.Parameter(torch.tensor(float(motion_gate_init)))

    def _make_windows(
        self, hidden_states: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build every video's windows with one vectorized gather."""
        output_lengths = lengths // self.stride
        total_frames = hidden_states.shape[0]
        radius = self.window_size // 2
        offsets = torch.arange(
            -radius, radius + 1, device=hidden_states.device
        )
        frame_indices = torch.arange(total_frames, device=hidden_states.device)

        # Map every packed frame to its video without materialising padded
        # sequences. ``right=True`` assigns a boundary index to the next video.
        video_ends = torch.cumsum(lengths, dim=0)
        video_ids = torch.searchsorted(video_ends, frame_indices, right=True)
        video_starts = video_ends - lengths
        local_indices = frame_indices - video_starts[video_ids]

        # Divisibility makes this mask select exactly T / stride centres per
        # video. All window positions are then gathered in a single operation.
        centre_mask = local_indices.remainder(self.stride).eq(0)
        centre_video_ids = video_ids[centre_mask]
        centre_local_indices = local_indices[centre_mask]
        centre_lengths = lengths[centre_video_ids]
        window_local_indices = centre_local_indices[:, None] + offsets[None, :]
        window_local_indices = torch.minimum(
            window_local_indices.clamp_min(0), centre_lengths[:, None] - 1
        )
        packed_window_indices = (
            video_starts[centre_video_ids, None] + window_local_indices
        )
        return hidden_states[packed_window_indices], output_lengths

    def forward(
        self, hidden_states: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must have shape [sum(T), D], got "
                f"{tuple(hidden_states.shape)}"
            )
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty 1D tensor")
        if lengths.is_floating_point() or lengths.is_complex():
            raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
        if bool((lengths <= 0).any()):
            raise ValueError("all temporal lengths must be positive")
        if int(lengths.sum().item()) != hidden_states.shape[0]:
            raise ValueError("lengths.sum() must equal the packed token count")
        if bool(lengths.remainder(self.stride).ne(0).any()):
            raise ValueError(
                "all temporal lengths must be divisible by stride "
                f"({self.stride})"
            )

        windows, output_lengths = self._make_windows(hidden_states, lengths)

        base = self.base_projection(self.base_norm(windows.mean(dim=1)))
        frame_features = windows.flatten(start_dim=1)
        temporal_deltas = (windows[:, 1:] - windows[:, :-1]).flatten(start_dim=1)
        fusion_input = torch.cat((frame_features, temporal_deltas), dim=-1)
        value, gate = self.fusion_in_projection(
            self.fusion_norm(fusion_input)
        ).chunk(2, dim=-1)
        motion = self.fusion_out_projection(value * F.silu(gate))
        output = base + torch.sigmoid(self.motion_gate) * motion
        return output, output_lengths


class DINOFrameAdapterCrossV2GroupedShuffle(nn.Module):
    """Apply V2, then fuse CLS and pooled-patch streams in sliding windows."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        cls_input_dim: int | None = None,
        temporal_hidden_dim: int | None = None,
        temperature: float = 0.1,
        temporal_gate_init: float = -2.0,
        temporal_window_size: int = 3,
        temporal_window_stride: int = 2,
        window_motion_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.temporal_window_size = temporal_window_size
        self.temporal_window_stride = temporal_window_stride

        self.frame_adapter = DINOFrameAdapterCrossV2(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cls_input_dim=cls_input_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            temperature=temperature,
            temporal_gate_init=temporal_gate_init,
        )

        # Global CLS and motion-oriented pooled patches intentionally use
        # different parameters while sharing exactly the same window layout.
        self.cls_temporal_window = PackedTemporalWindowAdapter(
            input_hidden_size=output_dim,
            output_hidden_size=output_dim,
            window_size=temporal_window_size,
            stride=temporal_window_stride,
            motion_gate_init=window_motion_gate_init,
        )
        self.patch_temporal_window = PackedTemporalWindowAdapter(
            input_hidden_size=output_dim,
            output_hidden_size=output_dim,
            window_size=temporal_window_size,
            stride=temporal_window_stride,
            motion_gate_init=window_motion_gate_init,
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
                "visual_length must be provided for "
                "DINOFrameAdapterCrossV2GroupedShuffle"
            )
        if permute_video_tokens:
            raise ValueError(
                "permute_video_tokens is incompatible with temporal windows"
            )

        frame_output = self.frame_adapter(
            visual_backbone_output,
            return_weights=return_weights,
            permute_video_tokens=False,
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

        cls_tokens, grouped_frame_length = self.cls_temporal_window(
            interleaved_tokens[0::2], frame_length
        )
        patch_tokens, patch_frame_length = self.patch_temporal_window(
            interleaved_tokens[1::2], frame_length
        )
        if not torch.equal(grouped_frame_length, patch_frame_length):
            raise RuntimeError("CLS and pooled-patch temporal lengths diverged")

        visual_features = torch.stack((cls_tokens, patch_tokens), dim=1).flatten(0, 1)
        visual_length = grouped_frame_length * 2
        if visual_features.shape[0] != int(visual_length.sum().item()):
            raise RuntimeError("windowed token count does not match visual_length")

        position_ids = torch.cat(
            [
                torch.arange(length, device=visual_features.device).repeat_interleave(2)
                for length in grouped_frame_length
            ]
        )
        extras = dict(frame_output.extras or {})
        extras.update(
            {
                "cls_window_motion_gate": torch.sigmoid(
                    self.cls_temporal_window.motion_gate
                ),
                "patch_window_motion_gate": torch.sigmoid(
                    self.patch_temporal_window.motion_gate
                ),
            }
        )

        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=visual_length,
            position_ids=position_ids,
            extras=extras,
        )
