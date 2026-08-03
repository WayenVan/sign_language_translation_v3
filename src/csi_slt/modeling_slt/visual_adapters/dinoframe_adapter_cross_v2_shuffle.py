"""Temporal-compression wrapper for :class:`DINOFrameAdapterCrossV2`."""

import torch
from torch import nn

from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2 import (
    DINOFrameAdapterCrossV2,
)
from csi_slt.modeling_slt.visual_adapters.patch_shuffle import TemporalShuffleAdapter


class DINOFrameAdapterCrossV2Shuffle(nn.Module):
    """Apply V2, then independently temporally compress its CLS/PATCH streams.

    The wrapped V2 adapter is unchanged and produces packed interleaved tokens
    ``[CLS_0, PATCH_0, CLS_1, PATCH_1, ...]``.  This wrapper de-interleaves
    those tokens, applies separate temporal shuffle modules to the two streams,
    and interleaves the compressed streams again.

    Every input video length must be divisible by ``temporal_scale_factor``.
    Consequently, a video with ``T`` frames emits
    ``2 * (T / temporal_scale_factor)`` visual tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        cls_input_dim: int | None = None,
        temporal_hidden_dim: int | None = None,
        temperature: float = 0.1,
        temporal_gate_init: float = -2.0,
        temporal_scale_factor: int = 2,
    ) -> None:
        super().__init__()
        if temporal_scale_factor < 2:
            raise ValueError(
                "temporal_scale_factor must be at least 2 for temporal compression"
            )

        self.temporal_scale_factor = temporal_scale_factor
        self.frame_adapter = DINOFrameAdapterCrossV2(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            cls_input_dim=cls_input_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            temperature=temperature,
            temporal_gate_init=temporal_gate_init,
        )
        # CLS and pooled-patch tokens have different distributions, so the
        # compression branches intentionally do not share parameters.
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

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        return_weights: bool = True,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        frame_length = visual_backbone_output.visual_length
        if frame_length is None:
            raise ValueError(
                "visual_length must be provided for DINOFrameAdapterCrossV2Shuffle"
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

        cls_tokens = interleaved_tokens[0::2]
        patch_tokens = interleaved_tokens[1::2]
        cls_tokens, compressed_frame_length = self.cls_temporal_shuffle(
            cls_tokens, frame_length
        )
        patch_tokens, patch_frame_length = self.patch_temporal_shuffle(
            patch_tokens, frame_length
        )
        if not torch.equal(compressed_frame_length, patch_frame_length):
            raise RuntimeError("CLS and pooled-patch temporal lengths diverged")

        visual_features = torch.stack((cls_tokens, patch_tokens), dim=1).flatten(0, 1)
        visual_length = compressed_frame_length * 2
        if visual_features.shape[0] != int(visual_length.sum().item()):
            raise RuntimeError(
                "compressed visual token count does not match visual_length"
            )

        position_ids = torch.cat(
            [
                torch.arange(length, device=visual_features.device).repeat_interleave(2)
                for length in compressed_frame_length
            ]
        )
        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=visual_length,
            position_ids=position_ids,
            extras=frame_output.extras,
        )


if __name__ == "__main__":
    import torch
    from torch import Tensor

    # Original adapter test
    B, N, D_PATCH, D_CLS = 8, 16, 768, 1024
    cls_token = torch.randn(B, D_CLS).cuda()
    patch_features = torch.randn(B, N, D_PATCH).cuda()
    visual_length = torch.tensor([2, 6]).cuda()

    visual_backbone_output = VisualBackboneOutput(
        visual_features=patch_features,
        pooled_visual_features=cls_token,
        visual_length=visual_length,
    )

    adapter = DINOFrameAdapterCrossV2Shuffle(
        input_dim=D_PATCH,
        cls_input_dim=D_CLS,
        output_dim=512,
    ).cuda()
    adapter.eval()
    with torch.no_grad():
        output = adapter(visual_backbone_output)
        print("Output shape:", output.visual_features.shape)
        print("Patch weights shape:", output.extras["patch_weights"].shape)
        print("Visual length:", output.visual_length)
