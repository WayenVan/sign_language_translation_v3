import torch
from torch import nn

from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v25_shuffle import (
    DINOFrameAdapterCrossV25Shuffle,
)


class _FakeFrameAdapter(nn.Module):
    def __init__(self, interleaved_tokens):
        super().__init__()
        self.interleaved_tokens = interleaved_tokens

    def forward(self, visual_backbone_output, **kwargs):
        del kwargs
        return VisualAdapterOutput(
            visual_features=self.interleaved_tokens,
            visual_length=visual_backbone_output.visual_length * 2,
            extras={"source": "fake"},
        )


class _PairMeanShuffle(nn.Module):
    def forward(self, tokens, lengths):
        return tokens.reshape(-1, 2, tokens.shape[-1]).mean(dim=1), lengths // 2


def test_v25_fuses_cls_and_patch_into_one_token_per_temporal_group():
    output_dim = 4
    frame_lengths = torch.tensor([4, 2])
    interleaved_tokens = torch.randn(int(frame_lengths.sum()) * 2, output_dim)
    adapter = DINOFrameAdapterCrossV25Shuffle(
        input_dim=3, cls_input_dim=5, output_dim=output_dim, temporal_scale_factor=2
    )
    adapter.frame_adapter = _FakeFrameAdapter(interleaved_tokens)
    adapter.cls_temporal_shuffle = _PairMeanShuffle()
    adapter.patch_temporal_shuffle = _PairMeanShuffle()

    output = adapter(VisualBackboneOutput(visual_length=frame_lengths))

    assert torch.equal(output.visual_length, torch.tensor([2, 1]))
    assert output.visual_features.shape == (3, output_dim)
    assert torch.equal(output.position_ids, torch.tensor([0, 1, 0]))
    torch.testing.assert_close(output.extras["cls_patch_fusion_gate"], torch.tensor(0.5))
    output.visual_features[0, 0].backward()
    assert adapter.fusion_gate.grad is not None
    assert adapter.fusion_gate.grad.abs() > 0


def test_v25_adapter_is_registered():
    assert VISUAL_ADAPTERS["dinoframe_cross_v25_shuffle"] is DINOFrameAdapterCrossV25Shuffle


def test_v25_optional_temporal_conv_runs_after_fusion():
    adapter = DINOFrameAdapterCrossV25Shuffle(
        input_dim=3,
        cls_input_dim=5,
        output_dim=4,
        temporal_scale_factor=2,
        use_short_temporal_conv=True,
    )
    adapter.frame_adapter = _FakeFrameAdapter(torch.randn(8, 4))
    adapter.cls_temporal_shuffle = _PairMeanShuffle()
    adapter.patch_temporal_shuffle = _PairMeanShuffle()
    output = adapter(VisualBackboneOutput(visual_length=torch.tensor([4])))
    assert adapter.short_temporal_conv is not None
    assert "short_temporal_gate" in output.extras
    assert not hasattr(adapter, "cls_short_temporal_conv")
    assert not hasattr(adapter, "patch_short_temporal_conv")
