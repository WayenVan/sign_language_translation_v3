import torch.nn as nn
import torch
from einops import rearrange

from ..output_utils import VisualAdapterOutput, VisualBackboneOutput


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class TemporalShuffleAdapter(nn.Module):
    def __init__(
        self, input_hidden_size, output_hidden_size, scale_factor, mlp_depth=1
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.modality_projection = build_mlp(
            depth=mlp_depth,
            hidden_size=input_hidden_size * scale_factor,
            output_hidden_size=output_hidden_size,
        )

    def temporal_shuffle(self, x, t_length, scale_factor=2):
        # x [BT, D]
        #
        assert t_length.fmod(scale_factor).eq(0).all(), (
            "temporal length of all frames must be divisible by scale_factor"
        )
        BT, D = x.size()
        x = rearrange(x, "(b s) d -> b  (s d)", s=scale_factor, d=D)
        return x

    def forward(self, visual_backbone_output: VisualBackboneOutput):
        video_hidden_states = visual_backbone_output.pooled_visual_features
        t_length = visual_backbone_output.visual_length

        if video_hidden_states is None or t_length is None:
            raise ValueError(
                "TemporalShuffleAdapter requires pooled_visual_features and visual_length from visual_backbone_output"
            )
        video_hidden_states = self.temporal_shuffle(
            video_hidden_states, t_length, self.scale_factor
        )
        video_hidden_states = self.modality_projection(video_hidden_states)

        if t_length is not None:
            t_length = t_length // self.scale_factor

        return VisualAdapterOutput(
            visual_features=video_hidden_states, visual_length=t_length
        )
