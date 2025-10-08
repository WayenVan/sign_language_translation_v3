import torch.nn as nn
from einops import rearrange

from ..output_utils import VisualAdapterOutput, VisualBackboneOutput


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class LinearAdapter(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, mlp_depth=1):
        super().__init__()
        self.linear = build_mlp(
            depth=mlp_depth,
            hidden_size=input_hidden_size,
            output_hidden_size=output_hidden_size,
        )

    def forward(self, visual_backbone_output: VisualBackboneOutput):
        video_hidden_states = visual_backbone_output.pooled_visual_features
        t_length = visual_backbone_output.visual_length

        video_hidden_states = self.modality_projection(video_hidden_states)
        return VisualAdapterOutput(
            visual_features=video_hidden_states, visual_length=t_length
        )
