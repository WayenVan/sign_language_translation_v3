import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class TemporalShuffleAdapter(nn.Module):
    def __init__(
        self, input_hidden_size, output_hidden_size, scale_factor, mlp_depth=1
    ):
        """Fuse every ``s`` consecutive frame tokens into one token.

        ``z = [x_t; ...; x_{t+s-1}; Δx_t; ...; Δx_{t+s-2}]``
        ``y = Project(mean(x)) + sigmoid(g) * SwiGLU(LN(z))``
        ``T_out = T_in / s``

        ``z`` concatenates ordered frame features and adjacent-frame
        differences (``Δx_i = x_{i+1} - x_i``). ``g`` is a learnable scalar
        gate that controls the motion residual strength.

        Each video length must be divisible by ``s`` so a window never spans
        two videos in the packed batch.
        """
        super().__init__()
        if scale_factor < 2:
            raise ValueError(
                "Motion-aware temporal fusion requires scale_factor to be at least 2"
            )

        self.scale_factor = scale_factor
        # Kept as an attribute for configuration compatibility.  The previous
        # stack of GELU MLP layers is replaced by a single SwiGLU fusion block.
        self.mlp_depth = mlp_depth

        # The base path preserves information shared by the frames in a local
        # window.  It gives the adapter a stable pooling-like initial route.
        self.base_norm = nn.LayerNorm(input_hidden_size)
        self.base_projection = nn.Linear(input_hidden_size, output_hidden_size)

        # Besides the ordered frame features, feed first-order temporal
        # differences to the nonlinear path.  For scale_factor=2 this is
        # exactly [x_t, x_{t+1}, x_{t+1} - x_t].
        fusion_input_size = input_hidden_size * (2 * scale_factor - 1)
        self.fusion_norm = nn.LayerNorm(fusion_input_size)
        self.fusion_in_projection = nn.Linear(
            fusion_input_size, output_hidden_size * 2
        )
        self.fusion_out_projection = nn.Linear(
            output_hidden_size, output_hidden_size
        )

        # Begin close to the stable base path, then learn how much motion
        # residual to inject. sigmoid(-2) is approximately 0.12.
        self.motion_gate = nn.Parameter(torch.tensor(-2.0))

    def temporal_shuffle(self, x, t_length, scale_factor=2):
        # x [BT, D]
        #
        assert t_length.fmod(scale_factor).eq(0).all(), (
            "temporal length of all frames must be divisible by scale_factor"
        )
        _, D = x.size()
        x = rearrange(x, "(n s) d -> n s d", s=scale_factor, d=D)
        return x

    def forward(self, hidden_states, t_length):
        """
        hidden_states: shape of [B1+B2+B3..., D] , the concatenation of all temporal tokens in the batch
        t_length: exact value of [B1, B2, B3...], the temporal length of each sample in the batch
        """
        if hidden_states is None or t_length is None:
            raise ValueError(
                "TemporalShuffleAdapter requires pooled_visual_features and visual_length from visual_backbone_output"
            )
        frame_windows = self.temporal_shuffle(
            hidden_states, t_length, self.scale_factor
        )

        # Static/context path: average frame content in each local window.
        base = self.base_projection(self.base_norm(frame_windows.mean(dim=1)))

        # Motion path: preserve ordered frames and explicitly expose their
        # frame-to-frame changes to a gated nonlinear projection.
        frame_features = frame_windows.flatten(start_dim=1)
        temporal_deltas = (frame_windows[:, 1:] - frame_windows[:, :-1]).flatten(
            start_dim=1
        )
        fusion_input = torch.cat((frame_features, temporal_deltas), dim=-1)
        value, gate = self.fusion_in_projection(self.fusion_norm(fusion_input)).chunk(
            2, dim=-1
        )
        motion = self.fusion_out_projection(value * F.silu(gate))
        hidden_states = base + torch.sigmoid(self.motion_gate) * motion

        if t_length is not None:
            t_length = t_length // self.scale_factor

        return hidden_states, t_length
