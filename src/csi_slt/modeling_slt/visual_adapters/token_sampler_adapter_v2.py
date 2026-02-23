import torch
from einops import rearrange, repeat
from torch import nn
import math

from typing import Optional, Type
from timm.models.vision_transformer import (
    Attention,
    DropPath,
    Mlp,
    LayerScale,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from ..output_utils import VisualAdapterOutput, VisualBackboneOutput


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class TokenSampleAdapterV2(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        num_heads,
        num_extra_queries,
        num_layers_sampler,
        num_layers_connector,
        use_temporal_shuffle=True,
        sampler_out_mlp_depth=1,
        mlp_ratio=2.0,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        eps=1e-6,
    ):
        """
        V2 upgrade:
        1. add norm before token shuffle
        2. add positional embedding for extra queries
        """
        super().__init__()
        self.token_sampler = TokenSampler(
            hidden_size=hidden_size,
            target_hidden_size=target_hidden_size,
            num_heads=num_heads,
            num_layers=num_layers_sampler,
            num_extra_queries=num_extra_queries,
            mlp_depth=sampler_out_mlp_depth,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            eps=eps,
        )
        self.temporal_shuffle_connector = TemporalShuffleConnector(
            dim=target_hidden_size,
            n_layers=num_layers_connector,
            use_shuffle=use_temporal_shuffle,
            eps=eps,
        )

    def forward(self, visual_backbone_output: VisualBackboneOutput):
        # visual_backbone_output: (B, T, HW, C)
        visual_features = visual_backbone_output.visual_features
        visual_length = visual_backbone_output.visual_length

        sampled_tokens, sampled_length = self.token_sampler(
            visual_features, visual_length
        )

        connected_tokens, connected_length = self.temporal_shuffle_connector(
            sampled_tokens, sampled_length
        )

        return VisualAdapterOutput(
            visual_features=connected_tokens,
            visual_length=connected_length,
        )


class TokenSampler(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        num_heads,
        num_layers,
        num_extra_queries,
        mlp_depth=1,
        mlp_ratio=2.0,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        eps=1e-6,
    ):
        super().__init__()
        self.num_extra_queries = num_extra_queries
        self.extra_queries = nn.Parameter(
            torch.randn(1, num_extra_queries, hidden_size), requires_grad=True
        )
        self.positional_embedding = nn.Embedding(num_extra_queries, hidden_size)
        self.blocks = nn.ModuleList(
            [
                VisualAttentionBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    mlp_layer=Mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp = build_mlp(
            mlp_depth, hidden_size * self.num_extra_queries, target_hidden_size
        )
        self.norm = Gemma3RMSNorm(target_hidden_size, eps=eps)

    def forward(self, x, v_length):
        # x: (B, T, HW, C)

        if x is None or v_length is None:
            raise ValueError("visual_features and visual_length cannot be None")

        BT, HW, C = x.shape

        # NOTE: positional encoding is added to the extra queries
        extra_queries = self.extra_queries + self.positional_embedding(
            torch.arange(self.num_extra_queries, device=x.device)
        )  # (1, num_extra_queries, hidden_size)

        extra_queries = repeat(extra_queries, "1 n c -> bt n c", bt=BT)

        for block in self.blocks:
            extra_queries = block(extra_queries, x)

        extra_queries = rearrange(
            extra_queries, "bt n c -> bt (n c)"
        )  # (B*T, num_extra_queries * hidden_size)

        feats = self.mlp(extra_queries)
        # (B*T, Target_hidden_size)
        feats = self.norm(feats)

        return feats, v_length


class TemporalShuffleConnector(nn.Module):
    def __init__(self, dim, n_layers, use_shuffle=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.use_shuffle = use_shuffle
        self.blocks = nn.ModuleList(
            [
                TemporalShuffleBlock(
                    input_hidden_size=dim,
                    output_hidden_size=dim,
                    scale_factor=2 if use_shuffle else 1,
                    use_shuffle=use_shuffle,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.temporal_scale_factor = math.pow(2, n_layers) if use_shuffle else 1

    def forward(self, x, t_length):
        for block in self.blocks:
            x, t_length = block(x, t_length)
        return x, t_length


class VisualAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=True,
            batch_first=True,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        keys = self.norm0(keys)
        x = queries + self.drop_path1(
            self.ls1(self.attn(self.norm1(queries), keys, keys)[0])
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TemporalShuffleBlock(nn.Module):
    def __init__(
        self,
        input_hidden_size,
        output_hidden_size,
        scale_factor,
        use_shuffle=True,
        eps=1e-6,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_shuffle = use_shuffle

        if not use_shuffle and scale_factor > 1:
            raise ValueError(
                "Temporal shuffle must be enabled if scale_factor is greater than 1"
            )
        self.norm = Gemma3RMSNorm(input_hidden_size, eps=eps)

        self.modality_projection = nn.Linear(
            input_hidden_size * scale_factor, output_hidden_size, bias=False
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

    def forward(self, video_hidden_states, t_length=None):

        # pre norm
        video_hidden_states = self.norm(video_hidden_states)

        if self.use_shuffle:
            video_hidden_states = self.temporal_shuffle(
                video_hidden_states, t_length, self.scale_factor
            )

        video_hidden_states = self.modality_projection(video_hidden_states)

        if t_length is not None:
            t_length = t_length // self.scale_factor

        return video_hidden_states, t_length


if __name__ == "__main__":
    # Example usage
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    num_extra_queries = 4
    target_hidden_size = 1024

    visual_adapter = VisualSampleAdapter(
        hidden_size, target_hidden_size, num_heads, num_layers, num_extra_queries
    )
    x = torch.randn(2, 128, 196, hidden_size)  # Example input
    output = visual_adapter(x, v_length=torch.tensor([128, 128]))  # Example v_length
    print(output[0].shape)  # Should be (2, 30, num_extra_queries, hidden_size)
