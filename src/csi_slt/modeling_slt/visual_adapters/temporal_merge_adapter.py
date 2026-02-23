import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
from timm.models.vision_transformer import (
    Attention,
    DropPath,
    Mlp,
    LayerScale,
)
from typing import Optional, Type
from ..output_utils import VisualAdapterOutput, VisualBackboneOutput
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class TemporalMergeAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        num_heads_spatial,
        num_layers_spatial,
        num_layers_temporal,
        num_extra_queries_spatial,
        num_heads_temporal,
        use_temporal_shuffle=True,
        mlp_depth_spatial=1,
        drop_out=0.1,
        drop_path=0.0,
        eps=1e-6,
    ):
        super().__init__()
        self.num_extra_queries = num_extra_queries_spatial
        self.extra_queries = nn.Parameter(
            torch.randn(1, num_extra_queries_spatial, hidden_size), requires_grad=True
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_size,
                    num_heads=num_heads_spatial,
                    mlp_ratio=2.0,
                    proj_drop=drop_out,
                    attn_drop=drop_out,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    mlp_layer=Mlp,
                )
                for _ in range(num_layers_spatial)
            ]
        )
        self.mlp = build_mlp(
            mlp_depth_spatial, hidden_size * self.num_extra_queries, target_hidden_size
        )
        self.norm = Gemma3RMSNorm(target_hidden_size, eps=eps)
        # self.positional_embedding = nn.Embedding(max_length, target_hidden_size)
        #
        self.use_temporal_shuffle = use_temporal_shuffle
        self.temporal_merge_connector = TemporalMergeConnector(
            target_hidden_size,
            num_heads_temporal,
            use_temporal_shuffle,
            drop_out,
            num_layers_temporal,
        )

    def forward(self, visual_backbone_output: VisualBackboneOutput):
        # x: (B, T, HW, C)
        x = visual_backbone_output.visual_features
        v_length = visual_backbone_output.visual_length

        if x is None or v_length is None:
            raise ValueError("visual_features and visual_length cannot be None")

        BT, HW, C = x.shape

        extra_queries = repeat(self.extra_queries, "1 n c -> bt n c", bt=BT)
        for block in self.blocks:
            extra_queries = block(extra_queries, x)

        extra_queries = rearrange(
            extra_queries, "bt n c -> bt (n c)"
        )  # (B*T, num_extra_queries * hidden_size)
        feats = self.mlp(extra_queries)  # (B*T, Target_hidden_size)

        feats, v_length = self.temporal_merge_connector(
            feats, v_length
        )  # (B*T', Target_hidden_size), (B,)

        feats = self.norm(feats)

        return VisualAdapterOutput(
            visual_features=feats,  # (B*T', Target_hidden_size)
            visual_length=v_length,  # (B,)
        )


class TemporalMergeConnector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        use_shuffle=True,
        dropout=0.1,
        num_layers=1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TemporalDecoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    use_shuffle=use_shuffle,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_reduction_factor = math.prod(
            [layer.token_reduction_factor for layer in self.layers]
        )

    def forward(self, video_hidden_states, t_length):

        # video_hidden_states: (N1+N2+N3, input_dim)
        Z = video_hidden_states

        for layer in self.layers:
            Z, t_length = layer(Z, t_length)

        return Z, t_length


class TemporalDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads=4, use_shuffle=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_shuffle = use_shuffle
        self.token_reduction_factor = 8 if use_shuffle else 4

        self.shuffle_layer = TemporalShuffleLayer(
            input_hidden_size=dim,
            output_hidden_size=dim,
            scale_factor=2 if use_shuffle else 1,
            use_shuffle=use_shuffle,
        )

        self.merge_layer = TemporalMergeLayer(
            dim=dim, num_heads=num_heads, dropout=dropout
        )

    def forward(self, Z, t_length):
        # Z: (N1+N2+N3, d)
        N_ALL, D = Z.shape

        Z, t_length = self.shuffle_layer(Z, t_length)  # (B*T, d), T = t_length

        assert t_length.fmod(4).eq(0).all(), (
            "temporal length of all frames must be divisible by 4"
        )
        Z = rearrange(Z, "(b s) d -> b s d", s=4)  # (B, 4, d)
        Z = self.merge_layer(Z)  # (B, d)
        t_length = t_length // 4

        return Z, t_length


class TemporalMergeLayer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Learnable global token
        self.global_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional encoding for temporal awareness
        self.pos_emb = nn.Parameter(torch.randn(1, 500, dim))  # max length 500

        # Multi-head attention to merge
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # FFN after merge
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, Z):
        # Z: (B, N, d)
        B, N, d = Z.shape

        # 1) Add positional encoding
        pos = self.pos_emb[:, :N, :]
        Z_pos = Z + pos  # (B, N, d)

        # 2) Prepare global token for each batch
        global_tok = self.global_token.expand(B, -1, -1)  # (B, 1, d)

        # 3) Self-attention: global token queries all frame tokens
        #    Q = global_tok, K=V = Z_pos
        out, _ = self.attn(global_tok, Z_pos, Z_pos)
        # out: (B, 1, d)

        # 4) Add & Norm + FFN
        out = self.norm1(out + global_tok)
        out = self.norm2(out + self.ffn(out))

        # 5) Return fused token
        return out.squeeze(1)  # (B, d)


class TemporalShuffleLayer(nn.Module):
    def __init__(
        self,
        input_hidden_size,
        output_hidden_size,
        scale_factor,
        use_shuffle=True,
        activation=nn.GELU(),
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_shuffle = use_shuffle

        if not use_shuffle and scale_factor > 1:
            raise ValueError(
                "Temporal shuffle must be enabled if scale_factor is greater than 1"
            )

        self.modality_projection = nn.Sequential(
            nn.Linear(input_hidden_size * scale_factor, output_hidden_size),
            activation,
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
        if self.use_shuffle:
            video_hidden_states = self.temporal_shuffle(
                video_hidden_states, t_length, self.scale_factor
            )

        video_hidden_states = self.modality_projection(video_hidden_states)

        if t_length is not None:
            t_length = t_length // self.scale_factor

        return video_hidden_states, t_length


class Block(nn.Module):
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


if __name__ == "__main__":
    # Test the TemporalMergeAdapter with dummy data
    num_frames = [16, 32]
    t_length = torch.tensor(num_frames, dtype=torch.long).cuda()
    input_dim = 512
    output_dim = 512

    adapter = TemporalMergeAdapter(
        input_dim, output_dim, num_heads=4, use_shuffle=True
    ).cuda()
    video_hidden_states = torch.randn(sum(num_frames), input_dim).cuda()

    fused_tokens, new_t_length = adapter(video_hidden_states, t_length)
    print("Fused tokens shape:", fused_tokens.shape)  # Should be (B, output_dim)
    print(
        "New temporal length:", new_t_length
    )  # Should be [num_frames // reduction_factor]
