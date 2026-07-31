"""DINOv2 frame adapter with residual next-frame motion fusion."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.misc import random_derangement


class DINOFrameAdapterCrossV2(nn.Module):
    """Convert packed DINOv2 features into two tokens per frame.

    For each current frame, patches from the next frame are first aligned with
    the current patches using cosine-similarity aggregation. The aligned
    temporal difference is then fused through a residual MLP:

        fused_t = x_t + gate * MLP(LN(aligned(x_{t+1}) - x_t))

    Frames are packed along dimension 0 and ``visual_length`` defines the video
    boundaries. The last frame of every video has no next frame, so its
    temporal residual is explicitly set to zero. Each frame produces two
    interleaved LLM tokens: ``[mapped_cls_t, mapped_fused_patch_t]``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        temporal_hidden_dim: int | None = None,
        temperature: float = 0.1,
        temporal_gate_init: float = -2.0,  # will be passed through sigmoid to get initial gate value , sigmoid(-2) ~= 0.12
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        hidden_dim = hidden_dim or output_dim
        temporal_hidden_dim = temporal_hidden_dim or input_dim
        self.temperature = temperature

        # NOTE: V2 learns an explicit residual transformation of the aligned
        # next-frame difference instead of concatenating two frame features.
        self.temporal_norm = nn.LayerNorm(input_dim)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(input_dim, temporal_hidden_dim),
            nn.GELU(),
            nn.Linear(temporal_hidden_dim, input_dim),
        )

        # NOTE: Start with a small temporal contribution (sigmoid(-2) ~= 0.12)
        # so that early training remains close to the original DINO features.
        self.temporal_gate = nn.Parameter(torch.tensor(float(temporal_gate_init)))

        # NOTE: Fused patches remain D-dimensional in V2, rather than becoming
        # 2D-dimensional through concatenation.
        self.patch_score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1, bias=False),
        )

        # NOTE: CLS and fused-patch tokens have different distributions, so
        # they use separate LayerNorms while sharing the mapping MLP.
        self.cls_norm = nn.LayerNorm(input_dim)
        self.fused_patch_norm = nn.LayerNorm(input_dim)
        self.shared_mapper = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # NOTE: Token-type embeddings let the LLM distinguish global CLS tokens
        # from local motion-aware patch tokens despite the shared mapper.
        self.cls_type_embedding = nn.Parameter(torch.zeros(1, output_dim))
        self.fused_patch_type_embedding = nn.Parameter(torch.zeros(1, output_dim))

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
        return_weights: bool = True,
    ) -> VisualAdapterOutput:
        patch_features = visual_backbone_output.visual_features
        cls_token = visual_backbone_output.pooled_visual_features
        visual_length = visual_backbone_output.visual_length

        if visual_length is None:
            raise ValueError(
                "visual_length must be provided for DINOFrameAdapterCrossV2"
            )
        if patch_features is None or cls_token is None:
            raise ValueError(
                "patch_features and cls_token must be provided for "
                "DINOFrameAdapterCrossV2"
            )

        self._validate_inputs(patch_features, cls_token, visual_length)

        # A left content shift makes position t receive x_(t+1), so V2
        # combines the current frame with the next frame (not the previous one).
        next_patches, has_next = self._next_frame_shift(patch_features, visual_length)

        aligned_next = self.similarity_aggregate(patch_features, next_patches)

        # WARN: Do not create artificial motion at the last frame of a video.
        # Although its shifted value is itself, self-aggregation can still mix
        # spatial patches, so masking the residual is necessary.
        temporal_delta = aligned_next - patch_features
        temporal_delta = temporal_delta * has_next[:, None, None].to(
            dtype=temporal_delta.dtype
        )

        # Motion-aware residual fusion proposed for V2.
        temporal_residual = self.temporal_mlp(self.temporal_norm(temporal_delta))
        # Mask again after the biased MLP so an all-zero input cannot
        # produce a learned non-zero residual at a video's final frame.
        temporal_residual = temporal_residual * has_next[:, None, None].to(
            dtype=temporal_residual.dtype
        )
        fused_patches = (
            patch_features + torch.sigmoid(self.temporal_gate) * temporal_residual
        )

        patch_weights = self.patch_score(fused_patches).squeeze(-1)
        patch_weights = patch_weights.softmax(dim=1)
        pooled_patches = torch.bmm(patch_weights.unsqueeze(1), fused_patches).squeeze(1)

        # Map the two token types independently with type-specific norms
        # and a shared MLP, then interleave them frame by frame:
        # [CLS_0, PATCH_0, CLS_1, PATCH_1, ...].
        mapped_cls = (
            self.shared_mapper(self.cls_norm(cls_token)) + self.cls_type_embedding
        )
        mapped_fused_patches = (
            self.shared_mapper(self.fused_patch_norm(pooled_patches))
            + self.fused_patch_type_embedding
        )

        # if permute_video_tokens is True, randomly shuffle the order of frames within each video.
        if permute_video_tokens:
            mapped_cls, mapped_fused_patches, patch_weights = (
                self._permute_video_tokens(
                    mapped_cls, mapped_fused_patches, patch_weights, visual_length
                )
            )

        visual_features = torch.stack(
            (mapped_cls, mapped_fused_patches), dim=1
        ).flatten(0, 1)

        # NOTE: Both tokens belonging to frame t share temporal position t.
        position_ids = torch.cat(
            [
                torch.arange(length, device=visual_features.device).repeat_interleave(2)
                for length in visual_length
            ]
        )

        return VisualAdapterOutput(
            visual_features=visual_features,
            # NOTE: V2 emits two visual tokens per input frame.
            visual_length=visual_length * 2,
            position_ids=position_ids,
            extras={"patch_weights": patch_weights} if return_weights else None,
        )

    @staticmethod
    def _permute_video_tokens(cls, fused_patch, patch_weights, visual_length):
        permutation = random_derangement(visual_length, device=cls.device)
        return (cls[permutation], fused_patch[permutation], patch_weights[permutation])

    def similarity_aggregate(
        self,
        base: Tensor,
        shifted: Tensor,
    ) -> Tensor:
        """Align next-frame patches to current-frame patches."""
        base_norm = F.normalize(base, dim=-1)
        shifted_norm = F.normalize(shifted, dim=-1)
        similarity = torch.einsum("bnd,btd->bnt", base_norm, shifted_norm)

        # NOTE: Cosine logits lie in [-1, 1]; temperature prevents attention
        # over many patches from becoming excessively uniform.
        weights = F.softmax(similarity / self.temperature, dim=-1)
        return torch.einsum("bnt,btd->bnd", weights, shifted)

    @staticmethod
    def _next_frame_shift(
        visual_features: Tensor,
        visual_length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Move each next frame to the current position within each video.

        Returns:
            shifted: Next-frame features. A video's final position keeps its
                own features to ensure every gather index is valid.
            has_next: Boolean mask with shape ``[F]``. It is false at the last
                frame of every video.
        """
        total_frames = visual_features.shape[0]
        device = visual_features.device
        boundaries = torch.cumsum(visual_length.to(device=device), dim=0)
        ends = boundaries - 1

        base = torch.arange(total_frames, device=device)
        has_next = torch.ones(total_frames, dtype=torch.bool, device=device)
        has_next[ends] = False
        source_idx = base + has_next.to(dtype=base.dtype)

        # WARN: There is deliberately no torch.no_grad() here. Index gathering
        # should preserve gradients to next-frame features when DINO is tuned.
        return visual_features[source_idx], has_next

    @staticmethod
    def _validate_inputs(
        patch_features: Tensor,
        cls_token: Tensor,
        visual_length: Tensor,
    ) -> None:
        if patch_features.ndim != 3:
            raise ValueError(
                "patch_features must have shape [F, P, D], got "
                f"{tuple(patch_features.shape)}"
            )
        if cls_token.ndim != 2:
            raise ValueError(
                f"cls_token must have shape [F, D], got {tuple(cls_token.shape)}"
            )
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if bool((visual_length <= 0).any()):
            raise ValueError("all entries in visual_length must be positive")
        if patch_features.shape[0] != cls_token.shape[0]:
            raise ValueError("patch_features and cls_token must have the same F")
        if patch_features.shape[-1] != cls_token.shape[-1]:
            raise ValueError("patch_features and cls_token must have the same D")
        if int(visual_length.sum().item()) != patch_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed frames"
            )


if __name__ == "__main__":
    import torch
    from torch import Tensor

    # Original adapter test
    B, N, D = 3, 16, 768
    cls_token = torch.randn(B, D).cuda()
    patch_features = torch.randn(B, N, D).cuda()
    visual_length = torch.tensor([1, 2]).cuda()

    visual_backbone_output = VisualBackboneOutput(
        visual_features=patch_features,
        pooled_visual_features=cls_token,
        visual_length=visual_length,
    )

    adapter = DINOFrameAdapterCrossV2(input_dim=D, output_dim=512).cuda()
    adapter.eval()
    with torch.no_grad():
        output = adapter(visual_backbone_output)
        print("Output shape:", output.visual_features.shape)
        print("Patch weights shape:", output.extras["patch_weights"].shape)
        print("Visual length:", output.visual_length)
