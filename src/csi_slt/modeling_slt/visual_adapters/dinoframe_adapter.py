import torch
from torch import Tensor, nn
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput


class DINOFrameAdapter(nn.Module):
    """Convert DINOv2 spatial features into one feature per video frame.

    DINOv2 represents an image using two relevant types of output tokens:

    - ``cls_token``: a global image representation with shape ``[B, D]``.
      It is commonly used for image-level classification or retrieval.
    - ``patch_features``: local representations with shape ``[B, N, D]``,
      where ``N = H * W`` is the number of image patches. Each token
      corresponds to one spatial patch but has already interacted with all
      other tokens through the DINOv2 Transformer blocks.

    In the official DINOv2 implementation, these are exposed as
    ``x_norm_clstoken`` and ``x_norm_patchtokens`` after the backbone's final
    LayerNorm. Register tokens, when present, are returned separately and are
    intentionally not consumed by this adapter.

    This adapter performs three operations:

    1. Assigns a learned scalar importance score to every patch token.
    2. Applies softmax over all patches and computes a weighted average.
    3. Concatenates the pooled local feature with the global CLS feature and
       maps the result to ``output_dim`` using a two-layer GELU MLP.

    The patch pooling is not full Transformer self-attention. It does not
    compute pairwise patch-to-patch attention and therefore has linear
    complexity with respect to the number of patches.

    Args:
        input_dim:
            Hidden dimension ``D`` of the DINOv2 backbone.

        output_dim:
            Dimension of the output frame representation.

        hidden_dim:
            Hidden dimension of the adapter MLP. Defaults to ``output_dim``.

    Shape:
        - cls_token: ``[B, D]``
        - patch_features: ``[B, H * W, D]``
        - output: ``[B, output_dim]``

    Example:
        >>> features = dinov2.forward_features(images)
        >>> frame_features = adapter(
        ...     features["x_norm_clstoken"],
        ...     features["x_norm_patchtokens"],
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or output_dim

        # Produce one content-dependent importance score for each DINO patch.
        # Bias is unnecessary because softmax is invariant to adding the same
        # scalar offset to every patch score.
        self.patch_score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1, bias=False),
        )

        # Fuse DINOv2's global CLS representation with the pooled local
        # representation and adapt it to the temporal model's feature space.
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        return_weights: bool = True,
    ) -> VisualAdapterOutput:
        """Build one representation for each input frame.

        ``patch_weights[b, i]`` describes the relative contribution of patch
        ``i`` to the pooled local feature of frame ``b``. The weights are
        input-dependent, non-negative, and sum to one over the patch dimension.

        The weights should be interpreted as learned pooling coefficients,
        rather than guaranteed hand, face, or object localization maps. They
        are optimized only through the downstream training objective.
        """
        patch_features = visual_backbone_output.visual_features
        cls_token = visual_backbone_output.pooled_visual_features

        patch_weights = self.patch_score(patch_features).squeeze(-1)
        patch_weights = patch_weights.softmax(dim=1)  # [B, N]

        pooled_patches = torch.bmm(
            patch_weights.unsqueeze(1),
            patch_features,
        ).squeeze(1)  # [B, D]

        frame_features = self.adapter(torch.cat((cls_token, pooled_patches), dim=-1))

        output = VisualAdapterOutput(
            visual_features=frame_features,
            visual_length=visual_backbone_output.visual_length,
            extras={"patch_weights": patch_weights} if return_weights else None,
        )

        return output


if __name__ == "__main__":
    # Test the adapter with dummy data
    B, N, D = 3, 16, 768  # Batch size, number of patches, feature dimension
    cls_token = torch.randn(B, D).cuda()
    patch_features = torch.randn(B, N, D).cuda()
    visual_length = torch.tensor([1, 2]).cuda()

    visual_backbone_output = VisualBackboneOutput(
        visual_features=cls_token,
        pooled_visual_features=patch_features,
        visual_length=visual_length,
    )

    adapter = DINOFrameAdapter(input_dim=D, output_dim=512).cuda()
    adapter.eval()
    with torch.no_grad():
        output = adapter(visual_backbone_output)
        print("Output shape:", output.visual_features.shape)
        print("Patch weights shape:", output.extras["patch_weights"].shape)
        print("Visual length:", output.visual_length)
