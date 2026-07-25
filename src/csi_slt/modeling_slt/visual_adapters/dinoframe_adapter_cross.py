"""DINOv2 frame adapter with cross-frame patch aggregation.

Frames from a variable-length video batch are packed along dimension 0.  In
this module, ``F = sum(visual_length)`` is the number of packed frames and
``P`` is the number of DINO patch tokens *within one frame*.  ``P`` is never a
temporal dimension.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput


class DINOFrameAdapterCross(nn.Module):
    """Convert packed DINOv2 frame features into one token per frame.

    DINOv2 represents an image using two relevant types of output tokens:

    - ``cls_token``: a global image representation with shape ``[F, D]``.
      It is commonly used for image-level classification or retrieval.
    - ``patch_features``: local representations with shape ``[F, P, D]``,
      where ``P = H * W`` is the number of image patches. Each token
      corresponds to one spatial patch but has already interacted with all
      other tokens through the DINOv2 Transformer blocks.

    In the official DINOv2 implementation, these are exposed as
    ``x_norm_clstoken`` and ``x_norm_patchtokens`` after the backbone's final
    LayerNorm. Register tokens, when present, are returned separately and are
    intentionally not consumed by this adapter.

    The frames of each video are contiguous along dimension 0.  First, the
    patch features for each frame are shifted one frame to the right without
    crossing a video boundary.  Every patch in the current frame then uses
    cosine-similarity attention to aggregate all patches from that shifted
    frame.  Finally, the adapter pools the concatenated current and aggregated
    patch features, combines them with the CLS token, and projects the result.

    The adapter performs these operations:

    1. Right-shifts packed frame features within each video segment.
    2. Aggregates shifted-frame patch features for each current-frame patch.
    3. Scores and pools the resulting ``2D``-dimensional patch features.
    4. Concatenates the pooled local feature with the ``D``-dimensional CLS
       feature and maps the resulting ``3D`` vector to ``output_dim``.

    The cross-frame aggregation computes a ``[P, P]`` cosine-similarity matrix
    per packed frame, so its time and memory complexity are quadratic in the
    number of patches. The final learned patch pooling itself is linear in
    ``P`` and is not Transformer self-attention.

    Args:
        input_dim:
            Hidden dimension ``D`` of the DINOv2 backbone.

        output_dim:
            Dimension of the output frame representation.

        hidden_dim:
            Hidden dimension of the adapter MLP. Defaults to ``output_dim``.

    Shape:
        - ``visual_backbone_output.visual_features``: ``[F, P, D]``
        - ``visual_backbone_output.pooled_visual_features``: ``[F, D]``
        - ``visual_backbone_output.visual_length``: ``[B]``, with
          ``F = visual_length.sum()``
        - output: ``[F, output_dim]``

    Example:
        >>> backbone_output = backbone(packed_frames, video_lengths)
        >>> adapter_output = adapter(backbone_output)
        >>> adapter_output.visual_features.shape
        torch.Size([video_lengths.sum(), output_dim])
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
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, 1, bias=False),
        )

        # Fuse DINOv2's global CLS representation with the pooled local
        # representation and adapt it to the temporal model's feature space.
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        return_weights: bool = True,
    ) -> VisualAdapterOutput:
        """Build one representation for every packed input frame.

        Args:
            visual_backbone_output: DINOv2 output containing packed patch
                features ``[F, P, D]``, packed CLS features ``[F, D]``, and
                per-video frame lengths ``[B]``. The frame features for each
                video must occupy one contiguous segment along dimension 0.
            return_weights: Whether to include pooling weights in ``extras``.

        Returns:
            Adapted frame tokens ``[F, output_dim]`` with the original
            ``visual_length``. When requested, ``extras["patch_weights"]`` has
            shape ``[F, P]``.

        ``patch_weights[f, p]`` describes the relative contribution of patch
        ``p`` to the pooled local feature of packed frame ``f``. The weights are
        input-dependent, non-negative, and sum to one over the patch dimension.

        The weights should be interpreted as learned pooling coefficients,
        rather than guaranteed hand, face, or object localization maps. They
        are optimized only through the downstream training objective.
        """
        patch_features = visual_backbone_output.visual_features
        cls_token = visual_backbone_output.pooled_visual_features
        visual_length = visual_backbone_output.visual_length

        if visual_length is None:
            raise ValueError("visual_length must be provided for DINOFrameAdapter")

        if patch_features is None or cls_token is None:
            raise ValueError(
                "patch_features and cls_token must be provided for DINOFrameAdapter"
            )

        shifted_patches = self._visual_feature_shift(
            patch_features, visual_length, direction="right"
        )
        aggretaed_shifted = self.similarity_aggregate(patch_features, shifted_patches)
        aggretated_patches = torch.cat(
            (patch_features, aggretaed_shifted), dim=-1
        ).contiguous()

        patch_weights = self.patch_score(aggretated_patches).squeeze(-1)
        patch_weights = patch_weights.softmax(dim=1)  # [F, P]
        pooled_patches = torch.bmm(
            patch_weights.unsqueeze(1),
            aggretated_patches,
        ).squeeze(1)  # [F, 2D]

        frame_features = self.adapter(torch.cat((cls_token, pooled_patches), dim=-1))

        output = VisualAdapterOutput(
            visual_features=frame_features,
            visual_length=visual_length,
            extras={"patch_weights": patch_weights} if return_weights else None,
        )

        return output

    @staticmethod
    def similarity_aggregate(
        base: torch.Tensor,  # [F, P, D]
        shifted: torch.Tensor,  # [F, P, D]
    ) -> torch.Tensor:
        """Aggregate patches from the shifted frame for each current patch.

        ``F`` is the number of packed video frames and ``P`` is the number of
        spatial patch tokens per frame.  The attention is spatial: for every
        packed frame and every current patch, it normalizes cosine similarities
        over the ``P`` patches of that frame's shifted-frame features.  It does
        not attend over a time dimension.

        Args:
            base: Current-frame patch features ``[F, P, D]``.
            shifted: Previous- (or next-) frame patch features ``[F, P, D]``.

        Returns:
            Aggregated shifted features with shape ``[F, P, D]``.
        """
        base_norm = F.normalize(base, dim=-1)
        shifted_norm = F.normalize(shifted, dim=-1)

        # [F, P, P]: [packed frame, current patch, shifted-frame patch]
        similarity = torch.einsum("bnd,btd->bnt", base_norm, shifted_norm)

        # Normalize over the shifted-frame patch dimension.
        weights = F.softmax(similarity, dim=-1)

        # [F, P, P] × [F, P, D] -> [F, P, D]
        aggregated = torch.einsum("bnt,btd->bnd", weights, shifted)

        return aggregated

    @staticmethod
    @torch.no_grad()
    def _visual_feature_shift(
        visual_features: Tensor,
        visual_length: Tensor,
        direction: str = "right",
    ) -> Tensor:
        """Shift packed frames by one position within each video segment.

        ``visual_features`` has shape ``[F, P, D]`` and is a concatenation of
        variable-length videos along dimension 0. ``F = sum(visual_length)``
        is the total number of frames, while ``P`` is the number of spatial
        patch tokens in each frame. The function shifts complete ``[P, D]``
        frame slices; it does not shift or otherwise alter the patch dimension.
        The function is executed under ``torch.no_grad()``, so the shifted
        tensor is intentionally detached from ``visual_features``.

        Two shift directions are supported:

        - ``"right"``: frame ``i`` receives the features of frame ``i - 1``.
          The first frame stays unchanged (no previous frame). The last frame
          receives the second-to-last frame's features, which naturally
          satisfies the "copy previous frame" boundary condition.

        - ``"left"``: frame ``i`` receives the features of frame ``i + 1``.
          The last frame stays unchanged (no next frame). The first frame
          receives the second frame's features.

        Both directions use a single vectorized index gather, avoiding Python
        loops and per-segment memory copies.

        Args:
            visual_features: Concatenated frame features with shape
                ``[F, P, D]``.
            visual_length: Positive frame lengths of the packed video segments
                with shape ``[B]``; their sum must equal ``F``.
            direction: ``"right"`` (default) or ``"left"``.

        Returns:
            Shifted features with the same shape as ``visual_features``.
        """
        if direction not in ("left", "right"):
            raise ValueError(f"direction must be 'left' or 'right', got '{direction}'")

        B_total = visual_features.shape[0]
        device = visual_features.device

        # Boundary of each segment: cumsum gives the exclusive end index.
        boundaries = torch.cumsum(visual_length, dim=0)  # e.g. [3, 8, 10]

        # For each position in [0, B_total), decide the offset to the source.
        base = torch.arange(B_total, device=device)

        if direction == "right":
            # Mark the first frame of each segment → offset = 0 (keep itself).
            starts = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    boundaries[:-1],
                ]
            )  # [0, 3, 8]
            is_boundary = torch.zeros(B_total, dtype=torch.bool, device=device)
            is_boundary[starts] = True

            offset = torch.where(is_boundary, 0, 1)
            source_idx = base - offset
            # e.g. [0, 0, 1,  3, 3, 4, 5, 6,  8, 8]
        else:
            # Mark the last frame of each segment → offset = 0 (keep itself).
            #       boundaries = [3, 8, 10] → ends = [2, 7, 9]
            ends = boundaries - 1
            is_boundary = torch.zeros(B_total, dtype=torch.bool, device=device)
            is_boundary[ends] = True

            offset = torch.where(is_boundary, 0, 1)
            source_idx = base + offset
            # e.g. [1, 2, 2,  4, 5, 6, 7, 7,  9, 9]

        return visual_features[source_idx]


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

    adapter = DINOFrameAdapter(input_dim=D, output_dim=512).cuda()
    adapter.eval()
    with torch.no_grad():
        output = adapter(visual_backbone_output)
        print("Output shape:", output.visual_features.shape)
        print("Patch weights shape:", output.extras["patch_weights"].shape)
        print("Visual length:", output.visual_length)
