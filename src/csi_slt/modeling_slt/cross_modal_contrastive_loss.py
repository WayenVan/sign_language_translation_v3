"""Symmetric contrastive learning for global video and text features.

Computation flow
================

Single process::

    visual [B,D] --> FP32 --> normalize --> V @ T^T * scale --> CE(diagonal) --+-- V2T --+
                                                                                    +--> 0.5 * (V2T + T2V) --> loss
    text   [B,D] --> FP32 --> normalize --> T @ V^T * scale --> CE(diagonal) --+-- T2V --+

Distributed training with ``local_loss=True``::

    local V [b,D] --> normalize --+--> V_query @ T_global^T * scale --> CE(global target) --+-- V2T --+
                                  |                                                              |
                                  +--> differentiable/regular all-gather --> V_global [B,D]       +--> mean --> loss
                                  |                                                              |
    local T [b,D] --> normalize --+--> T_query @ V_global^T * scale --> CE(global target) --+-- T2V --+
                                  +--> differentiable/regular all-gather --> T_global [B,D]

The positive target is the paired sample on the similarity-matrix diagonal.
All other samples in the local/global batch act as negatives. The learnable
``logit_scale = 1 / temperature`` is clamped before it scales cosine logits.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather


class CrossModalContrastiveLoss(nn.Module):
    """CLIP-style loss for paired global video and text representations.

    ``visual_features[i]`` and ``text_features[i]`` must describe the same
    sample. Features are normalized in FP32 before a symmetric video-to-text
    and text-to-video cross-entropy objective is computed.

    When distributed training is initialized, global negatives are gathered by
    default. ``local_loss=True`` keeps only local queries while using features
    from every rank as candidates, avoiding a redundant global-logits matrix on
    every process.

    Args:
        temperature: Initial softmax temperature.
        learnable_temperature: Learn the inverse temperature when ``True``.
        max_logit_scale: Upper bound for the learned inverse temperature.
        gather_distributed: Gather cross-rank features as global negatives.
        gather_with_grad: Preserve gradients through gathered remote features.
        local_loss: Use local queries against global candidates under DDP.
        process_group: Optional distributed process group.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_logit_scale: float = 100.0,
        gather_distributed: bool = True,
        gather_with_grad: bool = False,
        local_loss: bool = True,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_logit_scale <= 0:
            raise ValueError("max_logit_scale must be positive")

        initial_scale = torch.tensor(math.log(1.0 / temperature))
        if learnable_temperature:
            self.logit_scale = nn.Parameter(initial_scale)
        else:
            self.register_buffer("logit_scale", initial_scale)

        self.max_logit_scale = float(max_logit_scale)
        self.gather_distributed = gather_distributed
        self.gather_with_grad = gather_with_grad
        self.local_loss = local_loss
        self.process_group = process_group

    @property
    def temperature(self) -> torch.Tensor:
        """Return the effective (possibly clamped) temperature."""
        return self._scale().reciprocal()

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE from two paired ``[B, D]`` tensors."""
        self._validate_features(visual_features, text_features)

        # Contrastive logits are especially sensitive to reduced-precision
        # normalization and also require both modalities to share a dtype.
        visual_features = F.normalize(visual_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)

        if not self._distributed_enabled():
            self._require_negatives(visual_features.shape[0])
            return self._symmetric_loss(
                visual_queries=visual_features,
                text_queries=text_features,
                visual_candidates=visual_features,
                text_candidates=text_features,
                targets=torch.arange(
                    visual_features.shape[0], device=visual_features.device
                ),
            )

        rank = dist.get_rank(group=self.process_group)
        batch_sizes = self._gather_batch_sizes(
            visual_features.shape[0], visual_features.device
        )
        self._require_negatives(sum(batch_sizes))

        all_visual_features = self._gather_features(
            visual_features, batch_sizes, rank
        )
        all_text_features = self._gather_features(text_features, batch_sizes, rank)

        if self.local_loss:
            targets = torch.arange(
                visual_features.shape[0], device=visual_features.device
            ) + sum(batch_sizes[:rank])
            return self._symmetric_loss(
                visual_queries=visual_features,
                text_queries=text_features,
                visual_candidates=all_visual_features,
                text_candidates=all_text_features,
                targets=targets,
            )

        targets = torch.arange(
            all_visual_features.shape[0], device=visual_features.device
        )
        return self._symmetric_loss(
            visual_queries=all_visual_features,
            text_queries=all_text_features,
            visual_candidates=all_visual_features,
            text_candidates=all_text_features,
            targets=targets,
        )

    def _symmetric_loss(
        self,
        visual_queries: torch.Tensor,
        text_queries: torch.Tensor,
        visual_candidates: torch.Tensor,
        text_candidates: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        scale = self._scale()
        video_to_text = scale * visual_queries @ text_candidates.t()
        text_to_video = scale * text_queries @ visual_candidates.t()
        return 0.5 * (
            F.cross_entropy(video_to_text, targets)
            + F.cross_entropy(text_to_video, targets)
        )

    def _scale(self) -> torch.Tensor:
        max_log_scale = math.log(self.max_logit_scale)
        return self.logit_scale.clamp(max=max_log_scale).exp()

    def _distributed_enabled(self) -> bool:
        return (
            self.gather_distributed
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(group=self.process_group) > 1
        )

    def _gather_batch_sizes(
        self, local_batch_size: int, device: torch.device
    ) -> tuple[int, ...]:
        local_size = torch.tensor(local_batch_size, device=device, dtype=torch.long)
        gathered_sizes = [
            torch.zeros_like(local_size)
            for _ in range(dist.get_world_size(group=self.process_group))
        ]
        dist.all_gather(gathered_sizes, local_size, group=self.process_group)
        return tuple(int(size.item()) for size in gathered_sizes)

    def _gather_features(
        self,
        features: torch.Tensor,
        batch_sizes: tuple[int, ...],
        rank: int,
    ) -> torch.Tensor:
        max_batch_size = max(batch_sizes)
        padded = F.pad(features, (0, 0, 0, max_batch_size - features.shape[0]))

        if self.gather_with_grad and features.requires_grad:
            gathered = differentiable_all_gather(padded, group=self.process_group)
        else:
            gathered = [torch.empty_like(padded) for _ in batch_sizes]
            dist.all_gather(gathered, padded, group=self.process_group)
            # Keep the local feature path differentiable even when remote
            # candidates are detached.
            gathered[rank] = padded

        return torch.cat(
            [rank_features[:size] for rank_features, size in zip(gathered, batch_sizes)],
            dim=0,
        )

    @staticmethod
    def _validate_features(
        visual_features: torch.Tensor, text_features: torch.Tensor
    ) -> None:
        if not isinstance(visual_features, torch.Tensor) or not isinstance(
            text_features, torch.Tensor
        ):
            raise TypeError("visual_features and text_features must be tensors")
        if visual_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError("global features must have shape [batch, dimension]")
        if visual_features.shape != text_features.shape:
            raise ValueError(
                "visual_features and text_features must have identical shapes, "
                f"got {tuple(visual_features.shape)} and "
                f"{tuple(text_features.shape)}"
            )
        if visual_features.shape[0] == 0 or visual_features.shape[1] == 0:
            raise ValueError("global features must be non-empty")
        if visual_features.device != text_features.device:
            raise ValueError("visual_features and text_features must share a device")
        if not visual_features.is_floating_point() or not text_features.is_floating_point():
            raise TypeError("visual_features and text_features must be floating point")

    @staticmethod
    def _require_negatives(global_batch_size: int) -> None:
        if global_batch_size < 2:
            raise ValueError(
                "contrastive learning requires at least two global sample pairs"
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    visual = torch.randn(4, 32, requires_grad=True)
    text = torch.randn(4, 32, requires_grad=True)
    criterion = CrossModalContrastiveLoss()
    loss = criterion(visual, text)
    loss.backward()
    print(f"loss: {loss.item():.6f}")
    print(f"temperature: {criterion.temperature.item():.6f}")
