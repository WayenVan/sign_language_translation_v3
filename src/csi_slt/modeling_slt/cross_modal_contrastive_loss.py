"""Symmetric contrastive learning for global video and text features.

Computation flow
================

Single process::

    visual [B,D] --> FP32 --> normalize --> V @ [T; text queue]^T --> MP-NCE --+-- V2T --+
                                                                                        +--> 0.5 * (V2T + T2V) --> loss
    text   [B,D] --> FP32 --> normalize --> T @ V^T * scale --> MP-NCE --+-- T2V --+

Distributed training with ``local_loss=True``::

    local V [b,D] --> normalize --+--> V_query @ T_global^T * scale --> CE(global target) --+-- V2T --+
                                  |                                                              |
                                  +--> differentiable/regular all-gather --> V_global [B,D]       +--> mean --> loss
                                  |                                                              |
    local T [b,D] --> normalize --+--> T_query @ V_global^T * scale --> CE(global target) --+-- T2V --+
                                  +--> differentiable/regular all-gather --> T_global [B,D]

When semantic IDs are supplied, every candidate sharing the query ID is a
positive; other IDs are negatives. Without IDs, the original paired-diagonal
cross-entropy behavior is retained. The learnable ``logit_scale = 1 /
temperature`` is clamped before it scales cosine logits.
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
        text_queue_size: Number of historical text features used as additional
            video-to-text negatives. Zero disables the queue.
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
        text_queue_size: int = 0,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_logit_scale <= 0:
            raise ValueError("max_logit_scale must be positive")
        if text_queue_size < 0:
            raise ValueError("text_queue_size must be non-negative")

        initial_scale = torch.tensor(math.log(1.0 / temperature))
        if learnable_temperature:
            self.logit_scale = nn.Parameter(initial_scale)
        else:
            self.register_buffer("logit_scale", initial_scale)

        self.max_logit_scale = float(max_logit_scale)
        self.gather_distributed = gather_distributed
        self.gather_with_grad = gather_with_grad
        self.local_loss = local_loss
        self.text_queue_size = text_queue_size
        self.process_group = process_group
        self.register_buffer("text_queue", torch.empty(0, 0), persistent=False)
        self.register_buffer(
            "text_queue_ids", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "text_queue_ptr", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "text_queue_count", torch.zeros((), dtype=torch.long), persistent=False
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Return the effective (possibly clamped) temperature."""
        return self._scale().reciprocal()

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        semantic_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute symmetric multi-positive InfoNCE from paired features."""
        self._validate_features(visual_features, text_features)
        self._validate_semantic_ids(semantic_ids, visual_features)

        # Contrastive logits are especially sensitive to reduced-precision
        # normalization and also require both modalities to share a dtype.
        visual_features = F.normalize(visual_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)

        if not self._distributed_enabled():
            if visual_features.shape[0] < 2:
                # No positive/negative comparison is possible; retain a valid
                # zero-gradient graph for batches with missing pseudo-glosses.
                return (
                    visual_features.sum() + text_features.sum() + self._scale()
                ) * 0.0
            loss = self._symmetric_loss(
                visual_queries=visual_features,
                text_queries=text_features,
                visual_candidates=visual_features,
                text_candidates=text_features,
                targets=torch.arange(
                    visual_features.shape[0], device=visual_features.device
                ),
                query_ids=semantic_ids,
                candidate_ids=semantic_ids,
            )
            self._enqueue_text_features(text_features, semantic_ids)
            return loss

        rank = dist.get_rank(group=self.process_group)
        batch_sizes = self._gather_batch_sizes(
            visual_features.shape[0], visual_features.device
        )
        global_batch_size = sum(batch_sizes)
        if global_batch_size < 2:
            # Every rank observes the same count, so all ranks safely take this
            # branch after participating in the size collective.
            return (
                visual_features.sum() + text_features.sum() + self._scale()
            ) * 0.0

        all_visual_features = self._gather_features(
            visual_features, batch_sizes, rank
        )
        all_text_features = self._gather_features(text_features, batch_sizes, rank)
        all_semantic_ids = (
            self._gather_ids(semantic_ids, batch_sizes)
            if semantic_ids is not None
            else None
        )

        if self.local_loss:
            if visual_features.shape[0] == 0:
                loss = (
                    all_visual_features.sum()
                    + all_text_features.sum()
                    + self._scale()
                ) * 0.0
            else:
                targets = torch.arange(
                    visual_features.shape[0], device=visual_features.device
                ) + sum(batch_sizes[:rank])
                loss = self._symmetric_loss(
                    visual_queries=visual_features,
                    text_queries=text_features,
                    visual_candidates=all_visual_features,
                    text_candidates=all_text_features,
                    targets=targets,
                    query_ids=semantic_ids,
                    candidate_ids=all_semantic_ids,
                )
                # DDP averages ranks; reweight uneven valid counts to recover
                # an average over valid samples rather than over ranks.
                loss = loss * dist.get_world_size(group=self.process_group)
                loss = loss * visual_features.shape[0] / global_batch_size
        else:
            targets = torch.arange(
                all_visual_features.shape[0], device=visual_features.device
            )
            loss = self._symmetric_loss(
                visual_queries=all_visual_features,
                text_queries=all_text_features,
                visual_candidates=all_visual_features,
                text_candidates=all_text_features,
                targets=targets,
                query_ids=all_semantic_ids,
                candidate_ids=all_semantic_ids,
            )

        # Every rank gathered features in the same rank order, so enqueuing the
        # global batch keeps all per-rank queues synchronized without another
        # collective operation.
        self._enqueue_text_features(all_text_features, all_semantic_ids)
        return loss

    def _symmetric_loss(
        self,
        visual_queries: torch.Tensor,
        text_queries: torch.Tensor,
        visual_candidates: torch.Tensor,
        text_candidates: torch.Tensor,
        targets: torch.Tensor,
        query_ids: torch.Tensor | None = None,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self._scale()
        queued_text = self._queued_text_features(
            feature_dim=text_candidates.shape[-1], device=text_candidates.device
        )
        if queued_text.numel() > 0:
            video_to_text_candidates = torch.cat(
                (text_candidates, queued_text), dim=0
            )
        else:
            video_to_text_candidates = text_candidates

        video_to_text = scale * visual_queries @ video_to_text_candidates.t()
        text_to_video = scale * text_queries @ visual_candidates.t()
        if query_ids is not None:
            if candidate_ids is None:
                raise ValueError("candidate_ids are required when query_ids are provided")
            queued_ids = self._queued_text_ids(device=candidate_ids.device)
            video_to_text_ids = (
                torch.cat((candidate_ids, queued_ids), dim=0)
                if queued_ids.numel() > 0
                else candidate_ids
            )
            return 0.5 * (
                self._multi_positive_nce(
                    video_to_text, query_ids[:, None].eq(video_to_text_ids[None, :])
                )
                + self._multi_positive_nce(
                    text_to_video, query_ids[:, None].eq(candidate_ids[None, :])
                )
            )
        return 0.5 * (
            F.cross_entropy(video_to_text, targets)
            + F.cross_entropy(text_to_video, targets)
        )

    def _queued_text_features(
        self, feature_dim: int, device: torch.device
    ) -> torch.Tensor:
        if (
            not self.training
            or self.text_queue_size == 0
            or int(self.text_queue_count.item()) == 0
        ):
            return torch.empty(0, feature_dim, dtype=torch.float32, device=device)
        return self.text_queue[: int(self.text_queue_count.item())]

    def _queued_text_ids(self, device: torch.device) -> torch.Tensor:
        if (
            not self.training
            or self.text_queue_size == 0
            or int(self.text_queue_count.item()) == 0
        ):
            return torch.empty(0, dtype=torch.long, device=device)
        return self.text_queue_ids[: int(self.text_queue_count.item())]

    @staticmethod
    def _multi_positive_nce(
        logits: torch.Tensor, positive_mask: torch.Tensor
    ) -> torch.Tensor:
        if not bool(positive_mask.any(dim=-1).all()):
            raise ValueError("every contrastive query must have at least one positive")
        positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
        return (
            torch.logsumexp(logits, dim=-1)
            - torch.logsumexp(positive_logits, dim=-1)
        ).mean()

    @torch.no_grad()
    def _enqueue_text_features(
        self,
        text_features: torch.Tensor,
        semantic_ids: torch.Tensor | None = None,
    ) -> None:
        if not self.training or self.text_queue_size == 0:
            return

        features = text_features.detach().float()
        ids = (
            semantic_ids.detach().long()
            if semantic_ids is not None
            else torch.full(
                (features.shape[0],),
                torch.iinfo(torch.long).min,
                dtype=torch.long,
                device=features.device,
            )
        )
        if self.text_queue.shape != (self.text_queue_size, features.shape[-1]):
            self.text_queue = torch.zeros(
                self.text_queue_size,
                features.shape[-1],
                dtype=torch.float32,
                device=features.device,
            )
            self.text_queue_ids = torch.full(
                (self.text_queue_size,),
                torch.iinfo(torch.long).min,
                dtype=torch.long,
                device=features.device,
            )
            self.text_queue_ptr.zero_()
            self.text_queue_count.zero_()

        if features.shape[0] >= self.text_queue_size:
            self.text_queue.copy_(features[-self.text_queue_size :])
            self.text_queue_ids.copy_(ids[-self.text_queue_size :])
            self.text_queue_ptr.zero_()
            self.text_queue_count.fill_(self.text_queue_size)
            return

        count = features.shape[0]
        ptr = int(self.text_queue_ptr.item())
        first_count = min(count, self.text_queue_size - ptr)
        self.text_queue[ptr : ptr + first_count].copy_(features[:first_count])
        self.text_queue_ids[ptr : ptr + first_count].copy_(ids[:first_count])
        remaining = count - first_count
        if remaining:
            self.text_queue[:remaining].copy_(features[first_count:])
            self.text_queue_ids[:remaining].copy_(ids[first_count:])

        self.text_queue_ptr.fill_((ptr + count) % self.text_queue_size)
        self.text_queue_count.fill_(
            min(self.text_queue_size, int(self.text_queue_count.item()) + count)
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

    def _gather_ids(
        self, semantic_ids: torch.Tensor, batch_sizes: tuple[int, ...]
    ) -> torch.Tensor:
        max_batch_size = max(batch_sizes)
        padded = F.pad(
            semantic_ids,
            (0, max_batch_size - semantic_ids.shape[0]),
            value=torch.iinfo(torch.long).min,
        )
        gathered = [torch.empty_like(padded) for _ in batch_sizes]
        dist.all_gather(gathered, padded, group=self.process_group)
        return torch.cat(
            [rank_ids[:size] for rank_ids, size in zip(gathered, batch_sizes)], dim=0
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
        if visual_features.shape[1] == 0:
            raise ValueError("global feature dimension must be non-empty")
        if visual_features.device != text_features.device:
            raise ValueError("visual_features and text_features must share a device")
        if not visual_features.is_floating_point() or not text_features.is_floating_point():
            raise TypeError("visual_features and text_features must be floating point")

    @staticmethod
    def _validate_semantic_ids(
        semantic_ids: torch.Tensor | None, visual_features: torch.Tensor
    ) -> None:
        if semantic_ids is None:
            return
        if semantic_ids.ndim != 1 or semantic_ids.shape[0] != visual_features.shape[0]:
            raise ValueError("semantic_ids must have shape [batch]")
        if semantic_ids.dtype != torch.long:
            raise TypeError("semantic_ids must use torch.long")
        if semantic_ids.device != visual_features.device:
            raise ValueError("semantic_ids and features must share a device")

if __name__ == "__main__":
    torch.manual_seed(42)
    visual = torch.randn(4, 32, requires_grad=True)
    text = torch.randn(4, 32, requires_grad=True)
    criterion = CrossModalContrastiveLoss()
    loss = criterion(visual, text)
    loss.backward()
    print(f"loss: {loss.item():.6f}")
    print(f"temperature: {criterion.temperature.item():.6f}")


# local_loss=True example
# =======================
#
# Two ranks, local batch size 2, global batch size 4, text queue size 3.
# The diagrams show rank 1, whose local positive targets are [2, 3].
#
# Video -> Text (text queue is used)
#
#                         Current global text                  Text queue
#                  T0       T1       T2       T3       Q0       Q1       Q2
#               +--------+--------+--------+--------+--------+--------+--------+
# V2 (rank 1)   |  neg   |  neg   |  POS   |  neg   |  neg   |  neg   |  neg   |
#               +--------+--------+--------+--------+--------+--------+--------+
# V3 (rank 1)   |  neg   |  neg   |  neg   |  POS   |  neg   |  neg   |  neg   |
#               +--------+--------+--------+--------+--------+--------+--------+
#
# logits shape: [local_batch, global_batch + text_queue_size] = [2, 7]
# targets: [2, 3]
#
# Text -> Video (text queue is not used)
#
#                         Current global visual
#                  V0       V1       V2       V3
#               +--------+--------+--------+--------+
# T2 (rank 1)   |  neg   |  neg   |  POS   |  neg   |
#               +--------+--------+--------+--------+
# T3 (rank 1)   |  neg   |  neg   |  neg   |  POS   |
#               +--------+--------+--------+--------+
#
# logits shape: [local_batch, global_batch] = [2, 4]
# targets: [2, 3]
