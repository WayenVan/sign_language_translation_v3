"""Direction-aware Source-Influence Distillation loss utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DSIDLossOutput:
    """D-SID loss and scalar diagnostics over valid target positions."""

    loss: torch.Tensor
    mean_js: torch.Tensor
    gate_coverage: torch.Tensor
    mean_weight: torch.Tensor
    mean_kl: torch.Tensor
    teacher_nll_gain: torch.Tensor


@dataclass(frozen=True)
class DSIDTeacherStatistics:
    """Per-target-token statistics from the two frozen teacher paths."""

    js: torch.Tensor
    direction_gate: torch.Tensor
    gloss_nll: torch.Tensor
    empty_nll: torch.Tensor
    target_ids: torch.Tensor
    valid_counts: torch.Tensor

    @property
    def teacher_nll_gain(self) -> torch.Tensor:
        """Positive values mean pseudo-gloss lowers target-token NLL."""
        return self.empty_nll - self.gloss_nll


def _select_target_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    path_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select next-token predictions identified by one path's label mask.

    Source spans can have different lengths, so absolute sequence positions are
    not comparable across paths. Each path is shifted independently and its
    valid target predictions are returned in batch/target order.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"{path_name} logits must have shape [B, L, V], got {tuple(logits.shape)}"
        )
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"{path_name} labels shape {tuple(labels.shape)} must match logits "
            f"batch and sequence dimensions {tuple(logits.shape[:2])}"
        )
    if logits.size(1) < 2:
        raise ValueError(f"{path_name} sequence length must be at least 2")
    if labels.device != logits.device:
        raise ValueError(f"{path_name} labels and logits must be on the same device")
    if bool(labels[:, 0].ne(-100).any()):
        raise ValueError(
            f"{path_name} labels cannot supervise the first sequence position"
        )

    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100)
    valid_counts = valid_mask.sum(dim=-1)
    if not bool(valid_counts.sum()):
        raise ValueError(f"{path_name} contains no valid target positions")

    target_logits = logits[:, :-1][valid_mask]
    target_ids = shifted_labels[valid_mask]
    return target_logits, target_ids, valid_counts


def _validate_corresponding_targets(
    reference_targets: torch.Tensor,
    reference_counts: torch.Tensor,
    compared_targets: torch.Tensor,
    compared_counts: torch.Tensor,
    *,
    reference_name: str,
    compared_name: str,
) -> None:
    if not torch.equal(compared_counts, reference_counts):
        raise ValueError(
            f"{compared_name} target counts {compared_counts.tolist()} do not "
            f"match {reference_name} target counts {reference_counts.tolist()}"
        )
    if not torch.equal(compared_targets, reference_targets):
        raise ValueError(
            f"{compared_name} target token IDs do not align with {reference_name}"
        )


def _compute_dsid_teacher_statistics(
    gloss_teacher_logits: torch.Tensor,
    gloss_teacher_labels: torch.Tensor,
    empty_teacher_logits: torch.Tensor,
    empty_teacher_labels: torch.Tensor,
) -> tuple[DSIDTeacherStatistics, torch.Tensor, torch.Tensor]:
    gloss_target_logits, gloss_targets, gloss_counts = _select_target_predictions(
        gloss_teacher_logits,
        gloss_teacher_labels,
        path_name="pseudo-gloss teacher",
    )
    empty_target_logits, empty_targets, empty_counts = _select_target_predictions(
        empty_teacher_logits,
        empty_teacher_labels,
        path_name="empty-source teacher",
    )
    if empty_target_logits.size(-1) != gloss_target_logits.size(-1):
        raise ValueError(
            f"empty-source teacher vocabulary size {empty_target_logits.size(-1)} "
            "does not match pseudo-gloss teacher vocabulary size "
            f"{gloss_target_logits.size(-1)}"
        )
    _validate_corresponding_targets(
        gloss_targets,
        gloss_counts,
        empty_targets,
        empty_counts,
        reference_name="pseudo-gloss teacher",
        compared_name="empty-source teacher",
    )

    # Keep probability-space calculations stable under bf16/fp16 model inference.
    # Both inputs are detached so these statistics can never train either path.
    gloss_log_probs = F.log_softmax(gloss_target_logits.detach().float(), dim=-1)
    empty_log_probs = F.log_softmax(empty_target_logits.detach().float(), dim=-1)
    gloss_probs = gloss_log_probs.exp()
    empty_probs = empty_log_probs.exp()
    log_mixture = torch.logaddexp(gloss_log_probs, empty_log_probs) - torch.log(
        gloss_log_probs.new_tensor(2.0)
    )
    js = 0.5 * (
        (gloss_probs * (gloss_log_probs - log_mixture)).sum(dim=-1)
        + (empty_probs * (empty_log_probs - log_mixture)).sum(dim=-1)
    )

    gold_indices = gloss_targets.unsqueeze(-1)
    gloss_nll = -gloss_log_probs.gather(-1, gold_indices).squeeze(-1)
    empty_nll = -empty_log_probs.gather(-1, gold_indices).squeeze(-1)
    statistics = DSIDTeacherStatistics(
        js=js.clamp_min(0.0),
        direction_gate=gloss_nll.lt(empty_nll),
        gloss_nll=gloss_nll,
        empty_nll=empty_nll,
        target_ids=gloss_targets,
        valid_counts=gloss_counts,
    )
    return statistics, gloss_log_probs, gloss_probs


def compute_dsid_teacher_statistics(
    gloss_teacher_logits: torch.Tensor,
    gloss_teacher_labels: torch.Tensor,
    empty_teacher_logits: torch.Tensor,
    empty_teacher_labels: torch.Tensor,
) -> DSIDTeacherStatistics:
    """Compute calibration statistics over aligned valid target positions."""
    statistics, _, _ = _compute_dsid_teacher_statistics(
        gloss_teacher_logits,
        gloss_teacher_labels,
        empty_teacher_logits,
        empty_teacher_labels,
    )
    return statistics


def compute_dsid_loss(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    gloss_teacher_logits: torch.Tensor,
    gloss_teacher_labels: torch.Tensor,
    empty_teacher_logits: torch.Tensor,
    empty_teacher_labels: torch.Tensor,
    *,
    js_tau: float,
) -> DSIDLossOutput:
    """Compute full-vocabulary D-SID over aligned target-token decisions.

    The two teacher distributions and the direction-aware weights are detached.
    Consequently, gradients can flow only through ``student_logits``.
    """
    if js_tau < 1e-8:
        raise ValueError(f"js_tau must be at least 1e-8, got {js_tau}")

    student_target_logits, student_targets, student_counts = _select_target_predictions(
        student_logits,
        student_labels,
        path_name="video student",
    )
    statistics, gloss_log_probs, gloss_probs = _compute_dsid_teacher_statistics(
        gloss_teacher_logits,
        gloss_teacher_labels,
        empty_teacher_logits,
        empty_teacher_labels,
    )
    if gloss_log_probs.size(-1) != student_target_logits.size(-1):
        raise ValueError(
            f"pseudo-gloss teacher vocabulary size {gloss_log_probs.size(-1)} "
            "does not match video student vocabulary size "
            f"{student_target_logits.size(-1)}"
        )
    _validate_corresponding_targets(
        student_targets,
        student_counts,
        statistics.target_ids,
        statistics.valid_counts,
        reference_name="video student",
        compared_name="pseudo-gloss teacher",
    )

    # The probability-space calculations deliberately run in float32 even when
    # model logits use bf16/fp16. Teacher logits are detached by construction so
    # D-SID cannot update either teacher path.
    student_log_probs = F.log_softmax(student_target_logits.float(), dim=-1)
    weights = (
        (statistics.js / float(js_tau)).clamp(max=1.0)
        * statistics.direction_gate.to(dtype=statistics.js.dtype)
    ).detach()

    token_kl = (
        (gloss_probs * (gloss_log_probs - student_log_probs)).sum(dim=-1).clamp_min(0.0)
    )
    valid_token_count = student_targets.numel()
    loss = (weights * token_kl).sum() / valid_token_count

    return DSIDLossOutput(
        loss=loss,
        mean_js=statistics.js.mean(),
        gate_coverage=statistics.direction_gate.float().mean(),
        mean_weight=weights.mean(),
        mean_kl=token_kl.mean(),
        teacher_nll_gain=statistics.teacher_nll_gain.mean(),
    )
