"""Minimal pseudo-label/video alignment with a learnable NULL target.

This module is intentionally training-only. It aligns video features with
pseudo-label embeddings through semi-unbalanced entropy-regularized optimal
transport, but never reorders or constructs inputs for an inference model.

Shape notation
--------------
B: batch size
M: number of temporal video positions
U: number of pseudo-label positions (padding included)
K: number of OT targets, K = U + 1 (column 0 is NULL)
Dv: input video-feature dimension
Dt: frozen pseudo-label embedding dimension and OT alignment dimension

Why semi-unbalanced OT?
-----------------------
The source marginal is enforced exactly, so every valid video position must
send its mass somewhere. The target marginal is a soft prior rather than an
equality constraint. This is important for NULL: under strictly balanced OT,
subtracting one scalar bias from the entire NULL column cannot change the
optimal plan when that column's total mass is fixed. Relaxing the target
marginal makes ``null_bias`` meaningful and lets NULL usage vary by sample.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
InfoValue = Union[Tensor, float]
AlignmentInfo = Dict[str, InfoValue]


def _normalized_nonnegative_mass(weights: Tensor, name: str) -> Tensor:
    """Normalize non-negative ``weights [B, L]`` to unit mass per sample."""
    if torch.any(weights < 0):
        raise ValueError(f"{name} must be non-negative")
    total = weights.sum(dim=-1, keepdim=True)
    if torch.any(total <= 0):
        raise ValueError(f"every sample must contain positive {name} mass")
    return weights / total


def _safe_log_mass(mass: Tensor) -> Tensor:
    """Return log mass while preserving exact zero-mass padding as ``-inf``."""
    negative_inf = torch.full_like(mass, -torch.inf)
    return torch.where(mass > 0, mass.log(), negative_inf)


def semi_unbalanced_sinkhorn(
    cost: Tensor,
    source_mass: Tensor,
    target_prior: Tensor,
    eps: float = 0.1,
    target_relaxation: float = 0.5,
    n_iters: int = 10,
) -> Tensor:
    """Solve entropy-regularized OT with exact rows and relaxed columns.

    Args:
        cost: Pairwise cost ``[B, M, K]``.
        source_mass: Exact visual marginal ``a [B, M]``.
        target_prior: Soft target-mass prior ``b [B, K]``.
        eps: Entropy coefficient; smaller values make the plan sharper.
        target_relaxation: Strength of the KL penalty that keeps the transported
            target mass near ``target_prior``. Larger values approach balanced
            OT; finite values allow NULL mass to respond to its feature/bias.
        n_iters: Number of alternating log-domain scaling iterations.

    Returns:
        Transport plan ``A [B, M, K]``. Its row sums equal ``source_mass``;
        its column sums are learned but regularized toward ``target_prior``.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if target_relaxation <= 0:
        raise ValueError("target_relaxation must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")

    source_mass = _normalized_nonnegative_mass(source_mass, "source")  # [B, M]
    target_prior = _normalized_nonnegative_mass(target_prior, "target")  # [B, K]

    log_kernel = -cost / eps  # [B, M, K]
    log_a = _safe_log_mass(source_mass)  # [B, M]
    log_b = _safe_log_mass(target_prior)  # [B, K]
    log_v = torch.zeros_like(target_prior)  # [B, K]

    # Generalized Sinkhorn exponent for a KL-relaxed target marginal.
    target_strength = target_relaxation / (target_relaxation + eps)

    for _ in range(n_iters):
        # Exact source update: sum_k A[m,k] = a[m].
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=-1)
        # Relaxed target update: column mass is encouraged, not forced, to match b.
        target_log_mass = torch.logsumexp(
            log_kernel.transpose(1, 2) + log_u.unsqueeze(1), dim=-1
        )
        log_v = target_strength * (log_b - target_log_mass)

    # A final source update makes the visual marginal exact after the last v update.
    log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=-1)
    log_plan = log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(1)  # [B, M, K]
    return torch.exp(log_plan)


def generalized_kl(actual: Tensor, prior: Tensor) -> Tensor:
    """Generalized KL divergence for non-negative masses ``[B, K]``.

    Computes ``sum(actual * log(actual/prior) - actual + prior)`` per sample.
    Zero-prior padding positions are excluded; the OT plan also assigns them
    exactly zero mass.
    """
    valid = prior > 0
    safe_actual = actual.clamp_min(1e-12)
    safe_prior = prior.clamp_min(1e-12)
    terms = safe_actual * (safe_actual.log() - safe_prior.log()) - actual + prior
    return (terms * valid).sum(dim=-1).mean()


class MinimalNullOTAlignment(nn.Module):
    """Align video features to frozen pseudo-label embeddings with NULL support.

    The module returns a scalar training loss and detached diagnostics. The
    detached plan may supervise another training-only branch, but is not
    intended to feed, reorder, or otherwise condition validation/inference.

    Args:
        video_dim: Input video-feature dimension ``Dv``. The input tensor has
            shape ``[B, M, Dv]``. If ``video_dim != text_dim``, the module learns
            one bias-free linear adapter ``Dv -> Dt``; otherwise it uses Identity.
            This adapter only resolves the dimensional/semantic-space mismatch
            and is not the original full ``Dt x Dt`` orthogonal matrix ``T``.
        text_dim: Pseudo-label embedding dimension ``Dt`` and the shared OT
            alignment dimension. Pseudo-label embeddings have shape
            ``[B, U, Dt]`` and are detached internally, so they act as fixed
            semantic anchors while gradients update the visual branch.
        eps: Entropy regularization coefficient of Sinkhorn OT. Larger values
            produce softer, more exploratory transport plans; smaller values
            produce sharper alignments. It is mutable and may be updated by
            ``CosineEpsilonScheduler`` during training.
        n_iters: Number of log-domain Sinkhorn scaling iterations per forward
            pass. More iterations improve marginal convergence but cost more
            compute. Ten iterations follows the VTaMo configuration.
        target_relaxation: Strength ``tau_b`` of the KL penalty on the transported
            target marginal. Larger values keep target mass closer to the prior
            and approach balanced OT; smaller values allow actual NULL/label mass
            to adapt more freely. It must remain finite for a scalar NULL bias to
            influence the total NULL mass.
        null_mass_prior: Prior fraction ``rho_null`` of target transport mass
            assigned to NULL before observing the cost. Unlike ``1/(U+1)``, this
            value is independent of pseudo-label sequence length. In the
            semi-unbalanced solver it is a soft prior, not a fixed column sum.
        null_ratio_max: Maximum preferred fraction of valid video positions for
            which NULL is the local soft winner. The one-sided NULL regularizer
            is zero below this threshold and quadratic above it.
        null_temperature: Temperature used only for the row-wise cost softmax
            that estimates ``P(NULL | video position)``. Lower values approximate
            a hard argmin over target costs; higher values give a softer estimate.
        beta_ot: Weight of the semi-unbalanced OT objective. That objective is
            the unbiased transport cost plus ``target_relaxation`` times the
            generalized KL divergence between realized and prior target mass.
        beta_null: Weight of the one-sided NULL-ratio regularization. Increasing
            it more strongly prevents the learnable NULL token/bias from becoming
            a universal low-cost escape route.
        beta_tv: Weight of the temporal-variation loss on the row-normalized
            real-token plan. It penalizes adjacent video positions flipping
            between tokens, keeping the learned alignment temporally coherent.
    """

    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        eps: float = 0.12,
        n_iters: int = 10,
        target_relaxation: float = 0.5,
        null_mass_prior: float = 0.2,
        null_ratio_max: float = 0.2,
        null_temperature: float = 0.1,
        beta_ot: float = 1.0,
        beta_null: float = 0.1,
        beta_tv: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 < null_mass_prior < 1:
            raise ValueError("null_mass_prior must lie in (0, 1)")
        if not 0 <= null_ratio_max <= 1:
            raise ValueError("null_ratio_max must lie in [0, 1]")
        if null_temperature <= 0:
            raise ValueError("null_temperature must be positive")
        if beta_ot < 0 or beta_null < 0 or beta_tv < 0:
            raise ValueError("beta weights must be non-negative")

        self.eps = float(eps)
        self.n_iters = int(n_iters)
        self.target_relaxation = float(target_relaxation)
        self.null_mass_prior = float(null_mass_prior)
        self.null_ratio_max = float(null_ratio_max)
        self.null_temperature = float(null_temperature)
        self.beta_ot = float(beta_ot)
        self.beta_null = float(beta_null)
        self.beta_tv = float(beta_tv)

        # This is only a dimension adapter. There is no D x D orthogonal T.
        self.video_proj: nn.Module
        if video_dim == text_dim:
            self.video_proj = nn.Identity()
        else:
            self.video_proj = nn.Linear(video_dim, text_dim, bias=False)

        self.null_token = nn.Parameter(torch.randn(text_dim) * 0.02)  # [Dt]
        self.null_bias = nn.Parameter(torch.tensor(0.0))  # scalar NULL cost offset

    def _build_target_prior(
        self,
        pseudo_mask: Tensor,
        pseudo_confidence: Optional[Tensor],
        dtype: torch.dtype,
    ) -> Tensor:
        """Build ``b [B, U+1]`` with an explicit, length-independent NULL prior."""
        real_weights = pseudo_mask.to(dtype)  # [B, U]
        if pseudo_confidence is not None:
            real_weights = real_weights * pseudo_confidence.to(dtype).clamp_min(0)

        real_total = real_weights.sum(dim=-1, keepdim=True)  # [B, 1]
        has_real_target = real_total > 0
        real_distribution = real_weights / real_total.clamp_min(1e-12)  # [B, U]

        batch_size = pseudo_mask.shape[0]
        null_prior = torch.full(
            (batch_size, 1),
            self.null_mass_prior,
            device=pseudo_mask.device,
            dtype=dtype,
        )  # [B, 1]

        # If a sample has no usable pseudo-label, route its entire target prior to NULL.
        null_prior = torch.where(has_real_target, null_prior, torch.ones_like(null_prior))
        real_mass = (1.0 - null_prior) * real_distribution  # [B, U]
        return torch.cat((null_prior, real_mass), dim=1)  # [B, U+1]

    def _compute_tv_loss(
        self, alignment: torch.Tensor, video_mask: torch.Tensor
    ) -> torch.Tensor:
        """Temporal-variation loss on the row-normalized real-token plan.

        Implements the VTaMo temporal-variation term (paper Eq.(6)):

        ``L_tv = (1 / ((M-1) * U)) * sum_{m,k} |A_hat[m+1,k] - A_hat[m,k]|``

        where ``A_hat`` is the transport plan row-normalized over the real
        pseudo-label columns (NULL excluded). Only adjacent video positions
        where BOTH are valid contribute. Row-normalization follows the paper
        and makes the penalty scale-invariant to per-position transported mass;
        the mask excludes padding/boundary pairs.

        Args:
            alignment: Transport plan ``[B, M, U+1]`` (column 0 is NULL).
            video_mask: Valid video positions ``[B, M]``.

        Returns:
            Scalar TV loss.
        """
        # Exclude the NULL column; keep only real pseudo-label columns.
        real_plan = alignment[:, :, 1:]  # [B, M, U]
        # Row-normalize over real tokens -> A_hat (paper Eq.(6)).
        real_plan = real_plan / real_plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # Adjacent-row absolute difference: [B, M-1, U].
        diff = (real_plan[:, 1:, :] - real_plan[:, :-1, :]).abs()
        # Only count pairs where both video positions are valid.
        pair_mask = (video_mask[:, 1:] * video_mask[:, :-1]).unsqueeze(-1)  # [B, M-1, 1]
        num_pairs = pair_mask.sum().clamp_min(1)
        return (diff * pair_mask).sum() / (num_pairs * real_plan.shape[-1])

    def forward(
        self,
        video_features: Tensor,
        pseudo_embeddings: Tensor,
        video_mask: Tensor,
        pseudo_mask: Tensor,
        pseudo_confidence: Optional[Tensor] = None,
    ) -> Tuple[Tensor, AlignmentInfo]:
        """Compute semi-unbalanced OT alignment and NULL regularization.

        Args:
            video_features: Temporal video features ``[B, M, Dv]``.
            pseudo_embeddings: Pseudo-label embeddings ``[B, U, Dt]``. They
                are detached internally and act as fixed semantic anchors.
            video_mask: Valid video positions ``[B, M]``.
            pseudo_mask: Valid pseudo-label positions ``[B, U]``.
            pseudo_confidence: Optional non-negative label reliability ``[B, U]``.

        Returns:
            total_loss: ``beta_ot * L_uot + beta_tv * L_tv + beta_null * L_null``.
            info: Detached component losses, NULL statistics, and alignment
                ``[B, M, U+1]`` for logging/visualization only.
        """
        batch_size, _, _ = video_features.shape
        _, _, text_dim = pseudo_embeddings.shape
        dtype, device = video_features.dtype, video_features.device

        projected_video = self.video_proj(video_features)  # [B, M, Dt]
        video_unit = F.normalize(projected_video, dim=-1)  # [B, M, Dt]

        # Stop pseudo-label gradients; only video features/projection and NULL learn.
        pseudo_unit = F.normalize(pseudo_embeddings.detach().to(dtype), dim=-1)  # [B, U, Dt]
        null_unit = F.normalize(self.null_token, dim=0)  # [Dt]
        null_unit = null_unit.view(1, 1, text_dim).expand(batch_size, 1, text_dim)
        targets = torch.cat((null_unit, pseudo_unit), dim=1)  # [B, U+1, Dt]

        similarity = torch.bmm(video_unit, targets.transpose(1, 2))  # [B, M, U+1]
        semantic_cost = 1.0 - similarity  # [B, M, U+1], unbiased cosine distance
        ot_cost = semantic_cost.clone()
        ot_cost[:, :, 0] = semantic_cost[:, :, 0] - self.null_bias

        # Padding targets must be unavailable both to OT and the NULL winner softmax.
        padded_target = ~pseudo_mask.bool()  # [B, U]
        ot_cost[:, :, 1:].masked_fill_(padded_target.unsqueeze(1), 1e4)

        source_mass = _normalized_nonnegative_mass(video_mask.to(dtype), "source")  # [B, M]
        target_prior = self._build_target_prior(
            pseudo_mask, pseudo_confidence, dtype
        )  # [B, U+1]

        alignment = semi_unbalanced_sinkhorn(
            cost=ot_cost.float(),
            source_mass=source_mass.float(),
            target_prior=target_prior.float(),
            eps=self.eps,
            target_relaxation=self.target_relaxation,
            n_iters=self.n_iters,
        ).to(dtype)  # [B, M, U+1]

        # EMD-style transport cost: A weights pairwise cosine distances directly.
        transport_loss = (alignment * semantic_cost).sum(dim=(1, 2)).mean()

        # The target marginal is soft; regularize its realized mass toward the prior.
        transported_target_mass = alignment.sum(dim=1)  # [B, U+1]
        target_mass_kl = generalized_kl(
            transported_target_mass.float(), target_prior.float()
        ).to(dtype)
        ot_loss = transport_loss + self.target_relaxation * target_mass_kl

        # A[m,0] is transport mass, not a row probability. Estimate local NULL
        # preference directly from the biased cost with a row-wise softmax.
        winner_logits = -ot_cost / self.null_temperature  # [B, M, U+1]
        log_normalizer = torch.logsumexp(winner_logits, dim=-1)  # [B, M]
        null_probability = torch.exp(winner_logits[:, :, 0] - log_normalizer)  # [B, M]

        valid_video = video_mask.to(dtype)  # [B, M]
        soft_null_per_sample = (null_probability * valid_video).sum(dim=-1)
        soft_null_per_sample = soft_null_per_sample / valid_video.sum(dim=-1).clamp_min(1)
        null_reg_loss = torch.relu(soft_null_per_sample - self.null_ratio_max).square().mean()

        # Temporal variation: keep adjacent video positions from flipping tokens.
        tv_loss = self._compute_tv_loss(alignment, video_mask)

        total_loss = (
            self.beta_ot * ot_loss
            + self.beta_tv * tv_loss
            + self.beta_null * null_reg_loss
        )

        hard_null = alignment.argmax(dim=-1).eq(0).to(dtype)  # [B, M]
        hard_null_ratio = (hard_null * valid_video).sum() / valid_video.sum().clamp_min(1)
        actual_null_mass = transported_target_mass[:, 0].mean()  # scalar diagnostic

        info: AlignmentInfo = {
            "ot_loss": ot_loss.detach(),
            "transport_loss": transport_loss.detach(),
            "target_mass_kl": target_mass_kl.detach(),
            "tv_loss": tv_loss.detach(),
            "null_reg_loss": null_reg_loss.detach(),
            "soft_null_ratio": soft_null_per_sample.mean().detach(),
            "hard_null_ratio": hard_null_ratio.detach(),
            "actual_null_mass": actual_null_mass.detach(),
            "null_bias": self.null_bias.detach(),
            "epsilon": self.eps,
            # Detached by design: diagnostics or a training-only teacher, never
            # a validation/inference input.
            "alignment": alignment.detach(),  # [B, M, U+1]
        }
        return total_loss, info


class CosineEpsilonScheduler:
    """Cosine-anneal OT entropy, then hold a sharp alignment value.

    For the first ``anneal_ratio`` fraction of optimizer steps, epsilon follows

    ``eps_min + 0.5 * (eps_max - eps_min) * (1 + cos(pi * progress))``.

    Once annealing is complete, epsilon remains at ``eps_min``. The default
    therefore performs ``0.12 -> 0.03`` over the first 80% of training and keeps
    ``0.03`` for the final 20%. Cosine interpolation has zero slope at both ends,
    avoiding the discontinuities of the original multi-stage schedule.

    Args:
        alignment_module: ``MinimalNullOTAlignment`` instance whose mutable
            ``eps`` attribute is updated in place.
        total_steps: Total number of optimizer steps, after accounting for
            gradient accumulation. Use Lightning's estimated stepping batches
            rather than the number of raw dataloader batches.
        eps_max: Initial high-entropy value for soft exploratory alignment.
        eps_min: Final low-entropy value for sharp alignment.
        anneal_ratio: Fraction of total optimizer steps used for cosine
            annealing. The remaining fraction holds ``eps_min`` fixed.

    Call ``step(global_step)`` once per optimizer step. ``state_dict`` and
    ``load_state_dict`` support checkpoint resumption.
    """

    def __init__(
        self,
        alignment_module: MinimalNullOTAlignment,
        total_steps: int,
        eps_max: float = 0.12,
        eps_min: float = 0.03,
        anneal_ratio: float = 0.8,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not eps_max >= eps_min > 0:
            raise ValueError("expected eps_max >= eps_min > 0")
        if not 0 < anneal_ratio <= 1:
            raise ValueError("anneal_ratio must lie in (0, 1]")

        self.alignment_module = alignment_module
        self.total_steps = int(total_steps)
        self.anneal_steps = max(1, int(total_steps * anneal_ratio))
        self.eps_max = float(eps_max)
        self.eps_min = float(eps_min)
        self.anneal_ratio = float(anneal_ratio)
        self.current_step = 0

        self.alignment_module.eps = self.eps_max

    def value_at(self, step: int) -> float:
        """Return epsilon at optimizer ``step`` without changing module state."""
        if step < 0:
            raise ValueError("step must be non-negative")

        progress = min(step / self.anneal_steps, 1.0)
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eps_min + (self.eps_max - self.eps_min) * cosine_weight

    def step(self, step: Optional[int] = None) -> float:
        """Update ``alignment_module.eps`` and return its new value."""
        self.current_step = self.current_step + 1 if step is None else int(step)
        epsilon = self.value_at(self.current_step)
        self.alignment_module.eps = epsilon
        return epsilon

    def state_dict(self) -> Dict[str, int]:
        """Return the minimal state required to resume annealing."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """Restore the step counter and synchronize the module epsilon."""
        if "current_step" not in state_dict:
            raise KeyError("scheduler state_dict is missing 'current_step'")
        self.step(int(state_dict["current_step"]))


__all__ = [
    "CosineEpsilonScheduler",
    "MinimalNullOTAlignment",
    "generalized_kl",
    "semi_unbalanced_sinkhorn",
]
