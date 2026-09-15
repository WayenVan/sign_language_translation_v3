"""Skip the optimizer step on rare, extreme gradient-norm spikes.

A spike here is a pre-clip global gradient norm far above the recent median.
Global norm clipping cannot neutralize one: it rescales the whole gradient, so a
spike concentrated in a few tensors still reaches their Adam moments at many
times the usual size and keeps pushing through momentum for ~10 steps. Skipping
the step entirely keeps the anomalous gradient out of the parameters and out of
both moments (the Megatron-style skip, extended from non-finite to outlier
norms).

The guard is deliberately narrow: it only acts on isolated outliers, and after
``max_consecutive`` skips it accepts the next outlier so a genuine shift in the
gradient scale re-bases the median instead of stalling training.
"""

from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class SpikeDecision:
    skip: bool
    # None for an ordinary step; otherwise "non_finite", "spike", or
    # "consecutive_limit" (an outlier accepted because the limit was reached).
    reason: Optional[str]
    grad_norm: float
    median: Optional[float]


class SpikeGuard:
    """Median-ratio outlier detector over recently accepted gradient norms.

    The history lives in memory only; a resumed run simply re-warms it over
    ``min_history`` steps, during which nothing finite is skipped.
    """

    def __init__(
        self,
        factor: float,
        window: int = 100,
        min_history: int = 50,
        max_consecutive: int = 2,
    ) -> None:
        if not factor > 1.0:
            raise ValueError(f"spike_skip_factor must be > 1, got {factor}")
        if window < 1:
            raise ValueError(f"spike_skip_window must be >= 1, got {window}")
        if not 1 <= min_history <= window:
            raise ValueError(
                "spike_skip_min_history must be in [1, spike_skip_window], "
                f"got {min_history} (window={window})"
            )
        if max_consecutive < 1:
            raise ValueError(
                f"spike_skip_max_consecutive must be >= 1, got {max_consecutive}"
            )
        self.factor = float(factor)
        self.min_history = min_history
        self.max_consecutive = max_consecutive
        self.history: collections.deque[float] = collections.deque(maxlen=window)
        self.consecutive = 0
        self.skipped_total = 0
        self.consecutive_limit_hits = 0

    def observe(self, grad_norm: float) -> SpikeDecision:
        median = statistics.median(self.history) if self.history else None

        # A non-finite update would corrupt the weights and both moments, so it
        # is always skipped and never counts toward the consecutive limit.
        if not math.isfinite(grad_norm):
            self.skipped_total += 1
            return SpikeDecision(True, "non_finite", grad_norm, median)

        if len(self.history) < self.min_history or grad_norm <= self.factor * median:
            self.history.append(grad_norm)
            self.consecutive = 0
            return SpikeDecision(False, None, grad_norm, median)

        if self.consecutive >= self.max_consecutive:
            # Several outliers in a row look like a new gradient scale rather
            # than an isolated spike: accept it and let the median follow.
            self.history.append(grad_norm)
            self.consecutive = 0
            self.consecutive_limit_hits += 1
            return SpikeDecision(False, "consecutive_limit", grad_norm, median)

        # Skipped norms stay out of the history so a spike cannot raise the
        # baseline it is judged against.
        self.consecutive += 1
        self.skipped_total += 1
        return SpikeDecision(True, "spike", grad_norm, median)


def synchronized_grad_norm(grad_norm, device) -> float:
    """Return one gradient norm that every process agrees on.

    DDP gradients are identical after all-reduce, so each rank's norm should
    already match; taking the MAX makes that a guarantee rather than an
    assumption. If ranks ever disagreed on skipping, their replicas would
    silently diverge (DDP never re-syncs parameters). Non-finite values become
    +inf first, so NaN on any rank reaches every rank.
    """
    if hasattr(grad_norm, "full_tensor"):  # DTensor under FSDP2
        grad_norm = grad_norm.full_tensor()
    norm = torch.as_tensor(grad_norm, dtype=torch.float64, device=device)
    norm = norm.detach().reshape(1).clone()
    norm = torch.nan_to_num(norm, nan=math.inf, posinf=math.inf)
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        torch.distributed.all_reduce(norm, op=torch.distributed.ReduceOp.MAX)
    return norm.item()


def install_optimizer_step_guard(
    optimizer: torch.optim.Optimizer, should_skip: Callable[[], bool]
) -> None:
    """Make ``optimizer.step`` a no-op whenever ``should_skip()`` is true.

    A skipped call leaves the parameters, both moments, and the optimizer's own
    step counter untouched. Zeroing or dropping gradients would not: momentum
    still moves the parameters, and optimi's StableAdamW advances its step
    counter even when every gradient is None.

    The guard is a bound method because ``torch.optim.lr_scheduler`` wraps
    ``optimizer.step`` through ``step.__func__``; installing it before the
    scheduler is created keeps that wrapper (and its warnings) working.
    """
    if getattr(optimizer, "_slt_step_guard_installed", False):
        return
    original_step = optimizer.step

    def guarded_step(self, *args, **kwargs):
        if should_skip():
            return None
        return original_step(*args, **kwargs)

    if hasattr(original_step, "_wrapped_by_lr_sched"):
        # Installed after the scheduler: keep its "already patched" marker.
        guarded_step._wrapped_by_lr_sched = True
    optimizer.step = MethodType(guarded_step, optimizer)
    optimizer._slt_step_guard_installed = True
