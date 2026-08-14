"""Schedulers for training objectives that are independent of the optimizer LR."""

from __future__ import annotations


class DSIDScheduler:
    """Schedule the coefficient of Direction-aware SID over optimizer steps.

    The schedule has three phases: linear warm-up, a constant plateau, and a
    linear decay. ``step`` is a zero-based optimizer-update index, so gradient
    accumulation micro-batches share the same coefficient.
    """

    def __init__(
        self,
        *,
        max_weight: float,
        total_steps: int,
        warmup_ratio: float = 0.1,
        decay_ratio: float = 0.3,
    ) -> None:
        if max_weight < 0.0:
            raise ValueError("max_weight must be non-negative")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if not 0.0 <= decay_ratio <= 1.0:
            raise ValueError("decay_ratio must be in [0, 1]")
        if warmup_ratio + decay_ratio > 1.0:
            raise ValueError("warmup_ratio and decay_ratio must sum to at most 1")

        self.max_weight = float(max_weight)
        self.total_steps = int(total_steps)
        self.warmup_ratio = float(warmup_ratio)
        self.decay_ratio = float(decay_ratio)
        self.current_step = 0
        self.current_weight = self.value_at(0)

    def value_at(self, step: int) -> float:
        """Return the coefficient for a zero-based optimizer-update index."""
        if step < 0:
            raise ValueError("step must be non-negative")

        # Using total_steps - 1 maps the first and final update exactly to
        # progress 0 and 1 respectively.
        denominator = max(self.total_steps - 1, 1)
        progress = min(step / denominator, 1.0)
        scale = 1.0
        if self.warmup_ratio > 0.0 and progress < self.warmup_ratio:
            scale = progress / self.warmup_ratio
        if self.decay_ratio > 0.0 and progress > 1.0 - self.decay_ratio:
            scale = min(scale, (1.0 - progress) / self.decay_ratio)
        return self.max_weight * max(scale, 0.0)

    def step(self, step: int | None = None) -> float:
        """Advance or synchronize the scheduler and return its coefficient."""
        next_step = self.current_step + 1 if step is None else int(step)
        self.current_step = next_step
        self.current_weight = self.value_at(next_step)
        return self.current_weight

    def state_dict(self) -> dict[str, int]:
        """Return the minimal state needed outside Trainer-managed runs."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore a scheduler created with the same immutable arguments."""
        if "current_step" not in state_dict:
            raise KeyError("scheduler state_dict is missing 'current_step'")
        self.step(int(state_dict["current_step"]))
