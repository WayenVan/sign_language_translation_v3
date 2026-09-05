"""Evaluate a scalar schedule at a point in training."""

from __future__ import annotations

import math
from typing import Optional

from .plans import ScalarAnnealSchedule


def value_at(
    schedule: ScalarAnnealSchedule, *, step: int, max_steps: Optional[int]
) -> float:
    """Interpolate ``schedule`` at ``step`` out of ``max_steps`` total steps.

    Falls back to ``schedule.start`` when the total step count is not known
    (for example a standalone ``trainer.evaluate()`` call that never ran
    ``train()``, where ``Trainer.state.max_steps`` stays 0) since there is no
    training progress to measure against.
    """
    if not max_steps or max_steps <= 0:
        return schedule.start
    anneal_steps = max(1.0, schedule.anneal_ratio * max_steps)
    progress = min(1.0, max(0.0, step / anneal_steps))
    if schedule.mode == "cosine":
        progress = (1 - math.cos(math.pi * progress)) / 2
    return schedule.start + (schedule.end - schedule.start) * progress
