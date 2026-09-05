"""Validated, immutable description of a scalar training-time schedule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

_SCHEDULE_MODES = frozenset({"linear", "cosine"})
_SCHEDULE_FIELDS = frozenset({"start", "end", "anneal_ratio", "mode"})


@dataclass(frozen=True)
class ScalarAnnealSchedule:
    """A ``start`` -> ``end`` interpolation over a leading fraction of training.

    Deliberately generic: it knows nothing about what it feeds (a codebook
    temperature today, potentially a loss weight later). ``policy.value_at``
    turns one of these plus a training step into a number; the caller decides
    what to do with it.
    """

    start: float
    end: float
    anneal_ratio: float = 1.0
    mode: Literal["linear", "cosine"] = "linear"

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            object.__setattr__(self, name, float(value))

        if isinstance(self.anneal_ratio, bool) or not isinstance(
            self.anneal_ratio, (int, float)
        ):
            raise TypeError("anneal_ratio must be a real number")
        if not 0.0 < self.anneal_ratio <= 1.0:
            raise ValueError("anneal_ratio must be in (0, 1]")
        object.__setattr__(self, "anneal_ratio", float(self.anneal_ratio))

        if self.mode not in _SCHEDULE_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_SCHEDULE_MODES)}, got {self.mode!r}"
            )

    @classmethod
    def from_mapping(
        cls, config: Optional[Mapping[str, Any]]
    ) -> Optional["ScalarAnnealSchedule"]:
        """Build a schedule from a config mapping, or ``None`` when absent.

        An empty/``None`` mapping means "no schedule": the caller keeps
        whatever static default it already had. This mirrors how
        ``SltTrainer`` treats an absent schedule -- unlike
        ``OptimizationPlan``, where "no overrides" is itself a valid plan,
        here "disabled" and "configured" are genuinely different states.
        """
        if not config:
            return None
        if not isinstance(config, Mapping):
            raise TypeError("schedule config must be a mapping")
        unknown = set(config).difference(_SCHEDULE_FIELDS)
        if unknown:
            raise ValueError(
                "schedule config contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        missing = {"start", "end"}.difference(config)
        if missing:
            raise ValueError(
                "schedule config is missing required fields: "
                + ", ".join(sorted(missing))
            )
        return cls(
            start=config["start"],
            end=config["end"],
            anneal_ratio=config.get("anneal_ratio", 1.0),
            mode=config.get("mode", "linear"),
        )
