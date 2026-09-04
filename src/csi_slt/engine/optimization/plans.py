"""Validated component overrides for optimizer hyperparameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


OPTIMIZABLE_COMPONENTS = frozenset(
    {
        "llm",
        "visual_backbone",
        "visual_adapter",
        "ctc_head",
        "ctc_codebook",
        "visual_position_embedding",
        "visual_boundary_embeddings",
    }
)


@dataclass(frozen=True)
class ComponentOptimization:
    learning_rate: float | None = None
    weight_decay: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"optimization {name} must be a real number or None")
            if value < 0:
                raise ValueError(f"optimization {name} must be non-negative")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class OptimizationPlan:
    """Sparse overrides; absent values inherit from TrainingArguments."""

    components: Mapping[str, ComponentOptimization]

    def __post_init__(self) -> None:
        unknown = set(self.components).difference(OPTIMIZABLE_COMPONENTS)
        if unknown:
            raise ValueError(
                "engine.optimization contains unknown components: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "OptimizationPlan":
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError("engine.optimization must be a mapping")
        unknown = set(config).difference(OPTIMIZABLE_COMPONENTS)
        if unknown:
            raise ValueError(
                "engine.optimization contains unknown components: "
                + ", ".join(sorted(unknown))
            )
        components = {}
        for name, values in config.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"engine.optimization.{name} must be a mapping")
            unknown_fields = set(values).difference({"learning_rate", "weight_decay"})
            if unknown_fields:
                raise ValueError(
                    f"engine.optimization.{name} contains unknown fields: "
                    + ", ".join(sorted(unknown_fields))
                )
            components[name] = ComponentOptimization(**dict(values))
        return cls(components)

    def get(self, name: str) -> ComponentOptimization:
        return self.components.get(name, ComponentOptimization())
