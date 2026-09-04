"""Validated component overrides for optimizer hyperparameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
class ParameterGroupOptimization:
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
class ComponentOptimization(ParameterGroupOptimization):
    parameter_groups: Mapping[str, ParameterGroupOptimization] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "parameter_groups", MappingProxyType(dict(self.parameter_groups))
        )


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
            unknown_fields = set(values).difference(
                {"learning_rate", "weight_decay", "parameter_groups"}
            )
            if unknown_fields:
                raise ValueError(
                    f"engine.optimization.{name} contains unknown fields: "
                    + ", ".join(sorted(unknown_fields))
                )
            raw_groups = values.get("parameter_groups", {})
            if not isinstance(raw_groups, Mapping):
                raise TypeError(
                    f"engine.optimization.{name}.parameter_groups must be a mapping"
                )
            parameter_groups = {}
            for group_name, group_values in raw_groups.items():
                if not isinstance(group_name, str) or not group_name:
                    raise ValueError(
                        "optimization parameter group names must be non-empty strings"
                    )
                if not isinstance(group_values, Mapping):
                    raise TypeError(
                        f"engine.optimization.{name}.parameter_groups.{group_name} "
                        "must be a mapping"
                    )
                unknown_group_fields = set(group_values).difference(
                    {"learning_rate", "weight_decay"}
                )
                if unknown_group_fields:
                    raise ValueError(
                        f"engine.optimization.{name}.parameter_groups.{group_name} "
                        "contains unknown fields: "
                        + ", ".join(sorted(unknown_group_fields))
                    )
                parameter_groups[group_name] = ParameterGroupOptimization(
                    **dict(group_values)
                )
            components[name] = ComponentOptimization(
                learning_rate=values.get("learning_rate"),
                weight_decay=values.get("weight_decay"),
                parameter_groups=parameter_groups,
            )
        return cls(components)

    def get(self, name: str) -> ComponentOptimization:
        return self.components.get(name, ComponentOptimization())
