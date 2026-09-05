"""Validated, immutable descriptions of SLT component trainability.

Plans contain user intent only. They do not inspect models, change
``requires_grad``, or maintain module ``train``/``eval`` modes; those jobs
belong to trainability policies and model runtime state.

What differs per component is data, not code: which ``parameter_mode`` values
make sense, whether the component has a runtime mode at all, and which extra
options it accepts. ``_COMPONENT_RULES`` is that data, and one
``ComponentTrainability`` validates itself against its own row -- so adding a
component is a new entry here, not a new dataclass with its own hand-written
``__post_init__``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


# Keep in sync with `engine.optimization.OPTIMIZABLE_COMPONENTS`: both plans
# partition the model's parameters, and an optimizer override is validated
# against the trainability decision made for the same component name.
_COMPONENT_RULES = {
    "llm": {
        "parameter_modes": frozenset({"frozen", "full", "lora"}),
        "supports_runtime": True,
        "options": frozenset(),
    },
    "visual_backbone": {
        "parameter_modes": frozenset({"frozen", "full", "lora", "last_n_layers"}),
        "supports_runtime": True,
        "options": frozenset(
            {"n_layers", "train_final_norm", "train_auxiliary_modules"}
        ),
    },
    "visual_adapter": {
        "parameter_modes": frozenset({"frozen", "full"}),
        "supports_runtime": True,
        "options": frozenset(),
    },
    "ctc_head": {
        "parameter_modes": frozenset({"frozen", "full"}),
        "supports_runtime": False,
        "options": frozenset(),
    },
    "visual_position_embedding": {
        "parameter_modes": frozenset({"frozen", "full"}),
        "supports_runtime": False,
        "options": frozenset(),
    },
    "visual_boundary_embeddings": {
        "parameter_modes": frozenset({"frozen", "full"}),
        "supports_runtime": False,
        "options": frozenset(),
    },
    "visual_scale": {
        "parameter_modes": frozenset({"frozen", "full"}),
        "supports_runtime": False,
        "options": frozenset(),
    },
}


@dataclass(frozen=True)
class ComponentTrainability:
    """Normalized parameter and runtime intent for one component.

    ``runtime_mode`` is a permission, not a state: it says whether the
    component may enter training mode *when the model itself does*, and
    ``model.eval()`` always wins over it. Hence ``follow`` rather than
    ``train`` -- nothing here pins a module into training mode.

    * omitted (the default): derived from ``parameter_mode`` -- ``frozen``
      stays deterministic, anything trained follows the model.
    * ``follow``: follow the model's own ``train()``/``eval()`` state.
    * ``eval``: stay in eval even while the model trains. This is the one
      combination the derivation cannot express: parameters that receive
      gradients while the module's stochastic behavior stays off, such as a
      LoRA visual backbone whose base encoder must stay deterministic.
    """

    name: str
    parameter_mode: str
    runtime_mode: str | None
    options: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls, name: str, config: Mapping[str, Any]
    ) -> "ComponentTrainability":
        if name not in _COMPONENT_RULES:
            raise ValueError(f"Unknown trainability component: {name!r}")
        if not isinstance(config, Mapping):
            raise TypeError(f"engine.trainability.{name} must be a mapping")

        allowed_fields = {"parameter_mode", "runtime_mode", "options"}
        unknown_fields = set(config).difference(allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"engine.trainability.{name} contains unknown fields: "
                + ", ".join(sorted(unknown_fields))
            )

        parameter_mode = config.get("parameter_mode", "frozen")
        rule = _COMPONENT_RULES[name]
        if parameter_mode not in rule["parameter_modes"]:
            raise ValueError(
                f"Unsupported parameter_mode for {name}: {parameter_mode!r}; "
                f"expected one of {sorted(rule['parameter_modes'])}"
            )

        runtime_mode = config.get("runtime_mode")
        if rule["supports_runtime"]:
            if runtime_mode is not None and runtime_mode not in ("follow", "eval"):
                raise ValueError(
                    f"runtime_mode for {name} must be follow or eval; omit it to "
                    "derive the mode from parameter_mode"
                )
        elif runtime_mode is not None:
            raise ValueError(f"{name} does not support runtime_mode")

        options = config.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError(f"engine.trainability.{name}.options must be a mapping")
        unknown_options = set(options).difference(rule["options"])
        if unknown_options:
            raise ValueError(
                f"engine.trainability.{name}.options contains unsupported keys: "
                + ", ".join(sorted(unknown_options))
            )
        options = dict(options)
        cls._validate_options(name, parameter_mode, options)
        return cls(
            name=name,
            parameter_mode=parameter_mode,
            runtime_mode=runtime_mode,
            options=MappingProxyType(options),
        )

    @staticmethod
    def _validate_options(
        name: str, parameter_mode: str, options: dict[str, Any]
    ) -> None:
        if name != "visual_backbone":
            return
        n_layers = options.get("n_layers")
        if parameter_mode == "last_n_layers":
            if (
                isinstance(n_layers, bool)
                or not isinstance(n_layers, int)
                or n_layers <= 0
            ):
                raise ValueError(
                    "visual_backbone.options.n_layers must be a positive integer "
                    "when parameter_mode='last_n_layers'"
                )
        elif n_layers is not None:
            raise ValueError(
                "visual_backbone.options.n_layers is only valid with "
                "parameter_mode='last_n_layers'"
            )
        for key in ("train_final_norm", "train_auxiliary_modules"):
            if key in options and not isinstance(options[key], bool):
                raise TypeError(f"visual_backbone.options.{key} must be a boolean")

    @property
    def supports_runtime_mode(self) -> bool:
        """Whether this component has a runtime mode at all."""
        return bool(_COMPONENT_RULES[self.name]["supports_runtime"])

    @property
    def resolved_runtime_mode(self) -> str | None:
        """``follow`` or ``eval``; ``None`` for components without a runtime mode.

        A stated ``runtime_mode`` wins. Omitting it derives the mode from
        ``parameter_mode``, which is what every current run wants: a frozen
        component stays deterministic, a trained one follows the model. The
        derivation is not a value one can write, so there is no spelling of the
        default that could drift away from it.
        """
        if not self.supports_runtime_mode:
            return None
        if self.runtime_mode is not None:
            return self.runtime_mode
        return "eval" if self.parameter_mode == "frozen" else "follow"

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)


@dataclass(frozen=True)
class SltTrainabilityPlan(Mapping[str, ComponentTrainability]):
    """Complete validated trainability intent for every SLT component."""

    components: Mapping[str, ComponentTrainability]

    def __post_init__(self) -> None:
        missing = set(_COMPONENT_RULES).difference(self.components)
        unknown = set(self.components).difference(_COMPONENT_RULES)
        if missing:
            raise ValueError(
                "trainability plan is missing components: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                "trainability plan contains unknown components: "
                + ", ".join(sorted(unknown))
            )
        for name, component in self.components.items():
            if not isinstance(component, ComponentTrainability):
                raise TypeError(f"trainability component {name!r} has an invalid type")
            if component.name != name:
                raise ValueError(
                    f"trainability component key {name!r} does not match "
                    f"component name {component.name!r}"
                )
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SltTrainabilityPlan":
        if not isinstance(config, Mapping):
            raise TypeError("engine.trainability must be a mapping")
        missing = set(_COMPONENT_RULES).difference(config)
        unknown = set(config).difference(_COMPONENT_RULES)
        if missing:
            raise ValueError(
                "engine.trainability is missing components: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                "engine.trainability contains unknown components: "
                + ", ".join(sorted(unknown))
            )
        return cls(
            {
                name: ComponentTrainability.from_mapping(name, config[name])
                for name in _COMPONENT_RULES
            }
        )

    def __getitem__(self, name: str) -> ComponentTrainability:
        return self.components[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)
