"""Immutable descriptions of which SLT components should be trainable.

Plans contain user intent only. They do not inspect models, change
``requires_grad``, or maintain module ``train``/``eval`` modes; those jobs
belong to trainability policies and model runtime state.
"""

from dataclasses import dataclass, field
from typing import Literal


ComponentTrainabilityMode = Literal["frozen", "full"]
LlmTrainabilityMode = Literal["frozen", "full", "lora"]
VisualBackboneTrainabilityMode = Literal["frozen", "last_n_layers", "full"]


@dataclass(frozen=True)
class ComponentTrainabilityPlan:
    """Trainability intent for a regular model component."""

    mode: ComponentTrainabilityMode = "frozen"

    def __post_init__(self) -> None:
        if self.mode not in ("frozen", "full"):
            raise ValueError(f"Unsupported component trainability mode: {self.mode!r}")


@dataclass(frozen=True)
class LlmTrainabilityPlan:
    """Trainability intent for the language model."""

    mode: LlmTrainabilityMode = "frozen"

    def __post_init__(self) -> None:
        if self.mode not in ("frozen", "full", "lora"):
            raise ValueError(f"Unsupported LLM trainability mode: {self.mode!r}")


@dataclass(frozen=True)
class VisualBackboneTrainabilityPlan:
    """Trainability intent for a visual encoder and backbone-owned modules."""

    mode: VisualBackboneTrainabilityMode = "frozen"
    n_layers: int | None = None
    train_final_norm: bool = True
    train_auxiliary_modules: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("frozen", "last_n_layers", "full"):
            raise ValueError(
                f"Unsupported visual backbone trainability mode: {self.mode!r}"
            )

        if self.mode == "last_n_layers":
            if (
                isinstance(self.n_layers, bool)
                or not isinstance(self.n_layers, int)
                or self.n_layers <= 0
            ):
                raise ValueError(
                    "n_layers must be a positive integer when mode is "
                    "'last_n_layers'"
                )
        elif self.n_layers is not None:
            raise ValueError(
                "n_layers can only be set when mode is 'last_n_layers'"
            )

        if not isinstance(self.train_final_norm, bool):
            raise TypeError("train_final_norm must be a boolean")
        if not isinstance(self.train_auxiliary_modules, bool):
            raise TypeError("train_auxiliary_modules must be a boolean")


@dataclass(frozen=True)
class SltTrainabilityPlan:
    """Top-level trainability plan composed from SLT component plans."""

    llm: LlmTrainabilityPlan = field(default_factory=LlmTrainabilityPlan)
    visual_backbone: VisualBackboneTrainabilityPlan = field(
        default_factory=VisualBackboneTrainabilityPlan
    )
    visual_adapter: ComponentTrainabilityPlan = field(
        default_factory=ComponentTrainabilityPlan
    )
    visual_semantic_encoder: ComponentTrainabilityPlan = field(
        default_factory=ComponentTrainabilityPlan
    )
