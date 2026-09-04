"""Immutable descriptions of which SLT components should be trainable.

Plans contain user intent only. They do not inspect models, change
``requires_grad``, or maintain module ``train``/``eval`` modes; those jobs
belong to trainability policies and model runtime state.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Literal


ComponentTrainabilityMode = Literal["frozen", "full"]
LlmTrainabilityMode = Literal["frozen", "full", "lora"]
LlmRuntimeMode = Literal["eval", "train"]
VisualBackboneTrainabilityMode = Literal["frozen", "last_n_layers", "full", "lora"]
VisualBackboneRuntimeMode = Literal["eval", "train"]
VisualAdapterRuntimeMode = Literal["eval", "train"]


@dataclass(frozen=True)
class ComponentTrainabilityPlan:
    """Trainability intent for a regular model component."""

    mode: ComponentTrainabilityMode = "frozen"

    def __post_init__(self) -> None:
        if self.mode not in ("frozen", "full"):
            raise ValueError(f"Unsupported component trainability mode: {self.mode!r}")


@dataclass(frozen=True)
class LlmTrainabilityPlan:
    """Gradient and runtime-mode intent for the language model."""

    mode: LlmTrainabilityMode = "frozen"
    runtime_mode: LlmRuntimeMode = "eval"

    def __post_init__(self) -> None:
        if self.mode not in ("frozen", "full", "lora"):
            raise ValueError(f"Unsupported LLM trainability mode: {self.mode!r}")
        if self.runtime_mode not in ("eval", "train"):
            raise ValueError(f"Unsupported LLM runtime mode: {self.runtime_mode!r}")


@dataclass(frozen=True)
class VisualBackboneTrainabilityPlan:
    """Gradient and runtime-mode intent for a visual backbone.

    ``runtime_mode`` currently has an explicit implementation in C-RADIO. It
    is deliberately independent of ``mode``: LoRA parameters can receive
    gradients while the frozen base encoder remains in deterministic eval
    mode.
    """

    mode: VisualBackboneTrainabilityMode = "frozen"
    runtime_mode: VisualBackboneRuntimeMode = "eval"
    n_layers: int | None = None
    train_final_norm: bool = True
    train_auxiliary_modules: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("frozen", "last_n_layers", "full", "lora"):
            raise ValueError(
                f"Unsupported visual backbone trainability mode: {self.mode!r}"
            )
        if self.runtime_mode not in ("eval", "train"):
            raise ValueError(
                f"Unsupported visual backbone runtime mode: {self.runtime_mode!r}"
            )

        if self.mode == "last_n_layers":
            if (
                isinstance(self.n_layers, bool)
                or not isinstance(self.n_layers, int)
                or self.n_layers <= 0
            ):
                raise ValueError(
                    "n_layers must be a positive integer when mode is 'last_n_layers'"
                )
        elif self.n_layers is not None:
            raise ValueError("n_layers can only be set when mode is 'last_n_layers'")

        if not isinstance(self.train_final_norm, bool):
            raise TypeError("train_final_norm must be a boolean")
        if not isinstance(self.train_auxiliary_modules, bool):
            raise TypeError("train_auxiliary_modules must be a boolean")


@dataclass(frozen=True)
class VisualAdapterTrainabilityPlan:
    """Gradient and runtime-mode intent for the visual adapter."""

    mode: ComponentTrainabilityMode = "frozen"
    runtime_mode: VisualAdapterRuntimeMode = "eval"

    def __post_init__(self) -> None:
        if self.mode not in ("frozen", "full"):
            raise ValueError(
                f"Unsupported visual adapter trainability mode: {self.mode!r}"
            )
        if self.runtime_mode not in ("eval", "train"):
            raise ValueError(
                f"Unsupported visual adapter runtime mode: {self.runtime_mode!r}"
            )


@dataclass(frozen=True)
class SltTrainabilityPlan:
    """Top-level trainability plan composed from SLT component plans."""

    llm: LlmTrainabilityPlan = field(default_factory=LlmTrainabilityPlan)
    visual_backbone: VisualBackboneTrainabilityPlan = field(
        default_factory=VisualBackboneTrainabilityPlan
    )
    visual_adapter: VisualAdapterTrainabilityPlan = field(
        default_factory=VisualAdapterTrainabilityPlan
    )
    ctc_head: ComponentTrainabilityPlan = field(
        default_factory=ComponentTrainabilityPlan
    )
    ctc_codebook: ComponentTrainabilityPlan = field(
        default_factory=ComponentTrainabilityPlan
    )
    visual_position_embedding: ComponentTrainabilityPlan = field(
        default_factory=ComponentTrainabilityPlan
    )
    visual_boundary_embeddings: ComponentTrainabilityPlan = field(
        default_factory=ComponentTrainabilityPlan
    )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SltTrainabilityPlan":
        """Build a complete plan from the colocated Hydra config."""
        if not isinstance(config, Mapping):
            raise TypeError("engine.trainability must be a mapping")

        expected = {
            "llm",
            "visual_backbone",
            "visual_adapter",
            "ctc_head",
            "ctc_codebook",
            "visual_position_embedding",
            "visual_boundary_embeddings",
        }
        missing = expected.difference(config)
        unknown = set(config).difference(expected)
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

        def values(name: str) -> dict[str, Any]:
            value = config[name]
            if not isinstance(value, Mapping):
                raise TypeError(f"engine.trainability.{name} must be a mapping")
            return dict(value)

        return cls(
            llm=LlmTrainabilityPlan(**values("llm")),
            visual_backbone=VisualBackboneTrainabilityPlan(**values("visual_backbone")),
            visual_adapter=VisualAdapterTrainabilityPlan(**values("visual_adapter")),
            ctc_head=ComponentTrainabilityPlan(**values("ctc_head")),
            ctc_codebook=ComponentTrainabilityPlan(**values("ctc_codebook")),
            visual_position_embedding=ComponentTrainabilityPlan(
                **values("visual_position_embedding")
            ),
            visual_boundary_embeddings=ComponentTrainabilityPlan(
                **values("visual_boundary_embeddings")
            ),
        )
