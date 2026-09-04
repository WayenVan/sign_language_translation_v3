"""Resolve optimizer fallbacks and group trainable parameters by component."""

from __future__ import annotations

from collections.abc import Collection

from torch import nn

from .plans import OPTIMIZABLE_COMPONENTS, OptimizationPlan


def _component_parameter_ids(model: nn.Module) -> dict[str, set[int]]:
    def module_ids(name: str) -> set[int]:
        module = getattr(model, name, None)
        return (
            {id(parameter) for parameter in module.parameters()}
            if isinstance(module, nn.Module)
            else set()
        )

    boundary_ids = {
        id(parameter)
        for name in ("start_video_embds", "end_video_embeds")
        if isinstance((parameter := getattr(model, name, None)), nn.Parameter)
    }
    return {
        "llm": module_ids("llm"),
        "visual_backbone": module_ids("visual_backbone"),
        "visual_adapter": module_ids("visual_adapter"),
        "ctc_head": module_ids("ctc_head"),
        "ctc_codebook": module_ids("ctc_codebook"),
        "visual_position_embedding": module_ids("visual_position_embedding"),
        "visual_boundary_embeddings": boundary_ids,
    }


def build_optimizer_parameter_groups(
    *,
    model: nn.Module,
    ownership_model: nn.Module,
    plan: OptimizationPlan,
    default_learning_rate: float,
    default_weight_decay: float,
    decay_parameter_names: Collection[str],
) -> list[dict]:
    """Build resolved decay/no-decay groups for every trainable component."""
    component_ids = _component_parameter_ids(ownership_model)
    trainable_components = {
        component
        for component, parameter_ids in component_ids.items()
        if any(
            parameter.requires_grad
            for parameter in ownership_model.parameters()
            if id(parameter) in parameter_ids
        )
    }
    configured_but_frozen = set(plan.components).difference(trainable_components)
    if configured_but_frozen:
        raise ValueError(
            "engine.optimization overrides frozen or absent components: "
            + ", ".join(sorted(configured_but_frozen))
        )

    parameter_component = {
        parameter_id: component
        for component, parameter_ids in component_ids.items()
        for parameter_id in parameter_ids
    }
    decay_parameter_names = set(decay_parameter_names)
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = parameter_component.get(id(parameter))
        if component is None:
            raise ValueError(f"Trainable parameter {name!r} has no optimizer component")
        grouped.setdefault((component, name in decay_parameter_names), []).append(
            parameter
        )

    optimizer_groups = []
    for (component, use_decay), parameters in grouped.items():
        override = plan.get(component)
        learning_rate = (
            default_learning_rate
            if override.learning_rate is None
            else override.learning_rate
        )
        weight_decay = (
            default_weight_decay
            if override.weight_decay is None
            else override.weight_decay
        )
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay if use_decay else 0.0,
                "slt_component": component,
            }
        )
    if not optimizer_groups:
        raise ValueError("Cannot create an optimizer without trainable parameters")
    if set(parameter_component.values()).difference(OPTIMIZABLE_COMPONENTS):
        raise AssertionError("optimizer ownership contains an unknown component")
    return optimizer_groups
