"""Resolve optimizer fallbacks and group trainable parameters by component."""

from __future__ import annotations

from collections.abc import Collection, Mapping

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

    def parameter_ids(*names: str) -> set[int]:
        """Own bare ``nn.Parameter`` attributes, which have no ``parameters()``."""
        return {
            id(parameter)
            for name in names
            if isinstance((parameter := getattr(model, name, None)), nn.Parameter)
        }

    return {
        "llm": module_ids("llm"),
        "visual_backbone": module_ids("visual_backbone"),
        "visual_adapter": module_ids("visual_adapter"),
        "ctc_head": module_ids("ctc_head"),
        "visual_position_embedding": module_ids("visual_position_embedding"),
        "visual_boundary_embeddings": parameter_ids(
            "start_video_embds", "end_video_embeds"
        ),
        "visual_scale": parameter_ids("visual_scale"),
    }


def _semantic_parameter_groups(
    *, component: str, module: nn.Module | None, component_ids: set[int]
) -> dict[str, tuple[nn.Parameter, ...]]:
    registration = getattr(module, "optimization_parameter_groups", None)
    if registration is None:
        return {}
    groups = registration()
    if not isinstance(groups, Mapping):
        raise TypeError(
            f"{component}.optimization_parameter_groups() must return a mapping"
        )

    normalized = {}
    seen: dict[int, str] = {}
    for group_name, parameters in groups.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"{component} registered an invalid parameter group name")
        try:
            parameters = tuple(parameters)
        except TypeError as error:
            raise TypeError(
                f"{component} parameter group {group_name!r} must be iterable"
            ) from error
        for parameter in parameters:
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(
                    f"{component} parameter group {group_name!r} contains a non-Parameter"
                )
            parameter_id = id(parameter)
            if parameter_id not in component_ids:
                raise ValueError(
                    f"{component} parameter group {group_name!r} contains a parameter "
                    "outside the component"
                )
            if parameter_id in seen:
                raise ValueError(
                    f"{component} parameter groups {seen[parameter_id]!r} and "
                    f"{group_name!r} overlap"
                )
            seen[parameter_id] = group_name
        normalized[group_name] = parameters
    return normalized


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

    parameter_semantic_group: dict[int, str] = {}
    for component, override in plan.components.items():
        if not override.parameter_groups:
            continue
        module = getattr(ownership_model, component, None)
        registered = _semantic_parameter_groups(
            component=component,
            module=module if isinstance(module, nn.Module) else None,
            component_ids=component_ids[component],
        )
        unknown_groups = set(override.parameter_groups).difference(registered)
        if unknown_groups:
            raise ValueError(
                f"engine.optimization.{component}.parameter_groups contains "
                "unregistered groups: " + ", ".join(sorted(unknown_groups))
            )
        for group_name in override.parameter_groups:
            parameters = registered[group_name]
            if not parameters:
                raise ValueError(
                    f"Configured parameter group {component}.{group_name} is empty"
                )
            trainable = [
                parameter for parameter in parameters if parameter.requires_grad
            ]
            if not trainable:
                raise ValueError(
                    f"Configured parameter group {component}.{group_name} is frozen"
                )
            parameter_semantic_group.update(
                {id(parameter): group_name for parameter in trainable}
            )

    parameter_component = {
        parameter_id: component
        for component, parameter_ids in component_ids.items()
        for parameter_id in parameter_ids
    }
    decay_parameter_names = set(decay_parameter_names)
    grouped: dict[tuple[str, str | None, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = parameter_component.get(id(parameter))
        if component is None:
            raise ValueError(f"Trainable parameter {name!r} has no optimizer component")
        semantic_group = parameter_semantic_group.get(id(parameter))
        grouped.setdefault(
            (component, semantic_group, name in decay_parameter_names), []
        ).append(parameter)

    optimizer_groups = []
    for (component, semantic_group, use_decay), parameters in grouped.items():
        override = plan.get(component)
        group_override = (
            override.parameter_groups[semantic_group]
            if semantic_group is not None
            else None
        )
        learning_rate = (
            group_override.learning_rate
            if group_override is not None and group_override.learning_rate is not None
            else (
                default_learning_rate
                if override.learning_rate is None
                else override.learning_rate
            )
        )
        weight_decay = (
            group_override.weight_decay
            if group_override is not None and group_override.weight_decay is not None
            else (
                default_weight_decay
                if override.weight_decay is None
                else override.weight_decay
            )
        )
        optimizer_group = {
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay if use_decay else 0.0,
            "slt_component": component,
        }
        if semantic_group is not None:
            optimizer_group["slt_parameter_group"] = semantic_group
        optimizer_groups.append(optimizer_group)
    if not optimizer_groups:
        raise ValueError("Cannot create an optimizer without trainable parameters")
    if set(parameter_component.values()).difference(OPTIMIZABLE_COMPONENTS):
        raise AssertionError("optimizer ownership contains an unknown component")
    return optimizer_groups
