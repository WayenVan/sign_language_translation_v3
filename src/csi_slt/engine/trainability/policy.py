"""Apply explicit SLT trainability plans to a constructed model."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from torch import nn

from .plans import ComponentTrainability, SltTrainabilityPlan

logger = logging.getLogger(__name__)


def _refreeze_always_frozen(model: nn.Module) -> list[str]:
    """Re-freeze modules that no plan is allowed to train, and name them.

    ``requires_grad_`` recurses, so a plan that trains a component also trains
    anything nested inside it. Some nested modules are fitted constants rather
    than parameters -- the hand-patch scorer's coefficients, for one, which were
    fitted offline against a specific backbone and whose gradients would quietly
    invalidate that provenance. They opt out by carrying ``always_frozen``.

    The marker is expected to be an *instance* attribute set by whoever owns the
    module, so ownership decides, and turning the option off in a config is
    still enough to make such a module trainable.
    """
    frozen = []
    for name, module in model.named_modules():
        if getattr(module, "always_frozen", False):
            module.requires_grad_(False)
            frozen.append(name or "<root>")
    return frozen


def _set_module_trainable(module: nn.Module | None, trainable: bool) -> None:
    # Some SLT components are optional. A frozen or full plan remains portable
    # across models that do not construct that component.
    if module is not None:
        module.requires_grad_(trainable)


def _enable_lora(module: nn.Module, component_name: str) -> None:
    parameters = [
        parameter for name, parameter in module.named_parameters() if "lora_" in name
    ]
    if not parameters:
        raise ValueError(
            f"{component_name} LoRA training was requested, but the model "
            "contains no matching LoRA parameters"
        )
    for parameter in parameters:
        parameter.requires_grad_(True)


def _resolve_module(root: nn.Module, path: str) -> nn.Module | None:
    current: object = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, nn.Module) else None


def _resolve_visual_layers(visual_encoder: nn.Module) -> Sequence[nn.Module]:
    for path in (
        "radio_model.blocks",
        "vision_model.encoder.layers",
        "encoder.layer",
        "encoder.layers",
        "blocks",
        "layers",
    ):
        layers = _resolve_module(visual_encoder, path)
        if isinstance(layers, (nn.ModuleList, nn.Sequential)):
            return layers
    raise TypeError(
        "Could not locate transformer layers for visual_backbone "
        "parameter_mode='last_n_layers'"
    )


def _apply_visual_backbone_plan(
    module: nn.Module,
    plan: ComponentTrainability,
) -> None:
    if plan.parameter_mode == "frozen":
        return
    if plan.parameter_mode == "full":
        module.requires_grad_(True)
        return
    if plan.parameter_mode == "lora":
        _enable_lora(module, "visual")
        return

    visual_encoder = getattr(module, "visual_encoder", None)
    if not isinstance(visual_encoder, nn.Module):
        raise TypeError(
            "visual_backbone parameter_mode='last_n_layers' requires visual_encoder"
        )
    layers = _resolve_visual_layers(visual_encoder)
    n_layers = plan.option("n_layers")
    if n_layers is None or n_layers > len(layers):
        raise ValueError(
            f"Requested the final {n_layers} visual layers, but the "
            f"backbone exposes only {len(layers)}"
        )
    for layer in layers[-n_layers:]:
        layer.requires_grad_(True)

    if plan.option("train_final_norm", True):
        for path in (
            "radio_model.norm",
            "vision_model.post_layernorm",
            "layernorm",
            "norm",
        ):
            norm = _resolve_module(visual_encoder, path)
            if norm is not None:
                norm.requires_grad_(True)

    if plan.option("train_auxiliary_modules", True):
        for name, parameter in module.named_parameters():
            if not name.startswith("visual_encoder."):
                parameter.requires_grad_(True)


def _apply_runtime_mode(
    owner: nn.Module,
    setter_name: str,
    plan: ComponentTrainability,
) -> None:
    """Hand a component's resolved runtime mode to whoever implements it.

    The hook is looked up by name because the plan layer never inspects models:
    ``_COMPONENT_RULES`` says a component *has* a runtime mode, and the module
    either implements the setter or does not. C-RADIO is currently the only
    visual backbone that does; the LLM and the visual adapter are driven by
    ``SltModel``'s own setters.

    When the hook is missing the two cases are not the same. A mode that was
    *derived* from ``parameter_mode`` is skipped: backbones that never opted in
    keep whatever train/eval semantics they already had. A mode the config
    *states* is an error -- the only reason to write one is to diverge from the
    derivation, so ignoring it would leave the run doing the opposite of what
    the file says, with nothing in the log to show for it.
    """
    setter = getattr(owner, setter_name, None)
    if setter is not None:
        setter(plan.resolved_runtime_mode)
        return
    if plan.runtime_mode is not None:
        raise ValueError(
            f"engine.trainability.{plan.name}.runtime_mode="
            f"{plan.runtime_mode!r} cannot be applied: "
            f"{type(owner).__name__} does not implement {setter_name}(). Omit "
            "it to accept the mode derived from parameter_mode."
        )


def apply_trainability_plan(model: nn.Module, plan: SltTrainabilityPlan) -> int:
    """Freeze everything, then enable exactly the components selected by plan."""
    model.requires_grad_(False)

    llm = getattr(model, "llm", None)
    if not isinstance(llm, nn.Module):
        raise TypeError("SLT model must expose an llm module")
    llm_plan = plan["llm"]
    if llm_plan.parameter_mode == "full":
        llm.requires_grad_(True)
    elif llm_plan.parameter_mode == "lora":
        _enable_lora(llm, "LLM")
    _apply_runtime_mode(model, "set_llm_runtime_mode", llm_plan)

    visual_backbone = getattr(model, "visual_backbone", None)
    if not isinstance(visual_backbone, nn.Module):
        raise TypeError("SLT model must expose a visual_backbone module")
    visual_backbone_plan = plan["visual_backbone"]
    _apply_visual_backbone_plan(visual_backbone, visual_backbone_plan)
    _apply_runtime_mode(visual_backbone, "set_runtime_mode", visual_backbone_plan)

    visual_adapter = getattr(model, "visual_adapter", None)
    visual_adapter_plan = plan["visual_adapter"]
    _set_module_trainable(
        visual_adapter, visual_adapter_plan.parameter_mode == "full"
    )
    _apply_runtime_mode(model, "set_visual_adapter_runtime_mode", visual_adapter_plan)

    for name, component_plan in (
        ("ctc_head", plan["ctc_head"]),
        ("ctc_codebook", plan["ctc_codebook"]),
        ("visual_position_embedding", plan["visual_position_embedding"]),
    ):
        _set_module_trainable(
            getattr(model, name, None), component_plan.parameter_mode == "full"
        )

    for name in ("start_video_embds", "end_video_embeds"):
        parameter = getattr(model, name, None)
        if parameter is not None:
            parameter.requires_grad_(
                plan["visual_boundary_embeddings"].parameter_mode == "full"
            )

    # After every plan decision, and before counting: a plan may have unfrozen a
    # component that holds one of these.
    always_frozen = _refreeze_always_frozen(model)
    if always_frozen:
        logger.info(
            "Kept %d module(s) frozen regardless of the plan: %s",
            len(always_frozen),
            ", ".join(always_frozen),
        )

    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_count == 0:
        raise ValueError("The trainability plan selected no trainable parameters")
    return trainable_count
