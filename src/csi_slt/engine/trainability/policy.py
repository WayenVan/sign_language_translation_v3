"""Apply explicit SLT trainability plans to a constructed model."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from .plans import SltTrainabilityPlan, VisualBackboneTrainabilityPlan


def _set_module_trainable(module: nn.Module | None, trainable: bool) -> None:
    # Some SLT components are optional. A frozen or full plan remains portable
    # across models that do not construct that component.
    if module is not None:
        module.requires_grad_(trainable)


def _enable_lora(module: nn.Module, component_name: str) -> None:
    parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if "lora_" in name
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
        "mode='last_n_layers'"
    )


def _apply_visual_backbone_plan(
    module: nn.Module,
    plan: VisualBackboneTrainabilityPlan,
) -> None:
    if plan.mode == "frozen":
        return
    if plan.mode == "full":
        module.requires_grad_(True)
        return
    if plan.mode == "lora":
        _enable_lora(module, "visual")
        return

    visual_encoder = getattr(module, "visual_encoder", None)
    if not isinstance(visual_encoder, nn.Module):
        raise TypeError(
            "visual_backbone mode='last_n_layers' requires visual_encoder"
        )
    layers = _resolve_visual_layers(visual_encoder)
    if plan.n_layers is None or plan.n_layers > len(layers):
        raise ValueError(
            f"Requested the final {plan.n_layers} visual layers, but the "
            f"backbone exposes only {len(layers)}"
        )
    for layer in layers[-plan.n_layers :]:
        layer.requires_grad_(True)

    if plan.train_final_norm:
        for path in (
            "radio_model.norm",
            "vision_model.post_layernorm",
            "layernorm",
            "norm",
        ):
            norm = _resolve_module(visual_encoder, path)
            if norm is not None:
                norm.requires_grad_(True)

    if plan.train_auxiliary_modules:
        for name, parameter in module.named_parameters():
            if not name.startswith("visual_encoder."):
                parameter.requires_grad_(True)


def apply_trainability_plan(model: nn.Module, plan: SltTrainabilityPlan) -> int:
    """Freeze everything, then enable exactly the components selected by plan."""
    model.requires_grad_(False)

    llm = getattr(model, "llm", None)
    if not isinstance(llm, nn.Module):
        raise TypeError("SLT model must expose an llm module")
    if plan.llm.mode == "full":
        llm.requires_grad_(True)
    elif plan.llm.mode == "lora":
        _enable_lora(llm, "LLM")

    visual_backbone = getattr(model, "visual_backbone", None)
    if not isinstance(visual_backbone, nn.Module):
        raise TypeError("SLT model must expose a visual_backbone module")
    _apply_visual_backbone_plan(visual_backbone, plan.visual_backbone)

    for name, component_plan in (
        ("visual_adapter", plan.visual_adapter),
        ("visual_semantic_encoder", plan.visual_semantic_encoder),
        ("ctc_head", plan.ctc_head),
        ("visual_position_embedding", plan.visual_position_embedding),
    ):
        _set_module_trainable(
            getattr(model, name, None), component_plan.mode == "full"
        )

    for name in ("start_video_embds", "end_video_embeds"):
        parameter = getattr(model, name, None)
        if parameter is not None:
            parameter.requires_grad_(plan.visual_boundary_embeddings.mode == "full")

    visual_scale = getattr(model, "visual_scale", None)
    if visual_scale is not None:
        visual_scale.requires_grad_(plan.visual_scale.mode == "full")

    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_count == 0:
        raise ValueError("The trainability plan selected no trainable parameters")
    return trainable_count
