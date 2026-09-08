"""Resolve which transformer layers a LoRA adapter should target.

A visual backbone's feature output can be taken from an intermediate block
(``output_layer``), so the "last N blocks" a visual LoRA should train are the N
blocks *ending at that block*, not the N final blocks of the module list --
blocks past ``output_layer`` never feed the loss and would collect no gradient.

The peft config files therefore stay pure ``LoraConfig`` and carry the layer
choice in a sibling ``*_lora_layers`` mapping:

* ``{"count": 4}``                           -- the last 4 blocks
* ``{"anchor": "output_layer", "count": 4}`` -- the 4 blocks ending at the
  backbone's ``output_layer``

This module turns that mapping into the concrete ``layers_to_transform`` list
peft expects, resolved against the live module. The result is written back onto
the ``LoraConfig`` before injection, so a checkpoint always serializes explicit
indices and the reload path never re-resolves anything.
"""

from __future__ import annotations

from collections.abc import Mapping

from peft import LoraConfig
from torch import nn

# Ordered attribute paths to the transformer block list of a visual encoder or
# a causal LM. First match wins; keep the most specific paths first.
_LAYER_LIST_PATHS = (
    "radio_model.blocks",
    "vision_model.encoder.layers",
    "model.layers",
    "encoder.layer",
    "encoder.layers",
    "blocks",
    "layers",
)

_ALLOWED_SPEC_KEYS = frozenset({"anchor", "count", "pattern"})
_ALLOWED_ANCHORS = ("last", "output_layer")


def _resolve_module(root: nn.Module, path: str) -> nn.Module | None:
    current: object = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, nn.Module) else None


def find_transformer_layers(module: nn.Module) -> nn.ModuleList | nn.Sequential:
    """Locate the ordered transformer block list inside ``module``."""
    for path in _LAYER_LIST_PATHS:
        layers = _resolve_module(module, path)
        if isinstance(layers, (nn.ModuleList, nn.Sequential)):
            return layers
    raise TypeError(
        f"Could not locate a transformer block list on {type(module).__name__}"
    )


def last_n_layer_indices(
    num_layers: int, count: int | None, *, end: int | None = None
) -> list[int] | None:
    """The ``count`` block indices ending at ``end`` (default: the last block).

    ``count is None`` returns ``None`` -- peft reads that as "every layer".
    ``end`` is a resolved, non-negative index: a visual backbone whose feature
    output is an intermediate block passes that block's index here so the span
    never extends past it.
    """
    if count is None:
        return None
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            f"lora layer count must be a positive integer, got {count!r}"
        )
    if num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    if end is None:
        end = num_layers - 1
    if not 0 <= end < num_layers:
        raise ValueError(
            f"lora layer anchor index {end} is out of range for a backbone with "
            f"{num_layers} blocks"
        )
    start = end - count + 1
    if start < 0:
        raise ValueError(
            f"requested the {count} blocks ending at index {end}, but only "
            f"{end + 1} block(s) precede and include it"
        )
    return list(range(start, end + 1))


def normalize_layer_spec(spec: Mapping[str, object] | None) -> dict | None:
    """Validate a ``*_lora_layers`` mapping; return a plain dict or ``None``.

    Keys: ``count`` (required, positive int), ``anchor`` (``last`` default, or
    ``output_layer``), ``pattern`` (optional -- the ``layers_pattern`` peft
    needs to read ``layers_to_transform`` against non-``layers`` module names,
    such as a timm ViT's ``blocks``). ``pattern`` lives here rather than in the
    LoRA config because peft rejects ``layers_pattern`` without an accompanying
    ``layers_to_transform``, which only exists once the span is resolved.
    """
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise TypeError("lora layers spec must be a mapping")
    unknown = set(spec).difference(_ALLOWED_SPEC_KEYS)
    if unknown:
        raise ValueError(
            "lora layers spec contains unknown keys: " + ", ".join(sorted(unknown))
        )
    if "count" not in spec:
        raise ValueError("lora layers spec must set 'count'")
    anchor = spec.get("anchor", "last")
    if anchor not in _ALLOWED_ANCHORS:
        raise ValueError(
            "lora layers spec anchor must be one of "
            f"{list(_ALLOWED_ANCHORS)}, got {anchor!r}"
        )
    pattern = spec.get("pattern")
    if pattern is not None and not isinstance(pattern, str):
        raise TypeError("lora layers spec 'pattern' must be a string")
    return {"anchor": anchor, "count": spec["count"], "pattern": pattern}


def apply_layer_spec(
    peft_config: LoraConfig,
    layers_to_transform: list[int] | None,
    normalized_spec: Mapping[str, object],
) -> None:
    """Write a resolved span onto ``peft_config`` in place.

    ``layers_to_transform`` and ``layers_pattern`` are set together -- peft
    validates them as a pair at injection.
    """
    if layers_to_transform is None:
        return
    peft_config.layers_to_transform = layers_to_transform
    if normalized_spec.get("pattern") is not None:
        peft_config.layers_pattern = normalized_spec["pattern"]


def resolve_layers_to_transform(
    module: nn.Module,
    spec: Mapping[str, object] | None,
    *,
    end_index: int | None = None,
) -> list[int] | None:
    """Concrete ``layers_to_transform`` for ``spec`` against a live ``module``.

    ``end_index`` overrides the span's last block and is required when the
    normalized spec anchors on ``output_layer``; callers that own an
    ``output_layer`` (the visual backbones) resolve it to a non-negative index
    first.
    """
    normalized = normalize_layer_spec(spec)
    if normalized is None:
        return None
    if normalized["anchor"] == "output_layer" and end_index is None:
        raise ValueError(
            "lora layers spec anchors on 'output_layer', but no end index was "
            "supplied by the caller"
        )
    layers = find_transformer_layers(module)
    resolved_end = end_index if normalized["anchor"] == "output_layer" else None
    return last_n_layer_indices(
        len(layers), normalized["count"], end=resolved_end
    )
