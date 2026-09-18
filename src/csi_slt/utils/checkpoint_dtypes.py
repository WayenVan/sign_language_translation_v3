"""Put a loaded model's tensors back to the dtypes its checkpoint stored.

``from_pretrained(..., dtype="auto")`` follows ``config.json``'s ``dtype``
field, but that field records the config object at save time, not the weights:
``csi_slt.commands.train`` casts the LLM with ``cast_module_dtype`` *after*
loading, which rewrites the module and leaves the config still saying
``float32``. A 14B stage-2 checkpoint consequently stores a bf16 LLM (29.4 GiB)
beside an fp32 visual tower (1.8 GiB) while announcing ``float32``, and
``"auto"`` upcasts all of it to ~63 GiB per rank -- twice what training used,
for no added precision.

This module reads each tensor's dtype out of the safetensors headers -- the
file's own record, without reading tensor data -- and casts the in-memory model
back to it. Loading at fp32 and narrowing afterwards is exact: every bf16 value
is representable in fp32, so the round trip returns the stored bits, and
tensors saved as fp32 are never touched. The result is the checkpoint's own
dtype layout, which is also the layout training ran with.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import torch
from transformers import logging


logger = logging.get_logger(__name__)


_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_FILE = "model.safetensors"

# Only floating-point storage types matter here; bool and integer buffers are
# left alone, exactly as ``Module.to(dtype=...)`` would.
_FLOAT_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def read_stored_dtypes(checkpoint_dir: str | Path) -> dict[str, torch.dtype]:
    """Map state-dict key to the floating-point dtype the checkpoint stores.

    Sharded and single-file safetensors checkpoints are both supported. Keys
    whose storage type is not floating point are omitted.
    """
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_dir)
    index_path = checkpoint_path / _SAFETENSORS_INDEX
    single_path = checkpoint_path / _SAFETENSORS_FILE

    shard_keys: dict[str, list[str]] = {}
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        for key, shard in weight_map.items():
            shard_keys.setdefault(shard, []).append(key)
    elif single_path.is_file():
        shard_keys[_SAFETENSORS_FILE] = []
    else:
        raise FileNotFoundError(
            f"no safetensors weights under {checkpoint_path}; pass an explicit "
            "engine.model_dtype (auto, float32, bfloat16, ...) instead of "
            "'checkpoint'"
        )

    stored: dict[str, torch.dtype] = {}
    for shard, keys in shard_keys.items():
        with safe_open(str(checkpoint_path / shard), framework="pt") as handle:
            # The single-file case has no index to enumerate keys from.
            for key in keys or handle.keys():
                dtype = _FLOAT_DTYPES.get(handle.get_slice(key).get_dtype())
                if dtype is not None:
                    stored[key] = dtype
    return stored


def _named_tensors(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Module, str, torch.Tensor, bool]]:
    """Yield ``(key, owner, attribute, tensor, is_parameter)`` for every tensor.

    Walking ``_parameters``/``_buffers`` rather than ``state_dict()`` keeps the
    owning module and attribute name, which is what an in-place cast needs.
    """
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for attribute, tensor in module._parameters.items():
            if tensor is not None:
                yield f"{prefix}{attribute}", module, attribute, tensor, True
        for attribute, tensor in module._buffers.items():
            if tensor is not None:
                yield f"{prefix}{attribute}", module, attribute, tensor, False


def restore_checkpoint_dtypes(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
) -> dict[str, int]:
    """Cast ``model``'s float tensors back to their stored dtypes, in place.

    Returns a count summary and logs it. Tensors absent from the checkpoint --
    tied weights, anything constructed at load time -- keep the dtype they were
    loaded with and are counted under ``missing``.
    """
    stored = read_stored_dtypes(checkpoint_dir)
    cast_counts: Counter[str] = Counter()
    summary = Counter({"cast": 0, "unchanged": 0, "missing": 0, "non_float": 0})

    for key, owner, attribute, tensor, is_parameter in _named_tensors(model):
        if not tensor.is_floating_point():
            summary["non_float"] += 1
            continue
        target = stored.get(key)
        if target is None:
            summary["missing"] += 1
            continue
        if tensor.dtype == target:
            summary["unchanged"] += 1
            continue
        if is_parameter:
            # Assigning ``.data`` keeps the Parameter object, so requires_grad
            # and any optimizer/LoRA bookkeeping that references it survive.
            tensor.data = tensor.data.to(target)
        else:
            owner._buffers[attribute] = tensor.to(target)
        cast_counts[str(target).removeprefix("torch.")] += 1
        summary["cast"] += 1

    cast_report = (
        ", ".join(f"{count} -> {dtype}" for dtype, count in sorted(cast_counts.items()))
        or "nothing to cast"
    )
    logger.info(
        "Restored checkpoint dtypes from %s: %s (%d already correct, "
        "%d not in the checkpoint, %d non-float)",
        Path(checkpoint_dir).name,
        cast_report,
        summary["unchanged"],
        summary["missing"],
        summary["non_float"],
    )
    return dict(summary)
