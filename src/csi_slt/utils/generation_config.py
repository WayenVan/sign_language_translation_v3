from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from transformers.generation.configuration_utils import GenerationConfig


def merge_generation_config(
    base: GenerationConfig,
    overrides: Mapping[str, Any],
) -> GenerationConfig:
    """Copy a model generation config and apply validated overrides."""
    merged = deepcopy(base)
    unused = merged.update(**overrides)
    if unused:
        names = ", ".join(sorted(unused))
        raise ValueError(f"Unknown generation config arguments: {names}")
    merged.validate(strict=True)
    return merged
