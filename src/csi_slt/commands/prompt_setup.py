"""Composition-root helpers for constructing split-specific prompt resolvers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from hydra.utils import instantiate

from csi_slt.engine.prompt_resolver import PromptResolver


def instantiate_prompt_resolvers(
    prompt_cfg: Mapping[str, object],
    splits: Iterable[str],
) -> dict[str, PromptResolver]:
    """Instantiate the configured resolver for every requested data split."""
    resolvers: dict[str, PromptResolver] = {}
    for split in splits:
        if split not in prompt_cfg:
            raise ValueError(f"Missing prompt configuration for split {split!r}.")
        resolvers[split] = instantiate(prompt_cfg[split], _convert_="all")
    return resolvers
