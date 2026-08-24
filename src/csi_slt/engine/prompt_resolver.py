"""Resolve a prompt for one dataset row under a configured evaluation policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from csi_slt.engine.prompt_sampler import PromptRecord, PromptSampler


class PromptResolver(Protocol):
    """Common interface consumed by a future data collator."""

    def resolve(
        self, row: Mapping[str, object], *, epoch: int | None = None
    ) -> PromptRecord:
        """Resolve the prompt assigned to one dataset row."""


class RandomPromptResolver:
    """Sample deterministically within each row's target-language group."""

    def __init__(
        self,
        sampler: PromptSampler,
        *,
        target_lang_column: str = "lang",
    ) -> None:
        self.sampler = sampler
        self.target_lang_column = target_lang_column

    def resolve(
        self, row: Mapping[str, object], *, epoch: int | None = None
    ) -> PromptRecord:
        target_lang = _required_string(row, self.target_lang_column)
        if epoch is not None and epoch != self.sampler.epoch:
            self.sampler.set_epoch(epoch)
        return self.sampler.random(target_lang)


class FixedPromptResolver:
    """Resolve one explicitly configured prompt ID per target language."""

    def __init__(
        self,
        sampler: PromptSampler,
        prompt_ids: Mapping[str, str],
        *,
        target_lang_column: str = "lang",
    ) -> None:
        self.sampler = sampler
        self.prompt_ids = dict(prompt_ids)
        self.target_lang_column = target_lang_column
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        for target_lang, prompt_id in self.prompt_ids.items():
            if not isinstance(target_lang, str) or not target_lang:
                raise TypeError("prompt_ids keys must be non-empty strings")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise TypeError("prompt_ids values must be non-empty strings")
            self.sampler.by_id(prompt_id, target_lang=target_lang)

    def resolve(
        self, row: Mapping[str, object], *, epoch: int | None = None
    ) -> PromptRecord:
        del epoch
        target_lang = _required_string(row, self.target_lang_column)
        try:
            prompt_id = self.prompt_ids[target_lang]
        except KeyError as error:
            raise KeyError(
                f"No fixed prompt configured for target_lang={target_lang!r}"
            ) from error
        return self.sampler.by_id(prompt_id, target_lang=target_lang)


class PromptIdFromRowResolver:
    """Resolve ``prompt_id`` directly from an expanded evaluation row."""

    def __init__(
        self,
        sampler: PromptSampler,
        *,
        prompt_id_column: str = "prompt_id",
        target_lang_column: str = "lang",
    ) -> None:
        self.sampler = sampler
        self.prompt_id_column = prompt_id_column
        self.target_lang_column = target_lang_column

    def resolve(
        self, row: Mapping[str, object], *, epoch: int | None = None
    ) -> PromptRecord:
        del epoch
        prompt_id = _required_string(row, self.prompt_id_column)
        target_lang = _required_string(row, self.target_lang_column)
        return self.sampler.by_id(prompt_id, target_lang=target_lang)


def _required_value(row: Mapping[str, object], column: str) -> object:
    try:
        return row[column]
    except KeyError as error:
        raise KeyError(f"dataset row is missing required column {column!r}") from error


def _required_string(row: Mapping[str, object], column: str) -> str:
    value = _required_value(row, column)
    if not isinstance(value, str) or not value:
        raise TypeError(f"row[{column!r}] must be a non-empty string")
    return value
