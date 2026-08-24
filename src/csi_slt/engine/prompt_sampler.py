"""Prompt-bank loading and selection for instruction-conditioned SLT."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


_VIDEO_SENTINEL = "{{ video_start_token }}"
_PROMPT_FIELDS = frozenset({"id", "target_lang", "template"})


@dataclass(frozen=True)
class PromptRecord:
    """One validated prompt-bank entry."""

    id: str
    target_lang: str
    template: str


class PromptSampler:
    """Load prompt banks and resolve training or evaluation prompts.

    Training calls :meth:`random` to select a prompt deterministically within
    the sample's target-language group. Standard evaluation calls :meth:`by_id`
    with the canonical ID configured for that target language. Expanded
    evaluations also call :meth:`by_id`, using the ``prompt_id`` carried by
    each expanded dataset row.

    Multiple JSONL paths may be loaded when one training condition draws from
    several non-overlapping banks. IDs must be globally unique across paths.
    """

    def __init__(
        self,
        prompt_paths: str | Path | Sequence[str | Path],
        *,
        seed: int = 0,
        supported_languages: tuple[str, ...] = ("de", "en", "zh"),
    ) -> None:
        self.prompt_paths = self._normalize_paths(prompt_paths)
        self.seed = int(seed)
        self.epoch = 0
        self._rng = random.Random(self.seed)
        self.supported_languages = frozenset(supported_languages)
        if not self.supported_languages:
            raise ValueError("supported_languages must not be empty")

        self._records = self._load_prompt_banks()
        self._records_by_id = {record.id: record for record in self._records}
        grouped: defaultdict[str, list[PromptRecord]] = defaultdict(list)
        for record in self._records:
            grouped[record.target_lang].append(record)
        self._records_by_target = {
            language: tuple(sorted(records, key=lambda record: record.id))
            for language, records in grouped.items()
        }

    @property
    def records(self) -> tuple[PromptRecord, ...]:
        return self._records

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by subsequent deterministic training samples."""

        if not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self._rng.seed(self.seed + self.epoch)

    def by_id(
        self, prompt_id: str, *, target_lang: str | None = None
    ) -> PromptRecord:
        """Return an exact prompt, optionally checking the sample language."""

        if not isinstance(prompt_id, str) or not prompt_id:
            raise TypeError("prompt_id must be a non-empty string")
        record = self._records_by_id.get(prompt_id)
        if record is None:
            raise KeyError(f"Unknown prompt id: {prompt_id!r}")
        if target_lang is not None:
            self._validate_target_language(target_lang)
            if record.target_lang != target_lang:
                raise ValueError(
                    f"Prompt {prompt_id!r} targets {record.target_lang!r}, but "
                    f"sample targets {target_lang!r}"
                )
        return record

    def random(
        self,
        target_lang: str,
    ) -> PromptRecord:
        """Randomly select a prompt from one target-language training group.

        Every call advances this sampler's RNG, so repeated occurrences of the
        same training example may receive different prompts. Call
        :meth:`set_epoch` at epoch boundaries to seed the epoch's draw sequence.
        """

        self._validate_target_language(target_lang)
        candidates = self._records_by_target.get(target_lang, ())
        if not candidates:
            raise KeyError(f"No prompts for target_lang={target_lang!r}")
        return self._rng.choice(candidates)

    def _load_prompt_banks(self) -> tuple[PromptRecord, ...]:
        records: list[PromptRecord] = []
        seen_ids: dict[str, Path] = {}
        for path in self.prompt_paths:
            rows = self._read_jsonl(path)
            if not rows:
                raise ValueError(f"Prompt bank is empty: {path}")
            for line_number, row in rows:
                fields = frozenset(row)
                missing = _PROMPT_FIELDS - fields
                extra = fields - _PROMPT_FIELDS
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} missing fields: {sorted(missing)}"
                    )
                if extra:
                    raise ValueError(
                        f"{path}:{line_number} unsupported fields: {sorted(extra)}"
                    )

                prompt_id = row["id"]
                target_lang = row["target_lang"]
                template = row["template"]
                if not isinstance(prompt_id, str) or not prompt_id:
                    raise TypeError(
                        f"{path}:{line_number} id must be a non-empty string"
                    )
                if prompt_id in seen_ids:
                    raise ValueError(
                        f"Duplicate prompt id {prompt_id!r} in {path}; already "
                        f"defined in {seen_ids[prompt_id]}"
                    )
                if not isinstance(target_lang, str):
                    raise TypeError(
                        f"{path}:{line_number} target_lang must be a string"
                    )
                self._validate_target_language(
                    target_lang, location=f"{path}:{line_number}"
                )
                if not isinstance(template, str) or not template:
                    raise TypeError(
                        f"{path}:{line_number} template must be a non-empty string"
                    )
                if template.count(_VIDEO_SENTINEL) != 1:
                    raise ValueError(
                        f"Prompt {prompt_id!r} must contain exactly one "
                        f"{_VIDEO_SENTINEL!r}"
                    )

                seen_ids[prompt_id] = path
                records.append(
                    PromptRecord(
                        id=prompt_id,
                        target_lang=target_lang,
                        template=template,
                    )
                )
        return tuple(records)

    def _validate_target_language(
        self, target_lang: str, *, location: str | None = None
    ) -> None:
        prefix = f"{location} " if location else ""
        if not isinstance(target_lang, str):
            raise TypeError(f"{prefix}target_lang must be a string")
        if target_lang not in self.supported_languages:
            raise ValueError(
                f"{prefix}unsupported target_lang {target_lang!r}; expected one "
                f"of {sorted(self.supported_languages)}"
            )

    @staticmethod
    def _normalize_paths(
        prompt_paths: str | Path | Sequence[str | Path],
    ) -> tuple[Path, ...]:
        if isinstance(prompt_paths, (str, Path)):
            paths = (Path(prompt_paths),)
        else:
            paths = tuple(Path(path) for path in prompt_paths)
        if not paths:
            raise ValueError("prompt_paths must contain at least one path")
        return paths

    @staticmethod
    def _read_jsonl(path: Path) -> list[tuple[int, dict[str, object]]]:
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        rows: list[tuple[int, dict[str, object]]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {error.msg}"
                    ) from error
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_number} must contain a JSON object")
                rows.append((line_number, row))
        return rows
