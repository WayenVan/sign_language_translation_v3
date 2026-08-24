"""Prompt-bank loading and sampling for instruction-conditioned SLT."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PromptStrategy = Literal["random", "fixed", "manifest"]
_VIDEO_SENTINEL = "{{ video_start_token }}"


@dataclass(frozen=True)
class PromptRecord:
    """One validated prompt-bank entry."""

    id: str
    split: str
    intent: str
    instruction_lang: str
    target_lang: str | None
    template: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class PromptAssignment:
    """One manifest row pairing a flattened translation with a prompt."""

    eval_id: str
    translation_id: str
    prompt_id: str


class PromptSampler:
    """Load a JSONL prompt bank and select prompts under one of three policies.

    ``random``
        Select a prompt deterministically from ``seed``, ``epoch``, sample ID,
        target language, and (when supplied) instruction language. This gives a
        sample a stable prompt within an epoch and a new prompt across epochs.

    ``fixed``
        Select the lexicographically first matching prompt, unless ``prompt_id``
        is supplied to :meth:`sample`.

    ``manifest``
        Resolve an evaluation instance through a separate JSONL manifest. Each
        row contains ``eval_id``, ``translation_id``, and ``prompt_id``. A
        translation may appear in multiple rows so it can be evaluated under
        several prompts; only ``eval_id`` must be unique.
    """

    def __init__(
        self,
        prompt_path: str | Path,
        strategy: PromptStrategy = "random",
        *,
        seed: int = 0,
        manifest_path: str | Path | None = None,
        expected_split: str | None = None,
        supported_languages: tuple[str, ...] = ("de", "en", "zh"),
    ) -> None:
        if strategy not in {"random", "fixed", "manifest"}:
            raise ValueError(
                "strategy must be one of 'random', 'fixed', or 'manifest'"
            )
        if strategy == "manifest" and manifest_path is None:
            raise ValueError("manifest_path is required for manifest strategy")
        if strategy != "manifest" and manifest_path is not None:
            raise ValueError("manifest_path is only valid for manifest strategy")

        self.prompt_path = Path(prompt_path)
        self.strategy = strategy
        self.seed = int(seed)
        self.epoch = 0
        self.supported_languages = frozenset(supported_languages)
        if not self.supported_languages:
            raise ValueError("supported_languages must not be empty")

        self._records = self._load_prompt_bank(expected_split=expected_split)
        self._records_by_id = {record.id: record for record in self._records}
        self._records_by_cell: dict[tuple[str, str], tuple[PromptRecord, ...]] = (
            self._index_prompt_cells()
        )
        self._fallback_rng = random.Random(self.seed)
        self._manifest = (
            self._load_manifest(Path(manifest_path))
            if manifest_path is not None
            else None
        )

    @property
    def records(self) -> tuple[PromptRecord, ...]:
        return self._records

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic random sampling."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
        self._fallback_rng.seed(self.seed + self.epoch)

    def sample(
        self,
        target_lang: str | None,
        sample_id: str | int | None = None,
        *,
        instruction_lang: str | None = None,
        prompt_id: str | None = None,
        eval_id: str | None = None,
    ) -> PromptRecord:
        """Return one prompt matching the requested target/instruction language.

        ``prompt_id`` is an explicit override for ``random`` and ``fixed``. In
        manifest mode, ``eval_id`` selects an auditable translation/prompt
        assignment. ``sample_id`` remains a random-strategy seed and is not
        used as a manifest key.
        """

        self._validate_requested_language(target_lang, "target_lang", allow_none=True)
        self._validate_requested_language(
            instruction_lang, "instruction_lang", allow_none=True
        )

        if self.strategy == "manifest":
            if prompt_id is not None:
                raise ValueError("prompt_id override is not allowed in manifest mode")
            if eval_id is None:
                raise ValueError("eval_id is required in manifest mode")
            assignment = self.get_assignment(eval_id)
            return self._get_and_validate_explicit_prompt(
                assignment.prompt_id, target_lang, instruction_lang
            )

        if eval_id is not None:
            raise ValueError("eval_id is only valid in manifest mode")

        if prompt_id is not None:
            return self._get_and_validate_explicit_prompt(
                prompt_id, target_lang, instruction_lang
            )

        candidates = self._matching_candidates(target_lang, instruction_lang)
        if self.strategy == "fixed":
            return candidates[0]

        if sample_id is None:
            return self._choose_random_candidate(
                candidates,
                target_lang=target_lang,
                instruction_lang=instruction_lang,
                choice_value=self._fallback_rng.getrandbits(64),
            )

        choice_seed = self._stable_choice_seed(
            sample_id=str(sample_id),
            target_lang=target_lang,
            instruction_lang=instruction_lang,
        )
        return self._choose_random_candidate(
            candidates,
            target_lang=target_lang,
            instruction_lang=instruction_lang,
            choice_value=choice_seed,
        )

    def get_assignment(self, eval_id: str) -> PromptAssignment:
        """Return the manifest row for ``eval_id``.

        The collator can use ``translation_id`` to fetch/validate the flattened
        dataset row and :meth:`sample` to obtain the associated prompt.
        """

        if self.strategy != "manifest":
            raise RuntimeError("get_assignment is only available in manifest mode")
        if not isinstance(eval_id, str) or not eval_id:
            raise TypeError("eval_id must be a non-empty string")
        assignment = self._manifest.get(eval_id)  # type: ignore[union-attr]
        if assignment is None:
            raise KeyError(f"No prompt assignment for eval_id {eval_id!r}")
        return assignment

    @staticmethod
    def _choose_random_candidate(
        candidates: tuple[PromptRecord, ...],
        *,
        target_lang: str | None,
        instruction_lang: str | None,
        choice_value: int,
    ) -> PromptRecord:
        """Choose instruction language first, then a template within its cell.

        This keeps instruction-language sampling balanced even when prompt-bank
        cells contain different numbers of templates.
        """

        if instruction_lang is not None or target_lang is None:
            return candidates[choice_value % len(candidates)]

        by_instruction_language: defaultdict[str, list[PromptRecord]] = defaultdict(
            list
        )
        for record in candidates:
            by_instruction_language[record.instruction_lang].append(record)
        languages = sorted(by_instruction_language)
        selected_language = languages[choice_value % len(languages)]
        cell = sorted(
            by_instruction_language[selected_language], key=lambda record: record.id
        )
        return cell[(choice_value // len(languages)) % len(cell)]

    def _load_prompt_bank(self, expected_split: str | None) -> tuple[PromptRecord, ...]:
        rows = self._read_jsonl(self.prompt_path)
        if not rows:
            raise ValueError(f"Prompt bank is empty: {self.prompt_path}")

        records: list[PromptRecord] = []
        seen_ids: set[str] = set()
        required = {
            "id",
            "split",
            "intent",
            "instruction_lang",
            "target_lang",
            "template",
        }
        for line_number, row in rows:
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f"{self.prompt_path}:{line_number} missing fields: "
                    f"{sorted(missing)}"
                )

            prompt_id = row["id"]
            if not isinstance(prompt_id, str) or not prompt_id:
                raise TypeError(
                    f"{self.prompt_path}:{line_number} id must be a non-empty string"
                )
            if prompt_id in seen_ids:
                raise ValueError(f"Duplicate prompt id: {prompt_id}")
            seen_ids.add(prompt_id)

            split = row["split"]
            intent = row["intent"]
            instruction_lang = row["instruction_lang"]
            target_lang = row["target_lang"]
            template = row["template"]
            for name, value in (
                ("split", split),
                ("intent", intent),
                ("instruction_lang", instruction_lang),
                ("template", template),
            ):
                if not isinstance(value, str) or not value:
                    raise TypeError(
                        f"{self.prompt_path}:{line_number} {name} must be a "
                        "non-empty string"
                    )
            if target_lang is not None and not isinstance(target_lang, str):
                raise TypeError(
                    f"{self.prompt_path}:{line_number} target_lang must be a "
                    "string or null"
                )
            if expected_split is not None and split != expected_split:
                raise ValueError(
                    f"Prompt {prompt_id!r} has split {split!r}, expected "
                    f"{expected_split!r}"
                )
            self._validate_requested_language(
                instruction_lang, "instruction_lang", allow_none=False
            )
            self._validate_requested_language(
                target_lang, "target_lang", allow_none=True
            )
            if template.count(_VIDEO_SENTINEL) != 1:
                raise ValueError(
                    f"Prompt {prompt_id!r} must contain exactly one "
                    f"{_VIDEO_SENTINEL!r}"
                )

            metadata = {
                key: value
                for key, value in row.items()
                if key not in required
            }
            records.append(
                PromptRecord(
                    id=prompt_id,
                    split=split,
                    intent=intent,
                    instruction_lang=instruction_lang,
                    target_lang=target_lang,
                    template=template,
                    metadata=metadata,
                )
            )

        return tuple(records)

    def _index_prompt_cells(self) -> dict[tuple[str, str], tuple[PromptRecord, ...]]:
        cells: defaultdict[tuple[str, str], list[PromptRecord]] = defaultdict(list)
        for record in self._records:
            if record.target_lang is not None:
                cells[(record.instruction_lang, record.target_lang)].append(record)
        return {
            key: tuple(sorted(records, key=lambda record: record.id))
            for key, records in cells.items()
        }

    def _load_manifest(self, path: Path) -> dict[str, PromptAssignment]:
        assignments: dict[str, PromptAssignment] = {}
        for line_number, row in self._read_jsonl(path):
            eval_id = row.get("eval_id")
            translation_id = row.get("translation_id")
            prompt_id = row.get("prompt_id")
            if not isinstance(eval_id, str) or not eval_id:
                raise TypeError(
                    f"{path}:{line_number} eval_id must be a non-empty string"
                )
            if not isinstance(translation_id, (str, int)):
                raise TypeError(
                    f"{path}:{line_number} translation_id must be a string or integer"
                )
            if not isinstance(prompt_id, str) or not prompt_id:
                raise TypeError(
                    f"{path}:{line_number} prompt_id must be a non-empty string"
                )
            if eval_id in assignments:
                raise ValueError(f"Duplicate manifest eval_id: {eval_id}")
            if prompt_id not in self._records_by_id:
                raise ValueError(
                    f"Manifest references unknown prompt id {prompt_id!r}"
                )
            assignments[eval_id] = PromptAssignment(
                eval_id=eval_id,
                translation_id=str(translation_id),
                prompt_id=prompt_id,
            )
        if not assignments:
            raise ValueError(f"Prompt manifest is empty: {path}")
        return assignments

    def _matching_candidates(
        self, target_lang: str | None, instruction_lang: str | None
    ) -> tuple[PromptRecord, ...]:
        if target_lang is None:
            candidates = tuple(
                record
                for record in self._records
                if record.target_lang is None
                and (
                    instruction_lang is None
                    or record.instruction_lang == instruction_lang
                )
            )
        elif instruction_lang is not None:
            candidates = self._records_by_cell.get(
                (instruction_lang, target_lang), ()
            )
        else:
            candidates = tuple(
                record
                for (candidate_instruction_lang, candidate_target_lang), records in self._records_by_cell.items()
                if candidate_target_lang == target_lang
                for record in records
            )
        candidates = tuple(sorted(candidates, key=lambda record: record.id))
        if not candidates:
            raise KeyError(
                "No prompt candidates for "
                f"target_lang={target_lang!r}, instruction_lang={instruction_lang!r}"
            )
        return candidates

    def _get_and_validate_explicit_prompt(
        self,
        prompt_id: str,
        target_lang: str | None,
        instruction_lang: str | None,
    ) -> PromptRecord:
        record = self._records_by_id.get(prompt_id)
        if record is None:
            raise KeyError(f"Unknown prompt id: {prompt_id!r}")
        if target_lang is not None and record.target_lang != target_lang:
            raise ValueError(
                f"Prompt {prompt_id!r} targets {record.target_lang!r}, but sample "
                f"targets {target_lang!r}"
            )
        if instruction_lang is not None and record.instruction_lang != instruction_lang:
            raise ValueError(
                f"Prompt {prompt_id!r} uses instruction language "
                f"{record.instruction_lang!r}, expected {instruction_lang!r}"
            )
        return record

    def _stable_choice_seed(
        self,
        *,
        sample_id: str,
        target_lang: str | None,
        instruction_lang: str | None,
    ) -> int:
        payload = "\0".join(
            (
                str(self.seed),
                str(self.epoch),
                sample_id,
                target_lang or "<none>",
                instruction_lang or "<any>",
            )
        ).encode("utf-8")
        digest = hashlib.blake2b(
            payload, digest_size=8, person=b"sltprompt"
        ).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _validate_requested_language(
        self, value: str | None, field_name: str, *, allow_none: bool
    ) -> None:
        if value is None:
            if allow_none:
                return
            raise TypeError(f"{field_name} must be a string")
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string or None")
        if value not in self.supported_languages:
            raise ValueError(
                f"Unsupported {field_name} {value!r}; expected one of "
                f"{sorted(self.supported_languages)}"
            )

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
                    raise TypeError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                rows.append((line_number, row))
        return rows
