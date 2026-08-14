"""Collator for the text-only D-SID calibration data path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class DSIDCalibrationCollator:
    """Build mutually aligned pseudo-gloss and empty-source teacher batches."""

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("D-SID calibration batch cannot be empty")

        names = tuple(sample["id"] for sample in batch)
        teacher_paths = self.processor.process_dsid_teacher_paths(
            text=tuple(sample["text"] for sample in batch),
            src_lang=tuple(sample["lang"] for sample in batch),
            pseudo_gloss=tuple(sample["pseudo_gloss"] for sample in batch),
        )
        return {**teacher_paths.data, "names": names}
