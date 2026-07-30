from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from accelerate import Accelerator
from accelerate.utils import gather_object

from csi_slt.engine.metrics import MetricContext


class XCometLiteMetric:
    """Distributed, reference-based XCOMET-Lite evaluation."""

    def __init__(
        self,
        accelerator: Accelerator,
        every_n_evaluations: int = 5,
        *,
        model_name: str = "myyycroft/XCOMET-lite",
        batch_size: int = 2,
        model: Any | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.every_n_evaluations = every_n_evaluations
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from xcomet.deberta_encoder import XCOMETLite
            except ImportError as exc:
                raise ImportError(
                    "XCOMET-Lite is unavailable. Make the official "
                    "NL2G/xCOMET-lite repository importable."
                ) from exc

            self._model = XCOMETLite.from_pretrained(self.model_name)

        return self._model

    def _predict(
        self,
        indexed_data: list[tuple[int, dict[str, str]]],
    ) -> list[tuple[int, float]]:
        if not indexed_data:
            return []

        output = self._load_model().predict(
            [sample for _, sample in indexed_data],
            batch_size=self.batch_size,
            gpus=int(self.accelerator.device.type == "cuda"),
        )
        return [
            (index, float(score))
            for (index, _), score in zip(indexed_data, output.scores, strict=True)
        ]

    @staticmethod
    def _aggregate(
        scores: Sequence[float],
        context: MetricContext,
    ) -> dict[str, float]:
        if not scores:
            return {
                "overall_macro_xcomet_lite": 0.0,
                "overall_weighted_xcomet_lite": 0.0,
            }

        score_array = np.asarray(scores, dtype=np.float64)
        metrics: dict[str, float] = {}
        language_scores: list[float] = []

        for language in sorted(context.language_groups):
            indices = context.language_groups[language]
            language_score = float(score_array[indices].mean())
            metric_language = language.strip().lower().replace("-", "_")
            metrics[f"{metric_language}_xcomet_lite"] = language_score
            language_scores.append(language_score)

        metrics["overall_macro_xcomet_lite"] = float(np.mean(language_scores))
        metrics["overall_weighted_xcomet_lite"] = float(score_array.mean())
        return metrics

    def compute(self, context: MetricContext) -> dict[str, float]:
        data = [
            (
                index,
                {"src": "", "mt": prediction, "ref": reference},
            )
            for index, (prediction, reference) in enumerate(
                zip(
                    context.batch.predictions,
                    context.batch.references,
                    strict=True,
                )
            )
        ]

        with self.accelerator.split_between_processes(data) as local_data:
            try:
                local_result = {
                    "scores": self._predict(local_data),
                    "error": None,
                }
            except Exception as exc:
                local_result = {
                    "scores": [],
                    "error": (
                        f"rank {self.accelerator.process_index}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

        results = gather_object([local_result])
        errors = [result["error"] for result in results if result["error"]]
        if errors:
            raise RuntimeError("XCOMET-Lite failed: " + "; ".join(errors))

        indexed_scores = [
            indexed_score
            for result in results
            for indexed_score in result["scores"]
        ]
        indexed_scores.sort(key=lambda item: item[0])
        return self._aggregate(
            [score for _, score in indexed_scores],
            context,
        )
