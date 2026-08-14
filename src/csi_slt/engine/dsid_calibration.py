"""Offline calibration of the D-SID Jensen-Shannon threshold."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from csi_slt.modeling_slt.dsid import (
    DSIDTeacherStatistics,
    compute_dsid_teacher_statistics,
)

logger = logging.getLogger(__name__)

_TEACHER_PREFIXES = ("pseudo_gloss_teacher", "empty_source_teacher")
_TEACHER_FIELDS = ("input_ids", "attention_mask", "position_ids", "labels")


@dataclass(frozen=True)
class DSIDCalibrationResult:
    """Calibrated threshold and dataset-level teacher diagnostics."""

    tau: float
    quantile: float
    interpolation: str
    sample_count: int
    valid_token_count: int
    mean_js: float
    min_js: float
    max_js: float
    gloss_nll: float
    empty_nll: float
    teacher_nll_gain: float
    gate_coverage: float
    passed_sanity_checks: bool
    stop_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["stop_reasons"] = list(self.stop_reasons)
        return data


class DSIDTauCalibrator:
    """Aggregate exact token-level statistics and select the JS quantile."""

    supported_interpolations = {
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    }

    def __init__(
        self,
        *,
        quantile: float = 0.75,
        interpolation: str = "linear",
        min_tau: float = 1e-8,
        min_gate_coverage: float = 0.0,
    ) -> None:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        if interpolation not in self.supported_interpolations:
            raise ValueError(
                f"unsupported interpolation {interpolation!r}; expected one of "
                f"{sorted(self.supported_interpolations)}"
            )
        if min_tau < 0.0:
            raise ValueError("min_tau must be non-negative")
        if not 0.0 <= min_gate_coverage <= 1.0:
            raise ValueError("min_gate_coverage must be in [0, 1]")

        self.quantile = float(quantile)
        self.interpolation = interpolation
        self.min_tau = float(min_tau)
        self.min_gate_coverage = float(min_gate_coverage)
        self._js_chunks: list[torch.Tensor] = []
        self._sample_count = 0
        self._valid_token_count = 0
        self._gloss_nll_sum = 0.0
        self._empty_nll_sum = 0.0
        self._gate_count = 0

    def update(self, statistics: DSIDTeacherStatistics) -> None:
        """Accumulate one batch without applying the direction gate to JS."""
        token_count = statistics.js.numel()
        if token_count == 0:
            raise ValueError("D-SID statistics batch contains no valid target tokens")
        for name, values in (
            ("js", statistics.js),
            ("gloss_nll", statistics.gloss_nll),
            ("empty_nll", statistics.empty_nll),
        ):
            if values.numel() != token_count:
                raise ValueError(
                    f"{name} contains {values.numel()} values, expected {token_count}"
                )
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"{name} contains non-finite values")
        if statistics.direction_gate.numel() != token_count:
            raise ValueError(
                "direction_gate length does not match the number of JS values"
            )
        if int(statistics.valid_counts.sum()) != token_count:
            raise ValueError("valid_counts do not sum to the number of JS values")

        # Only one scalar per valid target position is retained on CPU. Full
        # vocabulary logits and distributions are released after each batch.
        self._js_chunks.append(statistics.js.detach().to("cpu", torch.float32))
        self._sample_count += statistics.valid_counts.numel()
        self._valid_token_count += token_count
        self._gloss_nll_sum += float(statistics.gloss_nll.detach().sum().cpu())
        self._empty_nll_sum += float(statistics.empty_nll.detach().sum().cpu())
        self._gate_count += int(statistics.direction_gate.detach().sum().cpu())

    def finalize(self) -> DSIDCalibrationResult:
        """Compute the requested quantile and dataset-level sanity checks."""
        if not self._js_chunks:
            raise RuntimeError("cannot calibrate D-SID tau without any statistics")

        js = torch.cat(self._js_chunks)
        tau = float(
            torch.quantile(
                js,
                self.quantile,
                interpolation=self.interpolation,
            )
        )
        token_count = self._valid_token_count
        gloss_nll = self._gloss_nll_sum / token_count
        empty_nll = self._empty_nll_sum / token_count
        teacher_nll_gain = empty_nll - gloss_nll
        gate_coverage = self._gate_count / token_count

        stop_reasons = []
        if tau < self.min_tau:
            stop_reasons.append(
                f"calibrated tau {tau:.8g} is below min_tau {self.min_tau:.8g}"
            )
        if teacher_nll_gain <= 0.0:
            stop_reasons.append(
                "pseudo-gloss teacher does not improve mean target-token NLL "
                f"(gain={teacher_nll_gain:.8g})"
            )
        if gate_coverage < self.min_gate_coverage:
            stop_reasons.append(
                f"direction-gate coverage {gate_coverage:.6f} is below "
                f"min_gate_coverage {self.min_gate_coverage:.6f}"
            )

        return DSIDCalibrationResult(
            tau=tau,
            quantile=self.quantile,
            interpolation=self.interpolation,
            sample_count=self._sample_count,
            valid_token_count=token_count,
            mean_js=float(js.mean()),
            min_js=float(js.min()),
            max_js=float(js.max()),
            gloss_nll=gloss_nll,
            empty_nll=empty_nll,
            teacher_nll_gain=teacher_nll_gain,
            gate_coverage=gate_coverage,
            passed_sanity_checks=not stop_reasons,
            stop_reasons=tuple(stop_reasons),
        )


def _teacher_input_device(teacher: torch.nn.Module) -> torch.device:
    get_input_embeddings = getattr(teacher, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None and weight.device.type != "meta":
            return weight.device
    try:
        return next(teacher.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _path_forward(
    teacher: torch.nn.Module,
    batch: Mapping[str, Any],
    prefix: str,
    device: torch.device,
) -> torch.Tensor:
    inputs = {
        field: batch[f"{prefix}_{field}"].to(device)
        for field in _TEACHER_FIELDS
        if field != "labels"
    }
    return teacher(**inputs, use_cache=False).logits


@torch.inference_mode()
def calibrate_dsid_tau(
    teacher: torch.nn.Module,
    dataloader,
    calibrator: DSIDTauCalibrator,
    *,
    device: torch.device | str | None = None,
    log_every: int = 100,
) -> DSIDCalibrationResult:
    """Run both teacher paths over a data loader and calibrate D-SID tau."""
    if log_every < 0:
        raise ValueError("log_every must be non-negative")
    input_device = (
        torch.device(device) if device is not None else _teacher_input_device(teacher)
    )
    missing_fields = {
        f"{prefix}_{field}" for prefix in _TEACHER_PREFIXES for field in _TEACHER_FIELDS
    }

    was_training = teacher.training
    teacher.eval()
    disable_adapter = getattr(teacher, "disable_adapter", None)
    adapter_context = disable_adapter() if callable(disable_adapter) else nullcontext()
    try:
        with adapter_context:
            for batch_index, batch in enumerate(dataloader, start=1):
                missing = sorted(missing_fields.difference(batch))
                if missing:
                    raise KeyError(
                        "D-SID calibration batch is missing fields: "
                        + ", ".join(missing)
                    )
                gloss_logits = _path_forward(
                    teacher, batch, "pseudo_gloss_teacher", input_device
                )
                empty_logits = _path_forward(
                    teacher, batch, "empty_source_teacher", input_device
                )
                statistics = compute_dsid_teacher_statistics(
                    gloss_logits,
                    batch["pseudo_gloss_teacher_labels"].to(input_device),
                    empty_logits,
                    batch["empty_source_teacher_labels"].to(input_device),
                )
                calibrator.update(statistics)
                if log_every and batch_index % log_every == 0:
                    logger.info("calibrated %d batches", batch_index)
                del gloss_logits, empty_logits, statistics
    finally:
        teacher.train(was_training)

    return calibrator.finalize()
