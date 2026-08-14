"""Standalone command for calibrating D-SID tau on the training split."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from csi_slt.data.collators import DSIDCalibrationCollator
from csi_slt.data.datamodule import DataModule
from csi_slt.data.dsid_calibration import DSIDCalibrationDataset
from csi_slt.engine.dsid_calibration import (
    DSIDTauCalibrator,
    calibrate_dsid_tau,
)
from csi_slt.modeling_slt.slt import get_llm_cls_by_model_name

DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))


def _load_teacher(cfg: DictConfig):
    model_name = cfg.model.config.llm_model_name_or_path
    model_cls = get_llm_cls_by_model_name(model_name)
    load_kwargs = {}
    if cfg.teacher.get("dtype") is not None:
        load_kwargs["dtype"] = cfg.teacher.dtype
    if cfg.teacher.get("device_map") is not None:
        load_kwargs["device_map"] = cfg.teacher.device_map
    teacher = model_cls.from_pretrained(model_name, **load_kwargs)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def _save_result(
    cfg: DictConfig,
    result,
    *,
    dataset_size: int,
    calibrated_dataset_size: int,
) -> Path:
    output_path = Path(cfg.calibration.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "model_name_or_path": cfg.model.config.llm_model_name_or_path,
            "dtype": cfg.teacher.get("dtype"),
            "device_map": cfg.teacher.get("device_map"),
        },
        "dataset": {
            "split": "train",
            "dataset_size": dataset_size,
            "calibrated_sample_limit": cfg.calibration.get("max_samples"),
            "calibrated_dataset_size": calibrated_dataset_size,
            "used_full_training_split": calibrated_dataset_size == dataset_size,
        },
        "result": result.to_dict(),
        "suggested_override": (f"model.config.dsid_js_tau={result.tau:.10g}"),
    }
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name="dsid_calibration/base",
)
def main(cfg: DictConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "D-SID tau calibration must run as one process so every target "
            "token contributes exactly once to the global quantile"
        )

    model_name = cfg.model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Calibration bypasses SltModel.forward() completely: there are no
    # pixel_values, video decoding, visual backbone, or visual adapter. The
    # text-only processor builds pseudo-gloss and empty-source inputs, which are
    # forwarded directly through the frozen base LLM to obtain q_g and q_0.
    # DataModule.setup() is intentionally skipped because calibration also does
    # not need video lengths or length-bucket preparation.
    datamodule = DataModule(cfg.data, cfg.datamodule, tokenizer=tokenizer)
    source_dataset = instantiate(cfg.data.train.dataset)
    dataset_size = len(source_dataset)
    calibration_dataset = DSIDCalibrationDataset(source_dataset)

    max_samples = cfg.calibration.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("calibration.max_samples must be positive or null")
        calibration_dataset = Subset(
            calibration_dataset,
            range(min(max_samples, dataset_size)),
        )
    calibrated_dataset_size = len(calibration_dataset)
    if calibrated_dataset_size == 0:
        raise ValueError("training dataset contains no samples to calibrate")

    num_workers = int(cfg.calibration.num_workers)
    dataloader = DataLoader(
        calibration_dataset,
        batch_size=int(cfg.calibration.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.calibration.pin_memory),
        persistent_workers=num_workers > 0,
        collate_fn=DSIDCalibrationCollator(datamodule.processor),
    )

    teacher = _load_teacher(cfg)
    calibrator = DSIDTauCalibrator(
        quantile=float(cfg.calibration.quantile),
        interpolation=str(cfg.calibration.interpolation),
        min_tau=float(cfg.calibration.min_tau),
        min_gate_coverage=float(cfg.calibration.min_gate_coverage),
    )
    result = calibrate_dsid_tau(
        teacher,
        dataloader,
        calibrator,
        device=cfg.calibration.get("device"),
        log_every=int(cfg.calibration.log_every),
        show_progress=bool(cfg.calibration.show_progress),
    )
    output_path = _save_result(
        cfg,
        result,
        dataset_size=dataset_size,
        calibrated_dataset_size=calibrated_dataset_size,
    )

    print(f"D-SID tau: {result.tau:.10g}")
    print(f"Teacher NLL gain: {result.teacher_nll_gain:.8f}")
    print(f"Direction-gate coverage: {result.gate_coverage:.2%}")
    print(f"Saved calibration artifact: {output_path}")
    print(f"Suggested override: model.config.dsid_js_tau={result.tau:.10g}")

    if bool(cfg.calibration.fail_on_sanity_check) and not result.passed_sanity_checks:
        reasons = "; ".join(result.stop_reasons)
        raise RuntimeError(f"D-SID calibration failed sanity checks: {reasons}")


if __name__ == "__main__":
    main()
