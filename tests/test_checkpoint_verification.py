from pathlib import Path

import pytest
import torch

from csi_slt.utils import checkpoint_verification


class TinyModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))
        self.register_buffer("counter", torch.tensor([3]))


def test_verify_model_checkpoint_reports_exact_match(monkeypatch, tmp_path: Path):
    model = TinyModel(1.5)
    monkeypatch.setattr(
        checkpoint_verification.SltModel,
        "from_pretrained",
        lambda *args, **kwargs: (TinyModel(1.5), {}),
    )
    report_path = tmp_path / "report.json"

    report = checkpoint_verification.verify_model_checkpoint(
        model, tmp_path / "checkpoint-100", report_path=report_path
    )

    assert report["equal"] is True
    assert report["tensors_checked"] == 2
    assert report_path.is_file()


def test_verify_model_checkpoint_reports_difference(monkeypatch, tmp_path: Path):
    model = TinyModel(1.5)
    monkeypatch.setattr(
        checkpoint_verification.SltModel,
        "from_pretrained",
        lambda *args, **kwargs: (TinyModel(2.0), {}),
    )
    report_path = tmp_path / "report.json"

    with pytest.raises(AssertionError, match="different_tensors=.*weight"):
        checkpoint_verification.verify_model_checkpoint(
            model, tmp_path / "checkpoint-100", report_path=report_path
        )

    report = report_path.read_text(encoding="utf-8")
    assert '"equal": false' in report
    assert '"mismatched_elements": 1' in report
    assert '"max_abs_diff": 0.5' in report
