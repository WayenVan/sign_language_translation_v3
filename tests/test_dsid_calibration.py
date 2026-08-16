from types import SimpleNamespace

import pytest
import torch
from torch import nn

from csi_slt.data.collators import DSIDCalibrationCollator
from csi_slt.data.dsid_calibration import DSIDCalibrationDataset
from csi_slt.engine.sft.dsid_calibration import (
    DSIDTauCalibrator,
    calibrate_dsid_tau,
)
from csi_slt.modeling_slt.dsid import (
    DSIDTeacherStatistics,
    compute_dsid_teacher_statistics,
)


class _TextAwareDataset:
    def __init__(self):
        self.text_reads = 0
        self.video_reads = 0

    def __len__(self):
        return 1

    def get_text_item(self, index):
        self.text_reads += 1
        return {
            "id": f"sample-{index}",
            "text": "target",
            "lang": "en",
            "pseudo_gloss": "GLOSS",
            "ignored": "value",
        }

    def __getitem__(self, index):
        self.video_reads += 1
        raise AssertionError(f"video path must not be read for sample {index}")


def test_calibration_dataset_prefers_text_only_dataset_path():
    source = _TextAwareDataset()

    item = DSIDCalibrationDataset(source)[0]

    assert item == {
        "id": "sample-0",
        "text": "target",
        "lang": "en",
        "pseudo_gloss": "GLOSS",
    }
    assert source.text_reads == 1
    assert source.video_reads == 0


class _RecordingProcessor:
    def __init__(self):
        self.kwargs = None

    def process_dsid_teacher_paths(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(data={"teacher_tensor": torch.tensor([1])})


def test_calibration_collator_builds_only_teacher_paths():
    processor = _RecordingProcessor()
    collator = DSIDCalibrationCollator(processor)

    result = collator(
        [
            {
                "id": "a",
                "text": "first",
                "lang": "en",
                "pseudo_gloss": "ONE",
            },
            {
                "id": "b",
                "text": "second",
                "lang": "de",
                "pseudo_gloss": "TWO",
            },
        ]
    )

    assert result["names"] == ("a", "b")
    assert result["teacher_tensor"].tolist() == [1]
    assert processor.kwargs == {
        "text": ("first", "second"),
        "src_lang": ("en", "de"),
        "pseudo_gloss": ("ONE", "TWO"),
    }


def _path_logits(labels, target_logits):
    logits = torch.zeros((*labels.shape, len(target_logits[0])))
    logits[:, :-1][labels[:, 1:].ne(-100)] = torch.tensor(target_logits)
    return logits


def test_teacher_statistics_align_different_source_lengths():
    gloss_labels = torch.tensor([[-100, -100, -100, 0, 1]])
    empty_labels = torch.tensor([[-100, 0, 1]])

    statistics = compute_dsid_teacher_statistics(
        _path_logits(gloss_labels, [[3.0, 0.0], [0.0, 3.0]]),
        gloss_labels,
        _path_logits(empty_labels, [[0.0, 0.0], [0.0, 0.0]]),
        empty_labels,
    )

    assert statistics.target_ids.tolist() == [0, 1]
    assert statistics.valid_counts.tolist() == [2]
    assert statistics.js.shape == (2,)
    assert statistics.direction_gate.tolist() == [True, True]
    assert bool((statistics.teacher_nll_gain > 0).all())


def _statistics(
    js=(0.1, 0.2, 0.8, 0.9),
    *,
    gloss_nll=(1.0, 1.0, 1.0, 1.0),
    empty_nll=(2.0, 2.0, 2.0, 2.0),
):
    return DSIDTeacherStatistics(
        js=torch.tensor(js),
        # Deliberately exclude the high-JS positions: tau must not use this gate.
        direction_gate=torch.tensor([True, True, False, False]),
        gloss_nll=torch.tensor(gloss_nll),
        empty_nll=torch.tensor(empty_nll),
        target_ids=torch.tensor([0, 1, 2, 3]),
        valid_counts=torch.tensor([2, 2]),
    )


def test_tau_is_p75_of_all_valid_positions_without_direction_filtering():
    calibrator = DSIDTauCalibrator(quantile=0.75)
    calibrator.update(_statistics())

    result = calibrator.finalize()

    assert result.tau == pytest.approx(0.825)
    assert result.sample_count == 2
    assert result.valid_token_count == 4
    assert result.teacher_nll_gain == pytest.approx(1.0)
    assert result.gate_coverage == pytest.approx(0.5)
    assert result.passed_sanity_checks


def test_calibrator_reports_non_improving_teacher_as_stop_reason():
    calibrator = DSIDTauCalibrator()
    calibrator.update(
        _statistics(
            gloss_nll=(2.0, 2.0, 2.0, 2.0),
            empty_nll=(1.0, 1.0, 1.0, 1.0),
        )
    )

    result = calibrator.finalize()

    assert not result.passed_sanity_checks
    assert "does not improve" in result.stop_reasons[0]


class _TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 2)
        self.observations = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        self.observations.append((self.training, torch.is_grad_enabled(), kwargs))
        logits = torch.zeros((*input_ids.shape, 2), device=input_ids.device)
        if bool(input_ids.eq(9).any()):
            logits[..., 0] = 3.0
        return SimpleNamespace(logits=logits)


def test_calibration_runner_uses_eval_inference_and_restores_teacher_mode():
    teacher = _TinyTeacher()
    teacher.train()
    batch = {
        "pseudo_gloss_teacher_input_ids": torch.tensor([[4, 9, 0]]),
        "pseudo_gloss_teacher_attention_mask": torch.ones((1, 3), dtype=torch.long),
        "pseudo_gloss_teacher_position_ids": torch.arange(3).unsqueeze(0),
        "pseudo_gloss_teacher_labels": torch.tensor([[-100, -100, 0]]),
        "empty_source_teacher_input_ids": torch.tensor([[4, 0]]),
        "empty_source_teacher_attention_mask": torch.ones((1, 2), dtype=torch.long),
        "empty_source_teacher_position_ids": torch.arange(2).unsqueeze(0),
        "empty_source_teacher_labels": torch.tensor([[-100, 0]]),
    }

    result = calibrate_dsid_tau(
        teacher,
        [batch],
        DSIDTauCalibrator(),
        log_every=0,
        show_progress=False,
    )

    assert result.sample_count == 1
    assert result.valid_token_count == 1
    assert result.teacher_nll_gain > 0.0
    assert teacher.training
    assert len(teacher.observations) == 2
    assert all(not training for training, _, _ in teacher.observations)
    assert all(not grad_enabled for _, grad_enabled, _ in teacher.observations)
    assert all(
        observation_kwargs["use_cache"] is False
        for _, _, observation_kwargs in teacher.observations
    )
