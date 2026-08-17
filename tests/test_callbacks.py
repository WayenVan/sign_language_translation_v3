from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

from csi_slt.engine.sft.callbacks import (
    EvalInformationVisualizationCallback,
    ModelInfoCallback,
    TrainSubsetMetricsCallback,
)
from csi_slt.modeling_slt.info_utils import InformationOutput


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(4)
        self.norm.bias.requires_grad_(False)


def test_model_info_callback_prints_aligned_module_classes(capsys):
    trainer = SimpleNamespace(
        model=ExampleModel(),
        accelerator=SimpleNamespace(is_local_main_process=True),
    )

    ModelInfoCallback().on_train_begin(None, None, None, trainer=trainer)
    output = capsys.readouterr().out

    assert "Module" in output
    assert "Class" in output
    assert "encoder" in output
    assert "Linear" in output
    assert "norm" in output
    assert "LayerNorm" in output
    assert "Total: 24" in output
    assert "Trainable: 20 (83.33%)" in output
    assert "Frozen: 4" in output

    table_lines = [line for line in output.splitlines() if line.startswith(("+", "|"))]
    assert len({len(line) for line in table_lines}) == 1


def test_eval_information_callback_uses_evaluation_cadence_and_flat_paths(
    tmp_path, monkeypatch
):
    information = InformationOutput(
        llm_attentions=(torch.ones(1, 3, 3),),
        llm_visual_mask=torch.tensor([[False, True, False]]),
    )
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(is_main_process=True),
        collect_eval_information=lambda count: [
            {
                "sample_index": 0,
                "attention_mask": torch.ones(3, dtype=torch.long),
                "information": information,
            }
        ],
    )
    rendered_paths = []
    monkeypatch.setattr(
        "csi_slt.engine.sft.callbacks.render_llm_attention",
        lambda attention, visual_mask, output_path: rendered_paths.append(output_path),
    )
    callback = EvalInformationVisualizationCallback(
        every_n_evaluations=2,
        num_samples=1,
    )
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=120)

    callback.on_evaluate(args, state, None, trainer=trainer)
    assert rendered_paths == []

    callback.on_evaluate(args, state, None, trainer=trainer)
    assert rendered_paths == [
        Path(tmp_path) / "eval_info_step120" / "sample000_layer-1_llm_attention.png"
    ]
    assert rendered_paths[0].parent.is_dir()
    assert callback.state()["attributes"]["evaluation_count"] == 2


def test_train_subset_metrics_callback_uses_evaluation_cadence():
    calls = []
    logged = []
    trainer = SimpleNamespace(
        evaluate_train_subset=lambda **kwargs: calls.append(kwargs)
        or {"train_probe_bleu": 12.0},
        log=lambda metrics: logged.append(metrics),
    )
    callback = TrainSubsetMetricsCallback(
        every_n_evaluations=2,
        num_samples=200,
        seed=7,
    )

    callback.on_evaluate(None, None, None, trainer=trainer)
    assert calls == []
    callback.on_evaluate(None, None, None, trainer=trainer)

    assert calls == [
        {"num_samples": 200, "seed": 7, "metric_key_prefix": "train_probe"}
    ]
    assert logged == [{"train_probe_bleu": 12.0}]
    assert callback.state()["attributes"]["evaluation_count"] == 2
