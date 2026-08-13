from types import SimpleNamespace

from torch import nn

from csi_slt.engine.callbacks import ModelInfoCallback


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
