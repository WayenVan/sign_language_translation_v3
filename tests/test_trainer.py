from types import SimpleNamespace

import torch
from torch import nn
from transformers import Seq2SeqTrainingArguments

from csi_slt.engine.trainer import SltTrainer


class _MeanReducedModelWithKwargs(nn.Module):
    """Minimal model reproducing Transformers' ``**kwargs`` inference."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace()
        self.last_kwargs = None

    def forward(self, labels=None, **kwargs):
        self.last_kwargs = kwargs
        return {"loss": self.weight * 0 + 8.0}


def test_trainer_does_not_claim_model_handles_num_items_in_batch(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
    )

    model = _MeanReducedModelWithKwargs()
    trainer = SltTrainer(model=model, args=args)

    assert trainer.args.average_tokens_across_devices is True
    assert trainer.model_accepts_loss_kwargs is False

    loss = trainer.compute_loss(
        model,
        {"labels": torch.tensor([[1]])},
        num_items_in_batch=torch.tensor(1),
    )

    assert loss.item() == 8.0
    assert "num_items_in_batch" not in model.last_kwargs
