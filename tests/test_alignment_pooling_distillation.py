import torch
import pytest
from transformers.models.qwen3 import Qwen3Config

from csi_slt.configuration_slt.configuration import SltConfig
from csi_slt.modeling_slt.slt import SltModel


def test_alignment_pooling_distillation_prefers_non_null_positions():
    alignment = torch.tensor(
        [[[0.0, 0.5], [0.5, 0.0]]], requires_grad=True
    )
    visual_mask = torch.ones(1, 2, dtype=torch.long)
    good_logits = torch.tensor([[8.0, -8.0]], requires_grad=True)
    bad_logits = torch.tensor([[-8.0, 8.0]], requires_grad=True)

    good_loss = SltModel._alignment_pooling_distillation_loss(
        alignment,
        good_logits.softmax(dim=-1),
        visual_mask,
    )
    bad_loss = SltModel._alignment_pooling_distillation_loss(
        alignment,
        bad_logits.softmax(dim=-1),
        visual_mask,
    )

    assert good_loss < 1e-4
    assert bad_loss > 10.0
    bad_loss.backward()
    assert bad_logits.grad is not None
    assert alignment.grad is None


def test_alignment_pooling_distillation_ignores_padding():
    alignment = torch.tensor(
        [[[0.0, 0.5], [0.0, 0.5], [0.0, 0.0]]]
    )
    visual_mask = torch.tensor([[1, 1, 0]])
    student_attention = torch.tensor([[0.5, 0.5, 0.0]])

    loss = SltModel._alignment_pooling_distillation_loss(
        alignment,
        student_attention,
        visual_mask,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_alignment_pooling_distillation_requires_alignment_training():
    with pytest.raises(ValueError, match="alignment_loss_weight must be positive"):
        SltConfig(
            llm_config=Qwen3Config(hidden_size=8, num_attention_heads=1),
            alignment_loss_weight=0.0,
            alignment_pooling_distill_weight=0.1,
        )
