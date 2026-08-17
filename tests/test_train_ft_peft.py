import pytest
from torch import nn

from csi_slt.commands.train_ft_peft import freeze_except_lora_adapters


class _ModelWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_adapter = nn.Linear(3, 4)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)


def test_freeze_except_lora_adapters_only_keeps_lora_trainable():
    model = _ModelWithLoRA()

    trainable_count = freeze_except_lora_adapters(model)

    assert trainable_count == sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in model.named_parameters()
    )


def test_freeze_except_lora_adapters_rejects_model_without_lora():
    with pytest.raises(ValueError, match="No LoRA adapter parameters"):
        freeze_except_lora_adapters(nn.Linear(3, 4))
