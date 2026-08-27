import pytest
import torch
from torch import nn

from csi_slt.commands.train_ft_peft import cast_module_dtype, freeze_except_lora_adapters


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
    with pytest.raises(ValueError, match="No LoRA parameters"):
        freeze_except_lora_adapters(nn.Linear(3, 4))


def test_freeze_except_lora_adapters_supports_adapter_only_training():
    model = _ModelWithLoRA()
    del model.lora_A
    del model.lora_B

    trainable_count = freeze_except_lora_adapters(
        model,
        unfreeze_visual_adapter=True,
    )

    assert trainable_count == sum(
        parameter.numel() for parameter in model.visual_adapter.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.visual_adapter.parameters())


def test_cast_module_dtype_casts_floating_parameters():
    module = nn.Linear(3, 4)

    cast_module_dtype(module, "bfloat16")

    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_cast_module_dtype_auto_preserves_checkpoint_dtype():
    module = nn.Linear(3, 4).to(dtype=torch.float64)

    cast_module_dtype(module, "auto")

    assert all(parameter.dtype == torch.float64 for parameter in module.parameters())


def test_cast_module_dtype_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        cast_module_dtype(nn.Linear(3, 4), "not_a_dtype")
