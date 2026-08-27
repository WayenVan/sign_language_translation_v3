import pytest
import torch
from torch import nn

from csi_slt.commands.train_ft_peft import apply_peft_trainability, cast_module_dtype


class _ModelWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = nn.Module()
        self.llm.base = nn.Linear(4, 4)
        self.llm.lora_A = nn.Linear(4, 2, bias=False)
        self.llm.lora_B = nn.Linear(2, 4, bias=False)
        self.visual_backbone = nn.Module()
        self.visual_backbone.base = nn.Linear(4, 4)
        self.visual_backbone.lora_A = nn.Linear(4, 2, bias=False)
        self.visual_backbone.lora_B = nn.Linear(2, 4, bias=False)
        self.visual_adapter = nn.Linear(3, 4)


def test_apply_peft_trainability_independently_selects_lora_components():
    model = _ModelWithLoRA()

    trainable_count = apply_peft_trainability(
        model,
        train_llm_lora=True,
        train_visual_lora=False,
        train_visual_adapter=False,
    )

    assert trainable_count == sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("llm.") and "lora_" in name
    )
    assert all(
        parameter.requires_grad == (name.startswith("llm.") and "lora_" in name)
        for name, parameter in model.named_parameters()
    )


def test_apply_peft_trainability_rejects_missing_requested_lora():
    model = _ModelWithLoRA()
    del model.visual_backbone.lora_A
    del model.visual_backbone.lora_B

    with pytest.raises(ValueError, match="visual LoRA training was requested"):
        apply_peft_trainability(
            model,
            train_llm_lora=False,
            train_visual_lora=True,
            train_visual_adapter=False,
        )


def test_apply_peft_trainability_freezes_existing_lora_for_adapter_only_training():
    model = _ModelWithLoRA()

    trainable_count = apply_peft_trainability(
        model,
        train_llm_lora=False,
        train_visual_lora=False,
        train_visual_adapter=True,
    )

    assert trainable_count == sum(
        parameter.numel() for parameter in model.visual_adapter.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.visual_adapter.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("visual_adapter.")
    )


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
