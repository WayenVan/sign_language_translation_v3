from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig
from torch import nn

from csi_slt.modeling_slt.slt import SltModel


class _VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs)


class _VisualBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = _VisualEncoder()


def _model_shell() -> SltModel:
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(visual_lora=False, visual_lora_config={})
    model.visual_backbone = _VisualBackbone()
    return model


def test_visual_lora_is_injected_in_place_and_recorded_in_config():
    model = _model_shell()
    encoder = model.visual_backbone.visual_encoder

    model.inject_visual_lora(
        LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])
    )

    assert model.visual_backbone.visual_encoder is encoder
    assert model.config.visual_lora is True
    assert model.config.visual_lora_config["r"] == 2
    assert any("lora_" in name for name, _ in encoder.named_parameters())


def test_visual_lora_rejects_duplicate_injection():
    model = _model_shell()
    config = LoraConfig(r=2, target_modules=["proj"])
    model.inject_visual_lora(config)

    with pytest.raises(ValueError, match="already contains visual LoRA"):
        model.inject_visual_lora(config)


def test_visual_lora_structure_can_be_rebuilt_before_loading_weights():
    model = _model_shell()
    model.inject_visual_lora(LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.fill_(0.25)
    expected_state = model.state_dict()

    reloaded = _model_shell()
    reloaded._inject_visual_lora(LoraConfig(**model.config.visual_lora_config))
    load_result = reloaded.load_state_dict(expected_state, strict=True)

    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    for name, parameter in reloaded.named_parameters():
        if "lora_" in name:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))


def test_visual_lora_requires_a_visual_encoder():
    model = _model_shell()
    model.visual_backbone = nn.Linear(4, 4)

    with pytest.raises(TypeError, match="does not expose visual_encoder"):
        model.inject_visual_lora(LoraConfig(r=2, target_modules=["weight"]))
