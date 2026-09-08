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


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)


class _BlockEncoder(nn.Module):
    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        # Nested like C-RADIO (radio_model.blocks.N.qkv) so PEFT's
        # ``.*\.{pattern}\.(\d+)\.`` layer match sees a dot before "blocks".
        self.radio_model = nn.Module()
        self.radio_model.blocks = nn.ModuleList(_Block() for _ in range(num_blocks))


class _BlockBackbone(nn.Module):
    """Backbone whose blocks feed features from an intermediate output_layer."""

    def __init__(self, num_blocks: int, output_layer: int) -> None:
        super().__init__()
        self.visual_encoder = _BlockEncoder(num_blocks)
        self.output_layer = output_layer

    def resolve_lora_layers(self, spec):
        from csi_slt.modeling_slt.lora_layers import (
            find_transformer_layers,
            last_n_layer_indices,
            normalize_layer_spec,
        )

        normalized = normalize_layer_spec(spec)
        if normalized is None:
            return None
        blocks = find_transformer_layers(self.visual_encoder)
        end = (
            self.output_layer
            if self.output_layer >= 0
            else len(blocks) + self.output_layer
        )
        return last_n_layer_indices(
            len(blocks), normalized["count"], end=end
        )


def test_output_layer_anchored_span_injects_into_the_right_blocks():
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(visual_lora=False, visual_lora_config={})
    model.visual_backbone = _BlockBackbone(num_blocks=12, output_layer=-4)

    model.inject_visual_lora(
        LoraConfig(r=2, lora_alpha=4, target_modules=["qkv"]),
        layer_spec={"pattern": "blocks", "anchor": "output_layer", "count": 3},
    )

    # output_layer -4 on 12 blocks -> span ends at block 8; count 3 -> {6,7,8}.
    assert model.config.visual_lora_config["layers_to_transform"] == [6, 7, 8]
    assert model.config.visual_lora_config["layers_pattern"] == "blocks"
    adapted = {
        name.split(".qkv")[0]
        for name, _ in model.visual_backbone.visual_encoder.named_parameters()
        if "lora_" in name
    }
    assert adapted == {
        "radio_model.blocks.6",
        "radio_model.blocks.7",
        "radio_model.blocks.8",
    }


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


def test_layer_spec_is_resolved_via_backbone_and_serialized(monkeypatch):
    model = _model_shell()
    seen = {}

    def fake_resolve(spec):
        seen["spec"] = spec
        return [5, 6]

    model.visual_backbone.resolve_lora_layers = fake_resolve
    monkeypatch.setattr(
        model, "_inject_visual_lora", lambda cfg: seen.setdefault("cfg", cfg)
    )

    model.inject_visual_lora(
        LoraConfig(r=2, target_modules=["proj"]),
        layer_spec={"anchor": "output_layer", "count": 2},
    )

    assert seen["spec"] == {"anchor": "output_layer", "count": 2}
    assert seen["cfg"].layers_to_transform == [5, 6]
    assert model.config.visual_lora_config["layers_to_transform"] == [5, 6]


def test_layer_spec_resolving_to_none_leaves_layers_unset(monkeypatch):
    model = _model_shell()
    model.visual_backbone.resolve_lora_layers = lambda spec: None
    captured = {}
    monkeypatch.setattr(
        model, "_inject_visual_lora", lambda cfg: captured.setdefault("cfg", cfg)
    )

    model.inject_visual_lora(
        LoraConfig(r=2, target_modules=["proj"]), layer_spec={"count": 4}
    )

    assert captured["cfg"].layers_to_transform is None


def test_layer_spec_conflicts_with_explicit_layers_to_transform():
    model = _model_shell()
    model.visual_backbone.resolve_lora_layers = lambda spec: [1]

    with pytest.raises(ValueError, match="not both"):
        model.inject_visual_lora(
            LoraConfig(
                r=2,
                target_modules=["proj"],
                layers_pattern="blocks",
                layers_to_transform=[1],
            ),
            layer_spec={"count": 1},
        )


def test_layer_spec_requires_backbone_support():
    model = _model_shell()  # _VisualBackbone has no resolve_lora_layers

    with pytest.raises(TypeError, match="resolve_lora_layers"):
        model.inject_visual_lora(
            LoraConfig(r=2, target_modules=["proj"]), layer_spec={"count": 1}
        )
