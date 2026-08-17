import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

import csi_slt.modeling_slt.slt as slt_module
from csi_slt.configuration_slt.configuration import SltConfig
from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized
from csi_slt.modeling_slt.output_utils import (
    VisualAdapterOutput,
    VisualBackboneOutput,
)
from csi_slt.modeling_slt.slt import SltModel


class _Backbone(nn.Module):
    observed = None

    @classmethod
    def from_pretrained_backbone(cls, config, dtype="auto"):
        cls.observed = (config, dtype)
        backbone = cls()
        mark_module_tree_as_initialized(backbone)
        return backbone

    def forward(self, video, video_length):
        return VisualBackboneOutput(
            visual_features=video.new_zeros((video.shape[0], 8)),
            visual_length=video_length,
            extras={"backbone": True},
        )


class _Adapter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, visual_output, **kwargs):
        return VisualAdapterOutput(
            visual_features=visual_output.visual_features,
            visual_length=visual_output.visual_length,
        )


class _SemanticEncoder(nn.Module):
    observed = None

    def __init__(self):
        super().__init__()
        self.config = {"embedded_model_config": True}
        self.output_dim = 8

    @classmethod
    def from_pretrained_encoder(cls, config, dtype="auto"):
        cls.observed = (config, dtype)
        return cls()

    def forward(self, visual_output):
        return VisualAdapterOutput(
            visual_features=visual_output.visual_features + 1.0,
            visual_length=visual_output.visual_length,
            position_ids=visual_output.position_ids,
            extras=visual_output.extras,
        )


class _LLMFactory:
    observed = None

    @classmethod
    def from_pretrained(cls, model_id, dtype="auto"):
        cls.observed = (model_id, dtype)
        return Qwen3ForCausalLM(
            Qwen3Config(
                hidden_size=8,
                intermediate_size=16,
                num_attention_heads=2,
                num_hidden_layers=1,
                num_key_value_heads=2,
                vocab_size=32,
            )
        )


def _config(semantic_type="test_semantic"):
    llm_config = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
        vocab_size=32,
    )
    return SltConfig(
        video_soft_token_id=7,
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=llm_config,
        visual_backbone_type="test_backbone",
        visual_backbone_config={"id": "test"},
        visual_adapter_type="test_adapter",
        visual_adapter_kwargs={},
        visual_semantic_encoder_type=semantic_type,
        visual_semantic_encoder_config={"source": "test"},
    )


def test_from_pretrained_components_loads_and_applies_semantic_encoder(monkeypatch):
    monkeypatch.setitem(slt_module.VISUAL_BACKBONES, "test_backbone", _Backbone)
    monkeypatch.setitem(slt_module.VISUAL_ADAPTERS, "test_adapter", _Adapter)
    monkeypatch.setitem(
        slt_module.VISUAL_SEMANTIC_ENCODERS,
        "test_semantic",
        _SemanticEncoder,
    )
    monkeypatch.setattr(slt_module, "get_llm_cls_by_model_name", lambda _: _LLMFactory)
    config = _config()

    model = SltModel.from_pretrained_components(
        config,
        llm_dtype=torch.bfloat16,
        visual_backbone_dtype=torch.float16,
        visual_semantic_encoder_dtype=torch.float32,
    )
    output = model.get_visual_feats(
        torch.zeros(3, 3, 2, 2),
        torch.tensor([1, 2]),
    )

    assert _Backbone.observed == ({"id": "test"}, torch.float16)
    assert _SemanticEncoder.observed == ({"source": "test"}, torch.float32)
    assert _LLMFactory.observed == ("Qwen/Qwen3-test", torch.bfloat16)
    assert model.visual_semantic_encoder is not None
    assert config.visual_semantic_encoder_config == {"embedded_model_config": True}
    torch.testing.assert_close(output.visual_features, torch.ones(3, 8))


def test_get_visual_feats_skips_semantic_encoder_when_disabled():
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    model.visual_backbone = _Backbone()
    model.visual_adapter = _Adapter()
    model.visual_semantic_encoder = None

    output = model.get_visual_feats(torch.zeros(2, 3, 2, 2), torch.tensor([2]))

    torch.testing.assert_close(output.visual_features, torch.zeros(2, 8))
