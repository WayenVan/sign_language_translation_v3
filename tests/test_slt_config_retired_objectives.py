import pytest
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from csi_slt.configuration_slt.configuration import SltConfig


def _llm_config():
    return Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
        vocab_size=32,
    )


def test_retired_contrastive_and_alignment_config_keys_are_ignored():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        visual_backbone_type="test",
        visual_adapter_type="test",
        contrastive_loss_weight=0.25,
        contrastive_text_queue_size=1024,
        alignment_loss_weight=1.0,
        alignment_pooling_distill_weight=0.5,
    )

    assert not hasattr(config, "contrastive_loss_weight")
    assert not hasattr(config, "contrastive_text_queue_size")
    assert not hasattr(config, "alignment_loss_weight")
    assert not hasattr(config, "alignment_pooling_distill_weight")


def test_dsid_config_requires_a_calibrated_tau_when_enabled():
    with pytest.raises(ValueError, match="dsid_js_tau must be provided"):
        SltConfig(
            llm_model_name_or_path="Qwen/Qwen3-test",
            llm_config=_llm_config(),
            dsid_loss_weight=1.0,
        )


def test_dsid_config_accepts_valid_loss_values():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        dsid_loss_weight=0.75,
        dsid_js_tau=0.08,
    )

    assert config.dsid_loss_weight == 0.75
    assert config.dsid_js_tau == 0.08


def test_attention_diversity_loss_weight_is_configurable():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        attention_diversity_loss_weight=0.01,
    )

    assert config.attention_diversity_loss_weight == 0.01


@pytest.mark.parametrize("weight", (-0.1, True, "0.001"))
def test_attention_diversity_loss_weight_rejects_invalid_values(weight):
    error = ValueError if weight == -0.1 else TypeError
    with pytest.raises(error, match="attention_diversity_loss_weight"):
        SltConfig(
            llm_model_name_or_path="Qwen/Qwen3-test",
            llm_config=_llm_config(),
            attention_diversity_loss_weight=weight,
        )


def test_old_model_level_dsid_schedule_keys_are_ignored():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        dsid_warmup_ratio=0.2,
        dsid_decay_ratio=0.4,
    )

    assert not hasattr(config, "dsid_warmup_ratio")
    assert not hasattr(config, "dsid_decay_ratio")
