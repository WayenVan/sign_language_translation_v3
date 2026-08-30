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


def test_retired_dsid_config_keys_are_ignored():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        dsid_loss_weight=0.75,
        dsid_js_tau=0.08,
        dsid_warmup_ratio=0.2,
        dsid_decay_ratio=0.4,
    )

    for key in (
        "dsid_loss_weight",
        "dsid_js_tau",
        "dsid_warmup_ratio",
        "dsid_decay_ratio",
    ):
        assert not hasattr(config, key)


def test_retired_semantic_encoder_and_diversity_keys_are_ignored():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        visual_semantic_encoder_type="labse",
        visual_semantic_encoder_config={"encoder": {"id": "test"}},
        attention_diversity_loss_weight=0.01,
    )

    for key in (
        "visual_semantic_encoder_type",
        "visual_semantic_encoder_config",
        "attention_diversity_loss_weight",
    ):
        assert not hasattr(config, key)
