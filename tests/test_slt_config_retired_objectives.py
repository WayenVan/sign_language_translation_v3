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


@pytest.mark.parametrize(
    ("ctc_kwargs", "message"),
    [
        ({"ctc_blank_id": 0}, "ctc_vocab_size must be an int"),
        ({"ctc_vocab_size": 4}, "ctc_blank_id must be an int"),
    ],
)
def test_ctc_structure_is_mandatory(ctc_kwargs, message):
    with pytest.raises(TypeError, match=message):
        SltConfig(
            llm_model_name_or_path="Qwen/Qwen3-test",
            llm_config=_llm_config(),
            **ctc_kwargs,
        )


def test_retired_ctc_enabled_switch_is_ignored():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        ctc_enabled=False,
        ctc_vocab_size=4,
        ctc_blank_id=0,
    )

    assert not hasattr(config, "ctc_enabled")


@pytest.mark.parametrize(
    "mode", ["soft", "straight_through", "argmax"]
)
def test_ctc_codebook_training_modes_are_serialized(mode):
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        ctc_vocab_size=4,
        ctc_blank_id=0,
        ctc_codebook_training_mode=mode,
        ctc_codebook_default_temperature=0.5,
    )

    assert config.ctc_codebook_training_mode == mode
    assert config.ctc_codebook_default_temperature == 0.5


def test_ctc_hidden_size_defaults_to_llm_width_and_can_be_decoupled():
    default_config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        ctc_vocab_size=4,
        ctc_blank_id=0,
    )
    decoupled_config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        ctc_vocab_size=4,
        ctc_blank_id=0,
        ctc_hidden_size=8,
    )

    assert default_config.ctc_hidden_size == 16
    assert decoupled_config.hidden_size == 16
    assert decoupled_config.ctc_hidden_size == 8


def test_ctc_hidden_size_must_match_declared_adapter_output_width():
    with pytest.raises(ValueError, match="output_dim must match ctc_hidden_size"):
        SltConfig(
            llm_model_name_or_path="Qwen/Qwen3-test",
            llm_config=_llm_config(),
            visual_adapter_kwargs={"output_dim": 16},
            ctc_vocab_size=4,
            ctc_blank_id=0,
            ctc_hidden_size=8,
        )


def test_ctc_codebook_rejects_invalid_runtime_config():
    with pytest.raises(ValueError, match="ctc_codebook_training_mode"):
        SltConfig(
            llm_model_name_or_path="Qwen/Qwen3-test",
            llm_config=_llm_config(),
            ctc_vocab_size=4,
            ctc_blank_id=0,
            ctc_codebook_training_mode="unknown",
        )
    with pytest.raises(ValueError, match="default_temperature"):
        SltConfig(
            llm_model_name_or_path="Qwen/Qwen3-test",
            llm_config=_llm_config(),
            ctc_vocab_size=4,
            ctc_blank_id=0,
            ctc_codebook_default_temperature=0.01,
        )


def test_retired_contrastive_and_alignment_config_keys_are_ignored():
    config = SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        visual_backbone_type="test",
        visual_adapter_type="test",
        ctc_vocab_size=4,
        ctc_blank_id=0,
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
        ctc_vocab_size=4,
        ctc_blank_id=0,
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
        ctc_vocab_size=4,
        ctc_blank_id=0,
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
