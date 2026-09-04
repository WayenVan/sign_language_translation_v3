"""Guards for the bidirectional video-token attention overlay.

The overlay is built by ``create_causal_mask`` from the *language model's*
configuration. Transformers resolves ``_attn_implementation`` on a config only
when a model is instantiated from it, and never serializes it, so a config
object that no model ever used keeps ``None`` -- and ``create_causal_mask``
then early-exits and returns no mask at all, degrading the model to plain
causal attention without any warning. These tests pin both halves: the two
configs stay one object, and a mask that lost the overlay is rejected loudly.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from csi_slt.configuration_slt.configuration import SltConfig
from csi_slt.modeling_slt.slt import SltModel, token_type_ids_mask_function


# One prompt token, four video tokens, three text tokens.
TOKEN_TYPE_IDS = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0]])


def _llm_config(attn_implementation="sdpa"):
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
        vocab_size=32,
    )
    config._attn_implementation = attn_implementation
    return config


def _slt_config(**kwargs):
    return SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        visual_backbone_type="test",
        visual_adapter_type="test",
        ctc_vocab_size=4,
        ctc_blank_id=0,
        **kwargs,
    )


def _model_shell(attn_implementation="sdpa", bidirectional=True) -> SltModel:
    """Build the smallest object the mask validators need.

    Constructing a real ``SltModel`` would pull in the visual backbone, which
    these tests do not exercise.
    """
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    model.llm = SimpleNamespace(config=_llm_config(attn_implementation))
    model.config = SimpleNamespace(video_bidirectional_attention=bidirectional)
    model._bidirectional_mask_validated = False
    return model


def _build_mask(attn_implementation, *, with_overlay):
    config = _llm_config(attn_implementation)
    batch_size, length = TOKEN_TYPE_IDS.shape
    overlay = (
        token_type_ids_mask_function(TOKEN_TYPE_IDS) if with_overlay else None
    )
    return create_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(batch_size, length, config.hidden_size),
        attention_mask=torch.ones(batch_size, length, dtype=torch.long),
        past_key_values=None,
        position_ids=None,
        or_mask_function=overlay,
        # Without the overlay a plain causal mask is skippable, and skipping it
        # yields `None`. Materialize it so the caller gets what the
        # FlashAttention path produces: a real mask that lost the overlay.
        allow_is_causal_skip=with_overlay,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_checkpoints_predating_the_flag_keep_causal_attention():
    # A deserialized configuration always carries `transformers_version`.
    config = _slt_config(transformers_version="5.15.0")

    assert config.video_bidirectional_attention is False


def test_a_freshly_built_configuration_enables_the_overlay():
    assert _slt_config().video_bidirectional_attention is True


def test_an_explicit_value_wins_over_the_compatibility_default():
    config = _slt_config(
        transformers_version="5.15.0",
        video_bidirectional_attention=True,
    )

    assert config.video_bidirectional_attention is True


def test_the_flag_survives_serialization():
    config = _slt_config()

    assert SltConfig(**config.to_dict()).video_bidirectional_attention is True


def test_the_language_model_config_is_declared_as_a_sub_config():
    # This is what makes Transformers propagate `_attn_implementation` (and the
    # dtype plan) from the SLT config into the embedded language-model config.
    assert "llm_config" in SltConfig.sub_configs


def test_flash_attention_is_declared_unsupported():
    # The overlay is an arbitrary 4D mask; `flash_attention_mask` ignores the
    # mask function entirely, so Transformers must refuse the request outright.
    assert SltModel._supports_flash_attn is False
    assert SltModel._supports_sdpa is True
    assert SltModel._supports_flex_attn is True


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", ["sdpa", "eager", "flex_attention"])
def test_construction_accepts_implementations_that_carry_custom_masks(
    implementation,
):
    _model_shell(implementation)._validate_attention_support()


@pytest.mark.parametrize(
    "implementation", ["flash_attention_2", "flash_attention_3"]
)
def test_construction_rejects_implementations_that_drop_the_overlay(
    implementation,
):
    with pytest.raises(ValueError, match="video_bidirectional_attention"):
        _model_shell(implementation)._validate_attention_support()


def test_construction_ignores_the_attention_check_when_the_overlay_is_off():
    _model_shell(
        "flash_attention_2", bidirectional=False
    )._validate_attention_support()


# ---------------------------------------------------------------------------
# Per-forward validation
# ---------------------------------------------------------------------------


def test_a_missing_mask_is_rejected():
    # The exact regression: an unresolved config made `create_causal_mask`
    # return None, and the language model fell back to plain causal attention.
    with pytest.raises(RuntimeError, match="returned None"):
        _model_shell()._validate_bidirectional_mask(None, TOKEN_TYPE_IDS)


def test_a_causal_only_mask_is_rejected():
    # A non-None mask is not enough: the FlashAttention builder returns the 2D
    # padding mask and silently drops the overlay.
    mask = _build_mask("sdpa", with_overlay=False)
    assert mask is not None

    with pytest.raises(RuntimeError, match="does not implement"):
        _model_shell()._validate_bidirectional_mask(mask, TOKEN_TYPE_IDS)


@pytest.mark.parametrize("implementation", ["sdpa", "eager", "flex_attention"])
def test_the_real_overlay_passes_validation(implementation):
    mask = _build_mask(implementation, with_overlay=True)
    model = _model_shell(implementation)

    model._validate_bidirectional_mask(mask, TOKEN_TYPE_IDS)

    assert model._bidirectional_mask_validated is True


@pytest.mark.parametrize("implementation", ["sdpa", "eager", "flex_attention"])
def test_the_overlay_opens_video_pairs_without_leaking_future_text(
    implementation,
):
    mask = _build_mask(implementation, with_overlay=True)
    device = TOKEN_TYPE_IDS.device

    # Video token 1 attends forward to video token 4 ...
    assert SltModel._mask_allows(mask, 0, 1, 4, device)
    # ... but never forward to the text that follows the video block.
    assert not SltModel._mask_allows(mask, 0, 1, 5, device)
    # Text still attends back over the whole video block.
    assert SltModel._mask_allows(mask, 0, 5, 1, device)


def test_the_structural_check_runs_only_once():
    model = _model_shell()
    model._validate_bidirectional_mask(
        _build_mask("sdpa", with_overlay=True), TOKEN_TYPE_IDS
    )

    # A later batch is only checked for `None`, so the accelerator is not
    # synchronized on every forward -- a mask that would have failed the
    # structural check now passes.
    model._validate_bidirectional_mask(
        _build_mask("sdpa", with_overlay=False), TOKEN_TYPE_IDS
    )
