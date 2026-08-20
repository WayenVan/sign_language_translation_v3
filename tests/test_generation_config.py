from types import SimpleNamespace

import pytest
from transformers import GenerationConfig

from csi_slt.modeling_slt.slt import SltModel
from csi_slt.utils.generation_config import merge_generation_config


def test_merge_generation_config_copies_base_and_applies_overrides():
    base = GenerationConfig(max_length=512, max_new_tokens=None, do_sample=False)

    merged = merge_generation_config(base, {"max_new_tokens": 128})

    assert merged is not base
    assert merged.max_length == 512
    assert merged.max_new_tokens == 128
    assert base.max_new_tokens is None


def test_merge_generation_config_rejects_unknown_overrides():
    base = GenerationConfig()

    with pytest.raises(ValueError, match="unknown_option"):
        merge_generation_config(base, {"unknown_option": True})


def test_slt_model_preserves_model_specific_multiple_eos_ids():
    model = SimpleNamespace(
        llm=SimpleNamespace(
            generation_config=GenerationConfig(
                bos_token_id=2,
                eos_token_id=[1, 106, 50],
                pad_token_id=0,
            )
        ),
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
                layer_types=["sliding_attention"],
            )
        ),
    )

    SltModel._configure_generation(model)

    assert model.generation_config.eos_token_id == [1, 106, 50]
