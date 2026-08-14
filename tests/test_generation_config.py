import pytest
from transformers import GenerationConfig

from csi_slt.experiment.generation_config import merge_generation_config


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
