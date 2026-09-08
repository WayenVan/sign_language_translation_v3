"""Unit tests for LoRA layer-span resolution."""

import pytest
from torch import nn

from peft import LoraConfig

from csi_slt.modeling_slt.lora_layers import (
    apply_layer_spec,
    find_transformer_layers,
    first_n_layer_indices,
    last_n_layer_indices,
    normalize_layer_spec,
    resolve_layers_to_transform,
)


class _Encoder(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(n))


class _Decoder(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n))


def test_last_n_defaults_to_the_final_blocks():
    assert last_n_layer_indices(27, 4) == [23, 24, 25, 26]


def test_last_n_ends_at_an_explicit_intermediate_block():
    # output_layer -8 on a 27-block backbone -> span ends at block 19.
    assert last_n_layer_indices(27, 4, end=19) == [16, 17, 18, 19]


def test_last_n_none_count_selects_every_layer():
    assert last_n_layer_indices(27, None) is None


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_last_n_rejects_non_positive_int_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        last_n_layer_indices(27, count)


def test_last_n_rejects_span_running_off_the_front():
    with pytest.raises(ValueError, match="only 3 block"):
        last_n_layer_indices(27, 4, end=2)


def test_last_n_rejects_anchor_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        last_n_layer_indices(27, 2, end=27)


def test_first_n_counts_from_the_input_side():
    assert first_n_layer_indices(27, 4) == [0, 1, 2, 3]


def test_first_n_none_count_selects_every_layer():
    assert first_n_layer_indices(27, None) is None


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_first_n_rejects_non_positive_int_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        first_n_layer_indices(27, count)


def test_first_n_rejects_span_deeper_than_the_module():
    with pytest.raises(ValueError, match="only 27"):
        first_n_layer_indices(27, 28)


def test_normalize_keeps_first_anchor():
    assert normalize_layer_spec({"anchor": "first", "count": 4}) == {
        "anchor": "first",
        "count": 4,
        "pattern": None,
    }


def test_resolve_against_live_module_first_anchor():
    assert resolve_layers_to_transform(
        _Decoder(6), {"anchor": "first", "count": 2}
    ) == [0, 1]


def test_resolve_first_anchor_none_count_selects_every_layer():
    assert resolve_layers_to_transform(
        _Decoder(6), {"anchor": "first", "count": None}
    ) is None


def test_normalize_defaults_anchor_to_last():
    assert normalize_layer_spec({"count": 4}) == {
        "anchor": "last",
        "count": 4,
        "pattern": None,
    }


def test_normalize_keeps_output_layer_anchor():
    assert normalize_layer_spec({"anchor": "output_layer", "count": 8}) == {
        "anchor": "output_layer",
        "count": 8,
        "pattern": None,
    }


def test_normalize_passes_none_through():
    assert normalize_layer_spec(None) is None


def test_normalize_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys: stride"):
        normalize_layer_spec({"count": 4, "stride": 2})


def test_normalize_requires_count():
    with pytest.raises(ValueError, match="must set 'count'"):
        normalize_layer_spec({"anchor": "last"})


def test_normalize_rejects_unknown_anchor():
    with pytest.raises(ValueError, match="anchor must be one of"):
        normalize_layer_spec({"anchor": "middle", "count": 4})


def test_normalize_carries_pattern_and_defaults_it_to_none():
    assert normalize_layer_spec({"count": 4})["pattern"] is None
    assert normalize_layer_spec({"count": 4, "pattern": "blocks"})["pattern"] == (
        "blocks"
    )


def test_normalize_rejects_non_string_pattern():
    with pytest.raises(TypeError, match="'pattern' must be a string"):
        normalize_layer_spec({"count": 4, "pattern": 3})


def test_apply_layer_spec_sets_transform_and_pattern_together():
    config = LoraConfig(r=2, target_modules=["q_proj"])
    apply_layer_spec(
        config, [1, 2], {"anchor": "last", "count": 2, "pattern": "blocks"}
    )
    assert config.layers_to_transform == [1, 2]
    assert config.layers_pattern == "blocks"


def test_apply_layer_spec_without_pattern_leaves_it_unset():
    config = LoraConfig(r=2, target_modules=["q_proj"])
    apply_layer_spec(config, [3], {"anchor": "last", "count": 1, "pattern": None})
    assert config.layers_to_transform == [3]
    assert config.layers_pattern is None


def test_apply_layer_spec_ignores_none_indices():
    config = LoraConfig(r=2, target_modules=["q_proj"])
    apply_layer_spec(config, None, {"anchor": "last", "count": 1, "pattern": "x"})
    assert config.layers_to_transform is None
    assert config.layers_pattern is None


def test_find_transformer_layers_locates_blocks_and_decoder_layers():
    assert len(find_transformer_layers(_Encoder(5))) == 5
    assert len(find_transformer_layers(_Decoder(3))) == 3


def test_find_transformer_layers_raises_when_absent():
    with pytest.raises(TypeError, match="Could not locate"):
        find_transformer_layers(nn.Linear(2, 2))


def test_resolve_against_live_module_last_anchor():
    assert resolve_layers_to_transform(_Decoder(6), {"count": 2}) == [4, 5]


def test_resolve_requires_end_index_for_output_layer_anchor():
    with pytest.raises(ValueError, match="no end index"):
        resolve_layers_to_transform(
            _Encoder(6), {"anchor": "output_layer", "count": 2}
        )


def test_resolve_uses_supplied_end_index_for_output_layer_anchor():
    assert resolve_layers_to_transform(
        _Encoder(27), {"anchor": "output_layer", "count": 4}, end_index=19
    ) == [16, 17, 18, 19]
