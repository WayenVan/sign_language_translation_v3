"""Guards for the three visual position-encoding modes.

The language model already applies RoPE over the whole prefill, so the table
added here is a *second*, adapter-side position signal whose usefulness on a
7k-video training set is an open question. ``visual_position_embedding_type``
makes the three answers switchable -- keep the learned table, drop the signal
entirely, or replace it with a fixed sinusoidal one -- and these tests pin what
each mode puts in the model, what it puts in the checkpoint, and that the
default still reproduces every existing run.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from csi_slt.configuration_slt.configuration import SltConfig
from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS, VISUAL_BACKBONES
from csi_slt.modeling_slt.slt import SltModel


HIDDEN_SIZE = 16


def _llm_config():
    config = Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
        vocab_size=32,
    )
    config._attn_implementation = "sdpa"
    return config


def _slt_config(**kwargs):
    return SltConfig(
        llm_model_name_or_path="Qwen/Qwen3-test",
        llm_config=_llm_config(),
        visual_backbone_type="test",
        visual_adapter_type="test",
        video_soft_token_id=0,
        ctc_vocab_size=4,
        ctc_blank_id=0,
        **kwargs,
    )


class _StubVisualComponent(nn.Module):
    """Stand-in for the backbone/adapter, which these tests never run."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.projection = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)


@pytest.fixture
def registered_stub_components():
    VISUAL_BACKBONES["test"] = _StubVisualComponent
    VISUAL_ADAPTERS["test"] = _StubVisualComponent
    try:
        yield
    finally:
        del VISUAL_BACKBONES["test"]
        del VISUAL_ADAPTERS["test"]


def _model(position_embedding_type, registered_stub_components) -> SltModel:
    config = _slt_config(visual_position_embedding_type=position_embedding_type)
    llm = Qwen3ForCausalLM(_llm_config())
    mark_module_tree_as_initialized(llm)
    return SltModel(config, llm=llm)


def _shell(position_embedding_type, table=None, embedding=None) -> SltModel:
    """The smallest object ``visual_position_embedding_forward`` needs."""
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        visual_position_embedding_type=position_embedding_type
    )
    model.visual_position_embedding = embedding
    model.register_buffer("visual_position_table", table, persistent=False)
    return model


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_configurations_predating_the_field_keep_the_learned_table():
    config = _slt_config(transformers_version="5.15.0")
    assert config.visual_position_embedding_type == "learned"


def test_a_freshly_built_configuration_also_keeps_the_learned_table():
    assert _slt_config().visual_position_embedding_type == "learned"


@pytest.mark.parametrize("position_embedding_type", ["learned", "none", "sincos"])
def test_every_mode_survives_serialization(position_embedding_type):
    config = _slt_config(visual_position_embedding_type=position_embedding_type)
    restored = SltConfig.from_dict(config.to_dict())
    assert restored.visual_position_embedding_type == position_embedding_type


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="visual_position_embedding_type"):
        _slt_config(visual_position_embedding_type="sinusoidal")


def test_a_non_string_mode_is_rejected():
    with pytest.raises(TypeError, match="visual_position_embedding_type"):
        _slt_config(visual_position_embedding_type=3)


def test_ctc_hidden_size_drives_pre_ctc_modules(registered_stub_components):
    config = _slt_config(
        ctc_hidden_size=8,
        visual_position_embedding_type="learned",
    )
    llm = Qwen3ForCausalLM(_llm_config())
    mark_module_tree_as_initialized(llm)

    model = SltModel(config, llm=llm)

    assert model.ctc_head.in_features == 8
    assert model.visual_position_embedding.embedding_dim == 8
    assert model.ctc_codebook.qwen_hidden_size == HIDDEN_SIZE


# --------------------------------------------------------------------------
# What each mode puts in the model and in the checkpoint
# --------------------------------------------------------------------------


def test_the_learned_mode_owns_a_trainable_table(registered_stub_components):
    model = _model("learned", registered_stub_components)

    assert isinstance(model.visual_position_embedding, nn.Embedding)
    assert model.visual_position_table is None
    assert "visual_position_embedding.weight" in model.state_dict()


@pytest.mark.parametrize("position_embedding_type", ["none", "sincos"])
def test_the_parameterless_modes_leave_the_checkpoint_clean(
    position_embedding_type, registered_stub_components
):
    model = _model(position_embedding_type, registered_stub_components)

    assert model.visual_position_embedding is None
    assert not [
        key for key in model.state_dict() if key.startswith("visual_position")
    ]


def test_the_sincos_table_is_materialized_but_not_serialized(
    registered_stub_components,
):
    model = _model("sincos", registered_stub_components)

    assert model.visual_position_table.shape == (
        SltModel.MAX_TOKEN_LENGTH,
        HIDDEN_SIZE,
    )
    # A serialized copy would let a stale table silently override the formula.
    assert "visual_position_table" not in model.state_dict()
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("visual_position")
    )


# --------------------------------------------------------------------------
# The sinusoidal table itself
# --------------------------------------------------------------------------


def test_the_sincos_rows_carry_the_learned_table_initialization_scale():
    table = SltModel._build_sincos_position_table(64, 128)

    norms = table.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))
    # Raw sinusoids would be sqrt(D/2) = 8 here, i.e. ~9x the learned table's
    # 0.02 * sqrt(D) = 0.226 initialization.
    assert 0.02 * math.sqrt(128) == pytest.approx(0.226, abs=1e-3)


def test_the_sincos_table_separates_positions():
    table = SltModel._build_sincos_position_table(64, 128)

    similarities = table[:-1] @ table[1:].T
    assert torch.diagonal(similarities).max() < 0.9999
    # Deterministic: no run-to-run or checkpoint-to-checkpoint drift.
    torch.testing.assert_close(
        table, SltModel._build_sincos_position_table(64, 128)
    )


def test_the_sincos_table_handles_an_odd_hidden_size():
    table = SltModel._build_sincos_position_table(8, 5)

    assert table.shape == (8, 5)
    assert torch.isfinite(table).all()


# --------------------------------------------------------------------------
# The forward path
# --------------------------------------------------------------------------


def test_none_returns_the_features_untouched():
    features = torch.randn(6, HIDDEN_SIZE)
    model = _shell("none")

    output = model.visual_position_embedding_forward(
        features, torch.tensor([6]), torch.arange(6)
    )

    torch.testing.assert_close(output, features)


def test_none_accepts_positions_beyond_the_table_capacity():
    """Without a table there is no MAX_TOKEN_LENGTH to overflow."""
    features = torch.randn(2, HIDDEN_SIZE)
    model = _shell("none")

    positions = torch.tensor([SltModel.MAX_TOKEN_LENGTH, SltModel.MAX_TOKEN_LENGTH + 1])
    output = model.visual_position_embedding_forward(
        features, torch.tensor([2]), positions
    )

    torch.testing.assert_close(output, features)


def test_learned_adds_the_row_of_each_position():
    embedding = nn.Embedding(SltModel.MAX_TOKEN_LENGTH, HIDDEN_SIZE)
    features = torch.randn(3, HIDDEN_SIZE)
    model = _shell("learned", embedding=embedding)

    positions = torch.tensor([0, 1, 7])
    output = model.visual_position_embedding_forward(
        features, torch.tensor([3]), positions
    )

    torch.testing.assert_close(output, features + embedding.weight[positions])


def test_sincos_adds_the_row_of_each_position():
    table = SltModel._build_sincos_position_table(SltModel.MAX_TOKEN_LENGTH, HIDDEN_SIZE)
    features = torch.randn(3, HIDDEN_SIZE)
    model = _shell("sincos", table=table)

    positions = torch.tensor([0, 1, 7])
    output = model.visual_position_embedding_forward(
        features, torch.tensor([3]), positions
    )

    torch.testing.assert_close(output, features + table[positions])


def test_sincos_matches_the_feature_dtype():
    table = SltModel._build_sincos_position_table(SltModel.MAX_TOKEN_LENGTH, HIDDEN_SIZE)
    features = torch.randn(3, HIDDEN_SIZE, dtype=torch.bfloat16)
    model = _shell("sincos", table=table)

    output = model.visual_position_embedding_forward(
        features, torch.tensor([3]), torch.tensor([0, 1, 2])
    )

    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("position_embedding_type", ["learned", "sincos"])
def test_a_table_lookup_still_rejects_out_of_range_positions(position_embedding_type):
    model = _shell(
        position_embedding_type,
        table=SltModel._build_sincos_position_table(
            SltModel.MAX_TOKEN_LENGTH, HIDDEN_SIZE
        )
        if position_embedding_type == "sincos"
        else None,
        embedding=nn.Embedding(SltModel.MAX_TOKEN_LENGTH, HIDDEN_SIZE)
        if position_embedding_type == "learned"
        else None,
    )

    with pytest.raises(ValueError, match="visual position id"):
        model.visual_position_embedding_forward(
            torch.randn(1, HIDDEN_SIZE),
            torch.tensor([1]),
            torch.tensor([SltModel.MAX_TOKEN_LENGTH]),
        )
