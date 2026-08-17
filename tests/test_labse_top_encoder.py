import pytest
import torch
from transformers import BertConfig, BertModel

from csi_slt.modeling_slt.labse_top_encoder import (
    LaBSESemanticEncoder,
    LaBSETopEncoder,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput


def _tiny_labse() -> BertModel:
    return BertModel(
        BertConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            max_position_embeddings=12,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )


def _config(**overrides) -> dict:
    return {
        "id": "sentence-transformers/LaBSE-test",
        "num_layers": 2,
        "freeze": True,
        **overrides,
    }


def _semantic_config(**projector_overrides) -> dict:
    return {
        "encoder": _config(),
        "projector": {
            "hidden_dim": 12,
            "output_dim": 16,
            **projector_overrides,
        },
        "position_embedding": {
            "max_positions": 12,
            "init_std": 0.02,
        },
        "residual_gate_init": -4.0,
    }


def test_selects_original_top_layers_and_freezes_them():
    labse = _tiny_labse()
    expected_layers = list(labse.encoder.layer[-2:])

    encoder = LaBSETopEncoder(config=_config(), labse=labse)

    assert list(encoder.layers) == expected_layers
    assert encoder.source_layer_indices == (2, 3)
    assert encoder.hidden_size == 8
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert not encoder.training


def test_forward_matches_manual_top_layer_loop():
    labse = _tiny_labse().eval()
    encoder = LaBSETopEncoder(
        config=_config(freeze=False),
        labse=labse,
    ).eval()
    hidden_states = torch.randn(2, 5, 8)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    expected = hidden_states
    extended_mask = encoder._extend_attention_mask(attention_mask, hidden_states)
    for layer in encoder.layers:
        layer_output = layer(expected, attention_mask=extended_mask)
        expected = layer_output[0] if isinstance(layer_output, tuple) else layer_output

    actual = encoder(hidden_states, attention_mask)

    torch.testing.assert_close(actual, expected)


def test_masked_features_do_not_affect_valid_outputs():
    encoder = LaBSETopEncoder(
        config=_config(),
        labse=_tiny_labse(),
    )
    attention_mask = torch.tensor([[1, 1, 0]])
    first = torch.randn(1, 3, 8)
    second = first.clone()
    second[:, 2] = torch.randn(8) * 100

    first_output = encoder(first, attention_mask)
    second_output = encoder(second, attention_mask)

    torch.testing.assert_close(first_output[:, :2], second_output[:, :2])


def test_frozen_encoder_keeps_gradient_flow_to_visual_inputs():
    encoder = LaBSETopEncoder(config=_config(), labse=_tiny_labse())
    hidden_states = torch.randn(1, 4, 8, requires_grad=True)

    encoder(hidden_states).sum().backward()

    assert hidden_states.grad is not None
    assert all(parameter.grad is None for parameter in encoder.parameters())


@pytest.mark.parametrize("num_layers", [0, 5, True])
def test_invalid_layer_counts_are_rejected(num_layers):
    with pytest.raises((TypeError, ValueError)):
        LaBSETopEncoder(config=_config(num_layers=num_layers), labse=_tiny_labse())


def test_input_shape_mask_and_position_limit_are_validated():
    encoder = LaBSETopEncoder(config=_config(), labse=_tiny_labse())

    with pytest.raises(ValueError, match="width"):
        encoder(torch.randn(1, 4, 7))
    with pytest.raises(ValueError, match="attention_mask"):
        encoder(torch.randn(1, 4, 8), torch.ones(1, 3))
    with pytest.raises(ValueError, match="values must be 0 or 1"):
        encoder(torch.randn(1, 4, 8), torch.tensor([[1, 1, 1, 2]]))


def test_from_pretrained_encoder_loads_configured_model(monkeypatch):
    labse = _tiny_labse()
    observed = {}

    def fake_from_pretrained(model_id, *, dtype):
        observed.update(model_id=model_id, dtype=dtype)
        return labse

    monkeypatch.setattr(
        "csi_slt.modeling_slt.labse_top_encoder.AutoModel.from_pretrained",
        fake_from_pretrained,
    )

    encoder = LaBSETopEncoder.from_pretrained_encoder(
        _config(num_layers=3), dtype=torch.float16
    )

    assert observed == {
        "model_id": "sentence-transformers/LaBSE-test",
        "dtype": torch.float16,
    }
    assert encoder.source_layer_indices == (1, 2, 3)


def test_init_requires_external_encoder_and_configured_id():
    with pytest.raises(TypeError, match="externally constructed"):
        LaBSETopEncoder(config=_config(), labse=None)
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        LaBSETopEncoder(config={"num_layers": 2}, labse=_tiny_labse())


def test_config_validation_normalizes_defaults_without_loading_encoder():
    config = LaBSETopEncoder._validate_config({"id": "custom/labse"})

    assert config == {
        "id": "custom/labse",
        "num_layers": 4,
        "freeze": True,
    }


def test_semantic_encoder_processes_packed_features_and_preserves_metadata():
    top_encoder = LaBSETopEncoder(config=_config(), labse=_tiny_labse())
    semantic_encoder = LaBSESemanticEncoder(
        config=_semantic_config(), encoder=top_encoder
    )
    lengths = torch.tensor([2, 4])
    position_ids = torch.tensor([0, 1, 0, 1, 2, 3])
    extras = {"source": torch.tensor(1.0)}
    visual_output = VisualAdapterOutput(
        visual_features=torch.randn(6, 8, requires_grad=True),
        visual_length=lengths,
        position_ids=position_ids,
        extras=extras,
    )

    output = semantic_encoder(visual_output)

    assert output.visual_features.shape == (6, 16)
    assert output.visual_length is lengths
    assert output.position_ids is position_ids
    assert output.extras is extras
    output.visual_features.sum().backward()
    assert visual_output.visual_features.grad is not None
    assert all(parameter.grad is None for parameter in top_encoder.parameters())
    assert all(
        parameter.grad is not None for parameter in semantic_encoder.projector.parameters()
    )
    assert semantic_encoder.position_embeddings.weight.grad is not None
    assert semantic_encoder.input_layernorm.weight.grad is not None
    assert semantic_encoder.residual_gate.grad is not None


def test_semantic_encoder_from_pretrained_nests_top_encoder_loading(monkeypatch):
    labse = _tiny_labse()
    observed = {}

    def fake_from_pretrained(model_id, *, dtype):
        observed.update(model_id=model_id, dtype=dtype)
        return labse

    monkeypatch.setattr(
        "csi_slt.modeling_slt.labse_top_encoder.AutoModel.from_pretrained",
        fake_from_pretrained,
    )

    semantic_encoder = LaBSESemanticEncoder.from_pretrained_encoder(
        _semantic_config(), dtype=torch.bfloat16
    )

    assert observed == {
        "model_id": "sentence-transformers/LaBSE-test",
        "dtype": torch.bfloat16,
    }
    assert isinstance(semantic_encoder.encoder, LaBSETopEncoder)
    assert semantic_encoder.input_dim == 8
    assert semantic_encoder.output_dim == 16


def test_semantic_encoder_config_validation_normalizes_projector_hidden_dim():
    config = LaBSESemanticEncoder._validate_config(
        {
            "encoder": {"id": "custom/labse"},
            "projector": {"output_dim": 32},
        }
    )

    assert config["encoder"]["num_layers"] == 4
    assert config["projector"] == {"output_dim": 32, "hidden_dim": 32}
    assert config["position_embedding"] == {
        "max_positions": 1024,
        "init_std": 0.02,
    }
    assert config["residual_gate_init"] == -4.0


def test_semantic_encoder_rejects_invalid_packed_lengths():
    semantic_encoder = LaBSESemanticEncoder(
        config=_semantic_config(),
        encoder=LaBSETopEncoder(config=_config(), labse=_tiny_labse()),
    )
    visual_output = VisualAdapterOutput(
        visual_features=torch.randn(4, 8),
        visual_length=torch.tensor([1, 2]),
    )

    with pytest.raises(ValueError, match="packed token count"):
        semantic_encoder(visual_output)


def test_semantic_encoder_uses_fresh_sequential_positions_and_gated_residual():
    top_encoder = LaBSETopEncoder(config=_config(), labse=_tiny_labse())
    semantic_encoder = LaBSESemanticEncoder(
        config=_semantic_config(), encoder=top_encoder
    ).eval()
    features = torch.randn(5, 8)
    lengths = torch.tensor([2, 3])
    first_output = VisualAdapterOutput(
        visual_features=features,
        visual_length=lengths,
        position_ids=torch.tensor([9, 9, 8, 8, 8]),
    )
    second_output = VisualAdapterOutput(
        visual_features=features,
        visual_length=lengths,
        position_ids=torch.tensor([0, 1, 2, 3, 4]),
    )

    first = semantic_encoder(first_output)
    second = semantic_encoder(second_output)

    torch.testing.assert_close(first.visual_features, second.visual_features)
    torch.testing.assert_close(
        torch.sigmoid(semantic_encoder.residual_gate),
        torch.sigmoid(torch.tensor(-4.0)),
    )


def test_semantic_position_embedding_capacity_is_validated():
    config = _semantic_config()
    config["position_embedding"]["max_positions"] = 3
    semantic_encoder = LaBSESemanticEncoder(
        config=config,
        encoder=LaBSETopEncoder(config=_config(), labse=_tiny_labse()),
    )

    with pytest.raises(ValueError, match="position embedding capacity"):
        semantic_encoder(
            VisualAdapterOutput(
                visual_features=torch.randn(4, 8),
                visual_length=torch.tensor([4]),
            )
        )


def test_semantic_encoder_can_rebuild_skeleton_from_embedded_model_config():
    original = LaBSESemanticEncoder(
        config=_semantic_config(),
        encoder=LaBSETopEncoder(config=_config(), labse=_tiny_labse()),
    )

    rebuilt = LaBSESemanticEncoder.from_encoder_config(original.config)

    assert rebuilt.encoder.source_layer_indices == (2, 3)
    assert rebuilt.input_dim == 8
    assert rebuilt.hidden_dim == 12
    assert rebuilt.output_dim == 16
