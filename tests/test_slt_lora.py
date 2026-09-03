"""Unit tests plus an optional real-checkpoint SLT LoRA round trip.

Set ``BASE_CHECKPOINT`` below to a local, non-LoRA ``SltModel`` checkpoint,
then run this test directly. For example:

    python tests/test_slt_lora.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, TaskType
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS, VISUAL_BACKBONES
from csi_slt.modeling_slt.slt import (
    SltConfig,
    SltModel,
    validate_llm_lora_config_presence,
    validate_requested_llm_lora_config,
)


BASE_CHECKPOINT = Path(
    "outputs/qwen3-1.7b-dinoframev2-shuffle-cross-0731//checkpoint-68000"
)


def _tiny_native_llm() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            tie_word_embeddings=True,
        )
    )


def _slt_shell(llm: Qwen3ForCausalLM) -> SltModel:
    model = object.__new__(SltModel)
    torch.nn.Module.__init__(model)
    model.llm = llm
    model.config = SimpleNamespace(llm_lora=False, llm_lora_config={})
    return model


class _TinyBackbone(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)


class _TinyAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)


def test_llm_lora_is_injected_without_wrapping_native_llm():
    native_llm = _tiny_native_llm()
    model = _slt_shell(native_llm)

    model.inject_llm_lora(
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
        )
    )

    assert model.llm is native_llm
    assert type(model.llm) is Qwen3ForCausalLM
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert any("lora_" in name for name, _ in model.llm.named_parameters())
    assert not any(name.startswith("llm.base_model.") for name in model.state_dict())


def test_native_llm_tied_weights_survive_full_checkpoint_round_trip(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(VISUAL_BACKBONES, "tiny_test_backbone", _TinyBackbone)
    monkeypatch.setitem(VISUAL_ADAPTERS, "tiny_test_adapter", _TinyAdapter)
    llm = _tiny_native_llm()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
    )
    serialized_lora = lora_config.to_dict()
    serialized_lora["target_modules"] = sorted(serialized_lora["target_modules"])
    config = SltConfig(
        hidden_size=16,
        video_soft_token_id=1,
        llm_model_name_or_path="Qwen/tiny-test-model",
        llm_config=llm.config,
        llm_lora=True,
        llm_lora_config=serialized_lora,
        visual_backbone_type="tiny_test_backbone",
        visual_backbone_config={},
        visual_adapter_type="tiny_test_adapter",
        visual_adapter_kwargs={},
        video_bidirectional_attention=False,
        visual_position_embedding_type="none",
    )
    model = SltModel(config)

    assert type(model.llm) is Qwen3ForCausalLM
    assert model.all_tied_weights_keys == {
        "llm.lm_head.weight": "llm.model.embed_tokens.weight"
    }
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight

    checkpoint = tmp_path / "native-llm-lora"
    model.save_pretrained(checkpoint)
    reloaded, loading_info = SltModel.from_pretrained(
        checkpoint,
        output_loading_info=True,
    )

    assert type(reloaded.llm) is Qwen3ForCausalLM
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert (
        reloaded.get_input_embeddings().weight
        is reloaded.get_output_embeddings().weight
    )
    assert not any(
        name.startswith("llm.base_model.") for name in reloaded.state_dict()
    )


@pytest.mark.parametrize(
    ("enabled", "config", "message"),
    [
        (True, {}, "must be provided"),
        (False, {"r": 4}, "must be empty"),
    ],
)
def test_llm_lora_presence_requires_flag_and_config_to_agree(
    enabled, config, message
):
    with pytest.raises(ValueError, match=message):
        validate_llm_lora_config_presence(enabled=enabled, config=config)


def test_requested_lora_config_comparison_normalizes_defaults_and_sets():
    requested = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
    )
    checkpoint = {
        "task_type": "CAUSAL_LM",
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["v_proj", "q_proj"],
    }

    validate_requested_llm_lora_config(checkpoint, requested)


def test_requested_lora_config_comparison_reports_structural_mismatch():
    with pytest.raises(ValueError, match=r"r: checkpoint=4, requested=8"):
        validate_requested_llm_lora_config(
            {"r": 4, "target_modules": ["q_proj"]},
            LoraConfig(r=8, target_modules=["q_proj"]),
        )


def _base_checkpoint() -> Path:
    checkpoint_path = BASE_CHECKPOINT.expanduser()
    if not checkpoint_path.is_dir():
        pytest.skip(
            "Set BASE_CHECKPOINT in tests/test_slt_lora.py to an existing "
            f"checkpoint directory; current value: {checkpoint_path}"
        )
    return checkpoint_path


def _first_lora_target(llm: torch.nn.Module) -> str:
    """Select one internal LLM linear layer for a small, model-agnostic test."""
    for name, module in llm.named_modules():
        if isinstance(module, torch.nn.Linear) and name and name != "lm_head":
            return name
    pytest.fail("The checkpoint LLM does not contain a LoRA-compatible Linear layer")


def _lora_state(model: SltModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def test_create_save_and_reload_lora_checkpoint(tmp_path: Path):
    base_checkpoint = _base_checkpoint()

    base_model = SltModel.from_pretrained(base_checkpoint)
    assert not base_model.config.llm_lora, (
        "BASE_CHECKPOINT must point to a checkpoint without LoRA"
    )
    input_weight = base_model.get_input_embeddings().weight
    output_weight = base_model.get_output_embeddings().weight
    assert input_weight.data_ptr() == output_weight.data_ptr(), (
        "Input embeddings and lm_head are not tied after loading"
    )
    assert torch.equal(input_weight, output_weight)

    target_module = _first_lora_target(base_model.llm)
    del base_model

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=[target_module],
    )
    model = SltModel.from_pretrained(str(base_checkpoint))
    model.inject_llm_lora(peft_config)

    assert model.config.llm_lora is True
    assert model.config.llm_lora_config

    lora_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    assert lora_parameters, "No LoRA parameters were created"
    assert all(parameter.requires_grad for parameter in lora_parameters.values())

    non_lora_llm_parameters = [
        parameter
        for name, parameter in model.llm.named_parameters()
        if "lora_" not in name
    ]
    assert non_lora_llm_parameters
    assert all(not parameter.requires_grad for parameter in non_lora_llm_parameters)

    # LoRA B is normally initialized to zero. Put deterministic non-zero values
    # into every adapter parameter so the reload check cannot pass accidentally.
    with torch.no_grad():
        for index, parameter in enumerate(lora_parameters.values(), start=1):
            parameter.fill_(index / 100.0)
    expected_lora_state = _lora_state(model)

    saved_checkpoint = tmp_path / "slt-lora-checkpoint"
    model.save_pretrained(saved_checkpoint)
    del model

    reloaded, loading_info = SltModel.from_pretrained(
        saved_checkpoint,
        output_loading_info=True,
    )

    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info.get("mismatched_keys")
    assert reloaded.config.llm_lora is True

    input_weight = reloaded.get_input_embeddings().weight
    output_weight = reloaded.get_output_embeddings().weight
    assert input_weight.data_ptr() == output_weight.data_ptr(), (
        "Input embeddings and lm_head are not tied after loading"
    )
    assert torch.equal(input_weight, output_weight)

    actual_lora_state = _lora_state(reloaded)
    assert actual_lora_state.keys() == expected_lora_state.keys()
    for name, expected in expected_lora_state.items():
        torch.testing.assert_close(actual_lora_state[name], expected, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", *sys.argv[1:]]))
