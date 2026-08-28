"""Integration test for creating, saving, and reloading SLT LoRA weights.

Set ``BASE_CHECKPOINT`` below to a local, non-LoRA ``SltModel`` checkpoint,
then run this test directly. For example:

    python tests/test_slt_lora.py
"""

import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType

from csi_slt.modeling_slt.slt import SltModel


BASE_CHECKPOINT = Path(
    "outputs/qwen3-1.7b-dinoframev2-shuffle-cross-0731//checkpoint-68000"
)


def _base_checkpoint() -> Path:
    checkpoint_path = BASE_CHECKPOINT.expanduser()
    if not checkpoint_path.is_dir():
        pytest.fail(
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
