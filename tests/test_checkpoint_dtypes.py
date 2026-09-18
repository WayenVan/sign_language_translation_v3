import json

import pytest
import torch
from safetensors.torch import save_file

from csi_slt.utils.checkpoint_dtypes import (
    read_stored_dtypes,
    restore_checkpoint_dtypes,
)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.register_buffer("scale", torch.randn(4))
        self.register_buffer("mask", torch.ones(4, dtype=torch.bool))
        self.register_buffer("index", torch.arange(4))


class _Model(torch.nn.Module):
    """Stands in for SltModel: a low-precision LLM beside an fp32 visual tower."""

    def __init__(self) -> None:
        super().__init__()
        self.llm = _Block()
        self.visual_backbone = _Block()


def _write_checkpoint(directory, *, llm_dtype=torch.bfloat16, sharded=False):
    torch.manual_seed(0)
    reference = _Model()
    stored = {
        key: value.to(llm_dtype)
        if key.startswith("llm.") and value.is_floating_point()
        else value
        for key, value in reference.state_dict().items()
    }
    if sharded:
        llm = {key: value for key, value in stored.items() if key.startswith("llm.")}
        rest = {key: value for key, value in stored.items() if key not in llm}
        save_file(llm, str(directory / "model-00001-of-00002.safetensors"))
        save_file(rest, str(directory / "model-00002-of-00002.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                **{key: "model-00001-of-00002.safetensors" for key in llm},
                **{key: "model-00002-of-00002.safetensors" for key in rest},
            },
        }
        (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(stored, str(directory / "model.safetensors"))
    return stored


def _load_as_float32(stored):
    """Mimic ``from_pretrained`` at torch's default dtype."""
    model = _Model()
    model.load_state_dict(
        {
            key: value.to(torch.float32) if value.is_floating_point() else value
            for key, value in stored.items()
        }
    )
    return model


def test_reads_only_floating_point_dtypes(tmp_path):
    _write_checkpoint(tmp_path)

    stored_dtypes = read_stored_dtypes(tmp_path)

    assert stored_dtypes == {
        "llm.weight": torch.bfloat16,
        "llm.scale": torch.bfloat16,
        "visual_backbone.weight": torch.float32,
        "visual_backbone.scale": torch.float32,
    }


def test_restores_stored_dtypes_bit_for_bit(tmp_path):
    stored = _write_checkpoint(tmp_path)
    model = _load_as_float32(stored)

    summary = restore_checkpoint_dtypes(model, tmp_path)

    assert summary == {"cast": 2, "unchanged": 2, "missing": 0, "non_float": 4}
    for key, value in model.state_dict().items():
        expected = stored[key]
        assert value.dtype == expected.dtype, key
        assert torch.equal(
            value.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        ), key


def test_keeps_parameter_identity_and_grad_flag(tmp_path):
    stored = _write_checkpoint(tmp_path)
    model = _load_as_float32(stored)
    parameter = model.llm.weight

    restore_checkpoint_dtypes(model, tmp_path)

    assert model.llm.weight is parameter
    assert isinstance(model.llm.weight, torch.nn.Parameter)
    assert model.llm.weight.requires_grad


def test_reads_sharded_checkpoints(tmp_path):
    stored = _write_checkpoint(tmp_path, sharded=True)
    model = _load_as_float32(stored)

    summary = restore_checkpoint_dtypes(model, tmp_path)

    assert summary["cast"] == 2
    assert model.llm.weight.dtype == torch.bfloat16
    assert model.visual_backbone.weight.dtype == torch.float32


def test_all_float32_checkpoint_casts_nothing(tmp_path):
    stored = _write_checkpoint(tmp_path, llm_dtype=torch.float32)
    model = _load_as_float32(stored)

    summary = restore_checkpoint_dtypes(model, tmp_path)

    assert summary["cast"] == 0
    assert summary["unchanged"] == 4


def test_missing_weights_are_reported_not_cast(tmp_path):
    stored = _write_checkpoint(tmp_path)
    model = _load_as_float32(stored)
    model.extra = torch.nn.Parameter(torch.zeros(2))

    summary = restore_checkpoint_dtypes(model, tmp_path)

    assert summary["missing"] == 1
    assert model.extra.dtype == torch.float32


def test_rejects_a_checkpoint_without_safetensors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no safetensors weights"):
        read_stored_dtypes(tmp_path)
