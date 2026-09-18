import pytest
import torch
from torch import nn

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    validate_rope_buffers,
)


class _Rotary(nn.Module):
    """Stands in for a transformers RotaryEmbedding: buffers only, no parameters."""

    def __init__(self, inv_freq: torch.Tensor | None = None) -> None:
        super().__init__()
        if inv_freq is None:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("running_scale", torch.ones(4))  # persistent
        self.rotary = _Rotary()


def test_marks_ordinary_modules():
    block = _Block()

    mark_module_tree_as_initialized(block)

    assert block._is_hf_initialized is True
    assert block.linear._is_hf_initialized is True


def test_leaves_modules_owning_non_persistent_buffers_unmarked():
    block = _Block()

    mark_module_tree_as_initialized(block)

    # Marking the rotary module would make transformers' _initialize_weights
    # return early, leaving the torch.empty_like buffer it re-allocates during
    # from_pretrained filled with uninitialized memory.
    assert getattr(block.rotary, "_is_hf_initialized", False) is False


def test_keeps_the_marker_when_a_module_also_owns_parameters():
    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))
            self.register_buffer("cached", torch.zeros(2), persistent=False)

    module = _Mixed()

    mark_module_tree_as_initialized(module)

    # Dropping the marker here would expose `weight` to re-initialization; this
    # project's mixed modules rebuild their private caches on the first forward.
    assert module._is_hf_initialized is True


def test_keeps_the_marker_when_a_module_owns_persistent_buffers():
    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("kept", torch.zeros(2))
            self.register_buffer("cached", torch.zeros(2), persistent=False)

    module = _Mixed()

    mark_module_tree_as_initialized(module)

    assert module._is_hf_initialized is True


def test_validate_rope_buffers_accepts_real_frequencies():
    assert validate_rope_buffers(_Block()) == 1


def test_validate_rope_buffers_rejects_uninitialized_memory():
    block = _Block()
    block.rotary.inv_freq = torch.full_like(block.rotary.inv_freq, 1.1e17)

    with pytest.raises(RuntimeError, match="outside"):
        validate_rope_buffers(block)


def test_validate_rope_buffers_rejects_non_finite_values():
    block = _Block()
    block.rotary.inv_freq = torch.full_like(block.rotary.inv_freq, float("nan"))

    with pytest.raises(RuntimeError, match="not recomputed|outside"):
        validate_rope_buffers(block)


def test_marks_only_peft_created_modules():
    from peft import LoraConfig, inject_adapter_in_model

    from csi_slt.modeling_slt.misc import mark_adapter_modules_as_initialized

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.rotary = _Rotary()

    model = _Tiny()
    inject_adapter_in_model(LoraConfig(r=2, target_modules=["q_proj"]), model)

    marked = mark_adapter_modules_as_initialized(model)

    assert marked > 0
    # PEFT's freshly initialized adapter weights are protected...
    assert model.q_proj._is_hf_initialized is True
    assert model.q_proj.lora_A._is_hf_initialized is True
    assert model.q_proj.lora_B["default"]._is_hf_initialized is True
    # ...while the pretrained layer it wraps and the rest of the tree are not,
    # so a checkpoint load and buffer recomputation still reach them.
    assert getattr(model.q_proj.base_layer, "_is_hf_initialized", False) is False
    assert getattr(model.rotary, "_is_hf_initialized", False) is False
    assert getattr(model, "_is_hf_initialized", False) is False
