"""KV caching must not mutate state during activation recomputation."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from transformers import Qwen3Config, Qwen3ForCausalLM

from csi_slt.modeling_slt.slt import SltModel


def _model(checkpointed):
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    config = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, use_cache=True,
    )
    config._attn_implementation = 'sdpa'
    model.llm = Qwen3ForCausalLM(config)
    model.llm.requires_grad_(False)
    model.llm.eval()  # Frozen LLM stays in eval even during adapter training.
    if checkpointed:
        for i, layer in enumerate(model.llm.model.layers):
            model.llm.model.layers[i] = checkpoint_wrapper(layer)
    model.config = SimpleNamespace(
        video_bidirectional_attention=False, label_smoothing=0.0, ctc_enabled=False,
    )
    model.has_sliding_layers = False
    return model


@pytest.mark.parametrize('use_cache', [None, True, False])
def test_frozen_llm_checkpoint_backward_matches_uncached_reference(use_cache):
    torch.manual_seed(42)
    model = _model(checkpointed=True)
    torch.manual_seed(42)
    reference = _model(checkpointed=False)
    embeds = torch.randn(2, 6, 16, requires_grad=True)
    reference_embeds = embeds.detach().clone().requires_grad_()
    labels = torch.randint(0, 32, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    output = model(input_ids=None, inputs_embeds=embeds, attention_mask=mask, labels=labels, use_cache=use_cache)
    output.loss.backward()
    expected = reference(input_ids=None, inputs_embeds=reference_embeds, attention_mask=mask, labels=labels, use_cache=False)
    expected.loss.backward()
    assert output.past_key_values is None
    assert torch.isfinite(embeds.grad).all()
    torch.testing.assert_close(output.loss, expected.loss)
    torch.testing.assert_close(embeds.grad, reference_embeds.grad)


def test_eval_cached_decoding_still_extends_cache():
    model = _model(checkpointed=False).eval()
    with torch.no_grad():
        output = model(input_ids=None, inputs_embeds=torch.randn(1, 4, 16), use_cache=True)
        assert output.past_key_values.get_seq_length() == 4
        output = model(input_ids=torch.tensor([[3]]), past_key_values=output.past_key_values, use_cache=True)
        assert output.past_key_values.get_seq_length() == 5


def test_training_rejects_existing_cache_without_mutating_it():
    model = _model(checkpointed=False)
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.llm.config)
    with pytest.raises(ValueError, match='past_key_values'):
        model(input_ids=None, inputs_embeds=torch.randn(1, 4, 16), past_key_values=cache)
    assert cache.get_seq_length() == 0
