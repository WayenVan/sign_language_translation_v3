import pytest
import torch
from torch import nn

from csi_slt.modeling_slt.ctc_codebook import CTCCodebookBridge
from csi_slt.modeling_slt.slt import SltModel


def _bridge(*, training_mode="soft"):
    torch.manual_seed(0)
    bridge = CTCCodebookBridge(
        ctc_vocab_size=4,
        qwen_hidden_size=8,
        blank_id=0,
        training_mode=training_mode,
    )
    bridge.initialize_from_qwen_embeddings(
        nn.Embedding(6, 8),
        [[], [1], [2, 4], [5]],
        qwen_pad_token_id=3,
    )
    return bridge


def test_soft_path_preserves_packed_layout_and_gradients():
    bridge = _bridge()
    logits = torch.randn(5, 4, requires_grad=True)

    output = bridge(logits, torch.tensor([2, 3]), temperature=1.0)

    assert output.embeddings.shape == (5, 8)
    assert output.lengths.tolist() == [2, 3]
    torch.testing.assert_close(output.token_distribution.sum(-1), torch.ones(5))
    output.embeddings.square().sum().backward()
    assert logits.grad is not None
    assert bridge.codebook.weight.grad is not None


def test_straight_through_path_is_hard_forward_and_differentiable():
    bridge = _bridge(training_mode="straight_through")
    logits = torch.randn(6, 4, requires_grad=True)

    output = bridge(logits, torch.tensor([6]), temperature=0.5)

    assert torch.all((output.token_distribution == 0) | (output.token_distribution == 1))
    torch.testing.assert_close(output.token_distribution.sum(-1), torch.ones(6))
    output.embeddings.sum().backward()
    assert logits.grad is not None


def test_eval_defaults_to_deterministic_argmax_and_keeps_blank_embedding():
    bridge = _bridge(training_mode="straight_through").eval()
    logits = torch.tensor(
        [[10.0, 0.0, 0.0, 0.0], [0.0, 1.0, 5.0, 2.0]]
    )

    first = bridge(logits, torch.tensor([2]))
    second = bridge(logits, torch.tensor([2]))

    assert first.predicted_ids.tolist() == [0, 2]
    torch.testing.assert_close(first.embeddings, second.embeddings)
    assert first.embeddings[0].abs().sum() > 0
    assert first.embeddings[1].abs().sum() > 0


def test_eval_forces_argmax_when_training_mode_is_soft():
    bridge = _bridge().eval()
    logits = torch.zeros(2, 4)

    output = bridge(logits, torch.tensor([2]))

    torch.testing.assert_close(output.blank_probability, torch.ones(2))


def test_initialization_averages_qwen_subtokens_and_uses_pad_for_blank():
    bridge = _bridge()
    qwen_embeddings = nn.Embedding(6, 8)
    with torch.no_grad():
        qwen_embeddings.weight.copy_(torch.arange(48).reshape(6, 8))

    bridge.initialize_from_qwen_embeddings(
        qwen_embeddings,
        [[], [1], [2, 4], [5]],
        qwen_pad_token_id=3,
    )

    torch.testing.assert_close(
        bridge.codebook.weight[0], qwen_embeddings.weight[3]
    )
    torch.testing.assert_close(bridge.codebook.weight[1], qwen_embeddings.weight[1])
    torch.testing.assert_close(
        bridge.codebook.weight[2],
        qwen_embeddings.weight[[2, 4]].mean(dim=0),
    )


def test_blank_codebook_row_is_trainable():
    bridge = _bridge().eval()
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

    output = bridge(logits, torch.tensor([1]))
    weights = torch.arange(1, 9, dtype=output.embeddings.dtype)
    (output.embeddings[0] * weights).sum().backward()

    assert bridge.codebook.weight.grad[bridge.blank_id].abs().sum() > 0


def test_blank_logging_scalars_are_detached_and_include_pad_drift():
    bridge = _bridge().eval()
    qwen_embeddings = nn.Embedding(6, 8)
    bridge.initialize_from_qwen_embeddings(
        qwen_embeddings,
        [[], [1], [2, 4], [5]],
        qwen_pad_token_id=3,
    )
    logits = torch.tensor(
        [[5.0, 0.0, 0.0, 0.0], [0.0, 4.0, 1.0, 0.0]],
        requires_grad=True,
    )

    output = bridge(logits, torch.tensor([2]))

    assert set(output.logging_scalars) == {
        "blank_probability_mean",
        "blank_argmax_ratio",
        "blank_embedding_norm",
        "blank_pad_cosine_similarity",
    }
    assert all(value.numel() == 1 for value in output.logging_scalars.values())
    assert all(
        not value.requires_grad for value in output.logging_scalars.values()
    )
    # blank_probability_mean is a plain temperature-1 softmax over the raw
    # logits -- independent of eval's forced argmax selection mode, unlike
    # the embedding path itself. Row 0 argmax is blank, row 1 argmax is not,
    # so blank_argmax_ratio is still exactly 0.5.
    expected_blank_probability = torch.softmax(logits.detach(), dim=-1)[:, 0].mean()
    torch.testing.assert_close(
        output.logging_scalars["blank_probability_mean"], expected_blank_probability
    )
    torch.testing.assert_close(
        output.logging_scalars["blank_argmax_ratio"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        output.logging_scalars["blank_pad_cosine_similarity"], torch.tensor(1.0)
    )


def test_blank_frequency_scalars_are_independent_of_selection_mode_and_temperature():
    # Phase-A (which never builds a codebook distribution) and joint
    # training (whose distribution depends on training_mode/temperature/
    # Gumbel noise) must report the same blank-frequency numbers for the
    # same logits, or the two phases' curves would not be comparable.
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 4.0, 1.0, 0.0]])

    baseline = CTCCodebookBridge.blank_frequency_scalars(logits, blank_id=0)
    for bridge in (
        _bridge(training_mode="soft"),
        _bridge(training_mode="straight_through"),
    ):
        for mode_kwargs in ({}, {"temperature": 0.3}):
            output = bridge(logits, torch.tensor([2]), **mode_kwargs)
            torch.testing.assert_close(
                (
                    output.logging_scalars["blank_probability_mean"],
                    output.logging_scalars["blank_argmax_ratio"],
                ),
                baseline,
            )


def test_rejects_temperature_below_straight_through_floor():
    bridge = _bridge(training_mode="straight_through")

    with pytest.raises(ValueError, match="temperature"):
        bridge(torch.randn(2, 4), torch.tensor([2]), temperature=0.01)


def test_rejects_lengths_that_do_not_match_packed_logits():
    bridge = _bridge()

    with pytest.raises(ValueError, match="sum to packed tokens"):
        bridge(torch.randn(3, 4), torch.tensor([2]))


def test_uninitialized_codebook_fails_before_first_forward():
    bridge = CTCCodebookBridge(
        ctc_vocab_size=4,
        qwen_hidden_size=8,
        blank_id=0,
    )

    with pytest.raises(RuntimeError, match="has not been initialized"):
        bridge(torch.randn(2, 4), torch.tensor([2]))


def test_initialized_state_survives_checkpoint_loading():
    source = _bridge()
    restored = CTCCodebookBridge(
        ctc_vocab_size=4,
        qwen_hidden_size=8,
        blank_id=0,
    )

    restored.load_state_dict(source.state_dict())
    restored.assert_initialized()

    assert restored._initialization_verified is True


def test_slt_model_exposes_codebook_initialization_without_tokenizers():
    class FakeLlm(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(6, 8)

        def get_input_embeddings(self):
            return self.embed_tokens

    model = SltModel.__new__(SltModel)
    nn.Module.__init__(model)
    model.llm = FakeLlm()
    model.ctc_codebook = CTCCodebookBridge(
        ctc_vocab_size=4,
        qwen_hidden_size=8,
        blank_id=0,
    )

    model.initialize_ctc_codebook(
        [[], [1], [2, 4], [5]],
        blank_init_token_id=3,
    )

    model.ctc_codebook.assert_initialized()
    torch.testing.assert_close(
        model.ctc_codebook.codebook.weight[0],
        model.llm.embed_tokens.weight[3],
    )
