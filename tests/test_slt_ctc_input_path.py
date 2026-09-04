from types import MethodType, SimpleNamespace

import torch
from torch import nn

from csi_slt.modeling_slt.ctc_codebook import CTCCodebookBridge
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput
from csi_slt.modeling_slt.slt import SltModel


class _FakeLlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embed_tokens


class _CountingCtcHead(nn.Linear):
    def __init__(self):
        super().__init__(4, 3, bias=False)
        self.call_count = 0

    def forward(self, inputs):
        self.call_count += 1
        return super().forward(inputs)


def _model_shell():
    model = SltModel.__new__(SltModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=4,
        video_soft_token_id=15,
        visual_position_embedding_type="none",
        ctc_codebook_default_temperature=1.0,
        ctc_blank_id=0,
        video_token_scale=1.0,
    )
    model.llm = _FakeLlm()
    model.ctc_head = _CountingCtcHead()
    model.ctc_codebook = CTCCodebookBridge(
        ctc_vocab_size=3,
        qwen_hidden_size=4,
        blank_id=0,
        training_mode="soft",
    )
    model.initialize_ctc_codebook(
        [[], [1], [2]],
        blank_init_token_id=3,
    )
    model.start_video_embds = nn.Parameter(torch.zeros(1, 4))
    model.end_video_embeds = nn.Parameter(torch.zeros(1, 4))

    visual_output = VisualAdapterOutput(
        visual_features=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        visual_length=torch.tensor([2]),
    )

    def get_visual_feats(self, *args, **kwargs):
        return visual_output

    model.get_visual_feats = MethodType(get_visual_feats, model)
    return model


def test_prepare_routes_one_ctc_head_result_through_codebook_and_into_qwen():
    model = _model_shell()
    input_ids = torch.tensor([[4, 15, 15, 15, 15, 5]])

    output = model.prepare_for_casual_lm(
        input_ids,
        video=torch.ones(2, 3, 1, 1),
        video_length=torch.tensor([2]),
    )

    assert model.ctc_head.call_count == 1
    assert output.ctc_logits.shape == (2, 3)
    assert output.ctc_lengths.tolist() == [2]
    assert output.inputs_embeds.shape == (1, 6, 4)
    torch.testing.assert_close(
        output.inputs_embeds[0, 2:4],
        output.ctc_logits.softmax(dim=-1) @ model.ctc_codebook.codebook.weight,
    )
    assert "blank_probability_mean" in output.ctc_codebook_logging_scalars


def test_ctc_loss_reuses_precomputed_logits_without_calling_head_again():
    model = _model_shell()
    input_ids = torch.tensor([[4, 15, 15, 15, 15, 5]])
    output = model.prepare_for_casual_lm(
        input_ids,
        video=torch.ones(2, 3, 1, 1),
        video_length=torch.tensor([2]),
    )

    loss = model._compute_ctc_loss(
        output.ctc_logits,
        output.ctc_lengths,
        pseudo_gloss_ids=torch.tensor([1]),
        pseudo_gloss_length=torch.tensor([1]),
    )

    assert loss.ndim == 0
    assert model.ctc_head.call_count == 1


def test_ctc_only_forward_bypasses_codebook_and_llm():
    model = _model_shell()

    class _ForbiddenModule(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("CTC-only must not call this module")

    model.ctc_codebook = _ForbiddenModule()
    model.llm = _ForbiddenModule()

    output = model(
        pixel_values=torch.ones(2, 3, 1, 1),
        pixel_values_length=torch.tensor([2]),
        pseudo_gloss_ids=torch.tensor([1]),
        pseudo_gloss_length=torch.tensor([1]),
        forward_mode="ctc_only",
    )

    assert output.loss.ndim == 0
    assert output.logits.shape == (2, 3)
    assert output.lengths.tolist() == [2]
    # Blank-frequency diagnostics come from `CTCCodebookBridge.blank_frequency_scalars`
    # called as a plain static function on the class, not through the
    # (forbidden) `model.ctc_codebook` instance -- that's the whole point of
    # it being computable straight from ctc_head's output.
    assert set(output.logging_scalars) == {
        "main_loss",
        "ctc_loss",
        "ctc_codebook/blank_probability_mean",
        "ctc_codebook/blank_argmax_ratio",
    }


def test_ctc_only_inference_returns_logits_without_targets():
    model = _model_shell()

    output = model(
        pixel_values=torch.ones(2, 3, 1, 1),
        pixel_values_length=torch.tensor([2]),
        forward_mode="ctc_only",
    )

    assert output.loss is None
    assert output.logits.shape == (2, 3)


def test_forward_rejects_unknown_explicit_mode():
    model = _model_shell()

    try:
        model(forward_mode="unknown")
    except ValueError as error:
        assert "forward_mode" in str(error)
    else:
        raise AssertionError("unknown forward mode was accepted")
