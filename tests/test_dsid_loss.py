from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from csi_slt.modeling_slt.dsid import compute_dsid_loss
from csi_slt.modeling_slt.slt import SltModel


def _path_logits(
    labels: torch.Tensor,
    target_logits: list[list[float]],
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    logits = torch.zeros((*labels.shape, len(target_logits[0])))
    valid_prediction_mask = labels[:, 1:].ne(-100)
    logits[:, :-1][valid_prediction_mask] = torch.tensor(target_logits)
    return logits.requires_grad_(requires_grad)


def test_dsid_aligns_different_source_lengths_and_stops_teacher_gradients():
    student_labels = torch.tensor([[-100, -100, 0, 1]])
    gloss_labels = torch.tensor([[-100, -100, -100, 0, 1]])
    empty_labels = torch.tensor([[-100, 0, 1]])

    student_logits = _path_logits(
        student_labels,
        [[0.2, -0.1], [-0.3, 0.4]],
        requires_grad=True,
    )
    gloss_logits = _path_logits(
        gloss_labels,
        [[3.0, 0.0], [3.0, 0.0]],
        requires_grad=True,
    )
    empty_logits = _path_logits(
        empty_labels,
        [[0.0, 0.0], [0.0, 0.0]],
        requires_grad=True,
    )

    result = compute_dsid_loss(
        student_logits,
        student_labels,
        gloss_logits,
        gloss_labels,
        empty_logits,
        empty_labels,
        js_tau=1.0,
    )

    gloss_log_probs = F.log_softmax(torch.tensor([3.0, 0.0]), dim=-1)
    empty_log_probs = F.log_softmax(torch.tensor([0.0, 0.0]), dim=-1)
    log_mixture = torch.logaddexp(gloss_log_probs, empty_log_probs) - torch.log(
        torch.tensor(2.0)
    )
    js = 0.5 * (
        (gloss_log_probs.exp() * (gloss_log_probs - log_mixture)).sum()
        + (empty_log_probs.exp() * (empty_log_probs - log_mixture)).sum()
    )
    first_student_log_probs = F.log_softmax(torch.tensor([0.2, -0.1]), dim=-1)
    first_kl = (
        gloss_log_probs.exp() * (gloss_log_probs - first_student_log_probs)
    ).sum()

    # Only target 0 passes the direction gate. Division remains by both valid
    # target positions, not by the number or sum of active weights.
    torch.testing.assert_close(result.loss, js * first_kl / 2)
    torch.testing.assert_close(result.gate_coverage, torch.tensor(0.5))
    torch.testing.assert_close(result.mean_weight, js / 2)

    result.loss.backward()
    assert student_logits.grad is not None
    assert gloss_logits.grad is None
    assert empty_logits.grad is None


def test_dsid_rejects_target_sequences_that_do_not_correspond():
    student_labels = torch.tensor([[-100, 0, 1]])
    gloss_labels = torch.tensor([[-100, -100, 1, 0]])
    empty_labels = torch.tensor([[-100, 0, 1]])

    with pytest.raises(ValueError, match="target token IDs do not align"):
        compute_dsid_loss(
            _path_logits(student_labels, [[0.0, 0.0], [0.0, 0.0]]),
            student_labels,
            _path_logits(gloss_labels, [[0.0, 0.0], [0.0, 0.0]]),
            gloss_labels,
            _path_logits(empty_labels, [[0.0, 0.0], [0.0, 0.0]]),
            empty_labels,
            js_tau=0.1,
        )


class _TeacherLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters_disabled = False
        self.observed_states = []

    @contextmanager
    def disable_adapter(self):
        self.adapters_disabled = True
        try:
            yield
        finally:
            self.adapters_disabled = False

    def forward(self, input_ids, **kwargs):
        self.observed_states.append(
            (self.training, self.adapters_disabled, torch.is_grad_enabled())
        )
        logits = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
        return SimpleNamespace(logits=logits)


def test_text_teacher_context_is_eval_no_grad_lora_off_and_restores_mode():
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(llm_lora=True, dsid_js_tau=0.1)
    model.llm = _TeacherLLM()
    model.llm.train()

    student_labels = torch.tensor([[-100, 1, 2]])
    gloss_labels = torch.tensor([[-100, -100, 1, 2]])
    empty_labels = torch.tensor([[-100, 1, 2]])
    model._compute_dsid_training_loss(
        _path_logits(
            student_labels,
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            requires_grad=True,
        ),
        student_labels,
        pseudo_gloss_teacher_input_ids=torch.zeros_like(gloss_labels),
        pseudo_gloss_teacher_attention_mask=torch.ones_like(gloss_labels),
        pseudo_gloss_teacher_position_ids=torch.arange(4).unsqueeze(0),
        pseudo_gloss_teacher_labels=gloss_labels,
        empty_source_teacher_input_ids=torch.zeros_like(empty_labels),
        empty_source_teacher_attention_mask=torch.ones_like(empty_labels),
        empty_source_teacher_position_ids=torch.arange(3).unsqueeze(0),
        empty_source_teacher_labels=empty_labels,
    )

    assert model.llm.observed_states == [(False, True, False), (False, True, False)]
    assert model.llm.training is True
    assert model.llm.adapters_disabled is False


class _ForwardLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(output_attentions=False)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            logits = inputs_embeds
        else:
            logits = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
            if bool(input_ids.eq(9).any()):
                logits[..., 1] = 2.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def test_slt_forward_adds_dsid_to_ce_and_reports_detached_diagnostics():
    model = object.__new__(SltModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        llm_lora=False,
        dsid_loss_weight=0.5,
        dsid_js_tau=0.1,
    )
    model.llm = _ForwardLLM()
    model.has_sliding_layers = False
    model._current_dsid_loss_weight = 0.5
    model.train()

    labels = torch.tensor([[-100, -100, 1]])
    student_logits = torch.zeros((1, 3, 3), requires_grad=True)
    output = model(
        input_ids=torch.tensor([[4, 5, 1]]),
        inputs_embeds=student_logits,
        attention_mask={},
        labels=labels,
        pseudo_gloss_teacher_input_ids=torch.tensor([[4, 9, 5, 1]]),
        pseudo_gloss_teacher_attention_mask=torch.ones((1, 4), dtype=torch.long),
        pseudo_gloss_teacher_position_ids=torch.arange(4).unsqueeze(0),
        pseudo_gloss_teacher_labels=torch.tensor([[-100, -100, -100, 1]]),
        empty_source_teacher_input_ids=torch.tensor([[4, 5, 1]]),
        empty_source_teacher_attention_mask=torch.ones((1, 3), dtype=torch.long),
        empty_source_teacher_position_ids=torch.arange(3).unsqueeze(0),
        empty_source_teacher_labels=labels,
    )

    torch.testing.assert_close(
        output.loss,
        output.loss_info["ce_loss"] + 0.5 * output.loss_info["dsid_loss"],
    )
    assert output.loss_info["dsid_loss"].item() > 0.0
    assert {
        "main_loss",
        "ce_loss",
        "dsid_loss",
        "dsid_weighted_loss",
        "dsid_loss_weight",
        "dsid_mean_js",
        "dsid_gate_coverage",
        "dsid_mean_weight",
        "dsid_mean_kl",
        "dsid_teacher_nll_gain",
    } == set(output.loss_info)
    assert all(not value.requires_grad for value in output.loss_info.values())

    output.loss.backward()
    assert student_logits.grad is not None
