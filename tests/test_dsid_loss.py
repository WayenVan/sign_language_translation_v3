import pytest
import torch
from torch.nn import functional as F

from csi_slt.modeling_slt.dsid import compute_dsid_loss


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
