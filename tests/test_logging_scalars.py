import torch
from torch.nn import functional as F

from csi_slt.modeling_slt.output_utils import SltCausalLMOutputWithPast
from csi_slt.modeling_slt.slt import SltModel, _ctc_head_blank_frequency_scalars


def test_logging_scalars_are_detached():
    source = torch.tensor(2.0, requires_grad=True)
    total_loss = source.square()
    logging_scalars = {
        "main_loss": total_loss.detach(),
    }

    output = SltCausalLMOutputWithPast(
        loss=total_loss,
        logits=torch.empty(0),
        logging_scalars=logging_scalars,
    )

    assert output.loss.requires_grad is True
    assert set(output.logging_scalars) == {"main_loss"}
    for value in output.logging_scalars.values():
        assert value.requires_grad is False
        assert value.grad_fn is None


def test_causal_lm_loss_is_shifted_ce_and_ignores_masked_targets():
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, -2.0],
                [-1.0, 0.0, 2.0],
                [5.0, -1.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 1, -100, 0]])

    loss = SltModel._compute_causal_lm_loss(logits, labels)
    expected = F.cross_entropy(
        torch.stack((logits[0, 0], logits[0, 2])),
        torch.tensor([1, 0]),
    )

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_ctc_head_blank_frequency_scalars_are_pure_and_detached():
    # A pure function of raw ctc_head logits -- no codebook state involved --
    # so Phase-A (ctc_only, no codebook distribution at all) and joint
    # training can both call it directly and land on the same numbers.
    logits = torch.tensor(
        [[5.0, 0.0, 0.0, 0.0], [0.0, 4.0, 1.0, 0.0]], requires_grad=True
    )

    scalars = _ctc_head_blank_frequency_scalars(logits, blank_id=0)

    assert set(scalars) == {"blank_probability_mean", "blank_argmax_ratio"}
    assert all(value.numel() == 1 and not value.requires_grad for value in scalars.values())
    # Row 0's argmax is blank, row 1's is not, so the ratio is exactly 0.5.
    expected_blank_probability = torch.softmax(logits.detach(), dim=-1)[:, 0].mean()
    torch.testing.assert_close(
        scalars["blank_probability_mean"], expected_blank_probability
    )
    torch.testing.assert_close(scalars["blank_argmax_ratio"], torch.tensor(0.5))


def test_ctc_head_blank_frequency_scalars_handles_empty_batch():
    logits = torch.empty(0, 4)

    scalars = _ctc_head_blank_frequency_scalars(logits, blank_id=0)

    assert scalars["blank_probability_mean"].item() == 0.0
    assert scalars["blank_argmax_ratio"].item() == 0.0
