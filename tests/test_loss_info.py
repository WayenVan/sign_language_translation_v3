import torch
from torch.nn import functional as F

from csi_slt.modeling_slt.output_utils import SltCausalLMOutputWithPast
from csi_slt.modeling_slt.slt import SltModel


def test_loss_info_is_a_detached_logging_dictionary():
    source = torch.tensor(2.0, requires_grad=True)
    total_loss = source.square()
    loss_info = {
        "main_loss": total_loss.detach(),
    }

    output = SltCausalLMOutputWithPast(
        loss=total_loss,
        logits=torch.empty(0),
        loss_info=loss_info,
    )

    assert output.loss.requires_grad is True
    assert set(output.loss_info) == {"main_loss"}
    for value in output.loss_info.values():
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
