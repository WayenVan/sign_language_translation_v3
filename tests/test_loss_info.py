import torch

from csi_slt.modeling_slt.output_utils import SltCausalLMOutputWithPast


def test_loss_info_is_a_detached_logging_dictionary():
    source = torch.tensor(2.0, requires_grad=True)
    total_loss = source.square()
    loss_info = {
        "main_loss": total_loss.detach(),
        "alignment/transport_loss": (source * 3).detach(),
        "alignment/target_mass_kl": (source * 4).detach(),
        "alignment/tv_loss": (source * 5).detach(),
    }

    output = SltCausalLMOutputWithPast(
        loss=total_loss,
        logits=torch.empty(0),
        loss_info=loss_info,
    )

    assert output.loss.requires_grad is True
    assert set(output.loss_info) == {
        "main_loss",
        "alignment/transport_loss",
        "alignment/target_mass_kl",
        "alignment/tv_loss",
    }
    for value in output.loss_info.values():
        assert value.requires_grad is False
        assert value.grad_fn is None
