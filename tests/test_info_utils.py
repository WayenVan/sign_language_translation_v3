import pytest
import torch

from csi_slt.modeling_slt.info_utils import (
    InformationOutput,
    InformationRequest,
    build_information_output,
)
from csi_slt.modeling_slt.output_utils import PrepareForCausalLMOutput


def test_information_request_defaults_to_disabled():
    request = InformationRequest()

    assert request.enabled is False
    assert request.sample_indices == (0,)
    assert request.llm_layers == (-1,)


def test_information_request_reports_enabled_selection():
    request = InformationRequest(
        llm_attentions=True,
        sample_indices=(0, 2),
        llm_layers=(0, -1),
    )

    assert request.enabled is True


def test_visual_backbone_extras_enable_information_request():
    assert InformationRequest(visual_backbone_extras=True).enabled is True


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"sample_indices": [0]}, TypeError),
        ({"sample_indices": (-1,)}, ValueError),
        ({"sample_indices": (0, 0)}, ValueError),
        ({"llm_layers": ()}, ValueError),
        ({"llm_layers": (True,)}, TypeError),
        ({"llm_attentions": 1}, TypeError),
        ({"visual_backbone_extras": 1}, TypeError),
    ],
)
def test_information_request_validates_selection(kwargs, exception):
    with pytest.raises(exception):
        InformationRequest(**kwargs)


def test_build_information_output_selects_samples_and_attention_layers():
    request = InformationRequest(
        llm_attentions=True,
        sample_indices=(2,),
        llm_layers=(-1,),
        reduce_heads=True,
    )
    prepare_output = PrepareForCausalLMOutput(
        input_ids=torch.zeros(3, 5, dtype=torch.long),
        inputs_embeds=torch.zeros(3, 5, 4),
        visual_mask=torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0]]),
        visual_lengths=torch.tensor([2, 1, 3]),
        packed_visual_position_ids=torch.tensor([0, 1, 0, 0, 1, 2]),
    )
    llm_attentions = (
        torch.zeros(3, 2, 5, 5),
        torch.arange(150, dtype=torch.float).reshape(3, 2, 5, 5),
    )

    information = build_information_output(
        request=request,
        batch_size=3,
        llm_attentions=llm_attentions,
        prepare_output=prepare_output,
    )

    assert isinstance(information, InformationOutput)
    assert information.visual_lengths.tolist() == [3]
    assert information.visual_position_ids.tolist() == [[0, 1, 2]]
    assert len(information.llm_attentions) == 1
    assert information.llm_attentions[0].shape == (1, 5, 5)


def test_build_information_output_detaches_visual_backbone_extras():
    source = torch.tensor([1.0], requires_grad=True)
    information = build_information_output(
        request=InformationRequest(visual_backbone_extras=True),
        batch_size=1,
        llm_attentions=None,
        prepare_output=None,
        visual_backbone_extras={"nested": {"weights": source * 2}},
    )

    weights = information.visual_backbone_extras["nested"]["weights"]
    assert weights.requires_grad is False
    assert weights.device == source.device


def test_information_output_detach_to_cpu_preserves_structure_and_dtype():
    source = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    information = InformationOutput(
        llm_attentions=(source * 6,),
        llm_visual_mask=torch.ones(1, dtype=torch.long),
        visual_lengths=torch.ones(1, dtype=torch.long),
    )

    cpu_information = information.detach_to_cpu()

    assert cpu_information is not information
    for tensor in (cpu_information.llm_attentions[0],):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None

    # Conversion returns a new tree and leaves the original autograd values intact.
    assert information.llm_attentions[0].requires_grad is True
