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
        alignment=True,
        global_pooling=True,
        sample_indices=(0, 2),
        llm_layers=(0, -1),
    )

    assert request.enabled is True


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"sample_indices": [0]}, TypeError),
        ({"sample_indices": (-1,)}, ValueError),
        ({"sample_indices": (0, 0)}, ValueError),
        ({"llm_layers": ()}, ValueError),
        ({"llm_layers": (True,)}, TypeError),
        ({"alignment": 1}, TypeError),
    ],
)
def test_information_request_validates_selection(kwargs, exception):
    with pytest.raises(exception):
        InformationRequest(**kwargs)


def test_build_information_output_selects_samples_layers_and_alignment_rows():
    request = InformationRequest(
        alignment=True,
        global_pooling=True,
        llm_attentions=True,
        sample_indices=(2,),
        llm_layers=(-1,),
        reduce_heads=True,
    )
    prepare_output = PrepareForCausalLMOutput(
        input_ids=torch.zeros(3, 5, dtype=torch.long),
        inputs_embeds=torch.zeros(3, 5, 4),
        visual_mask=torch.tensor(
            [[0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0]]
        ),
        contrastive_features=torch.zeros(6, 4),
        contrastive_visual_lengths=torch.tensor([2, 1, 3]),
        packed_visual_position_ids=torch.tensor([0, 1, 0, 0, 1, 2]),
    )
    packed_attention = torch.tensor([0.4, 0.6, 1.0, 0.2, 0.3, 0.5])
    llm_attentions = (
        torch.zeros(3, 2, 5, 5),
        torch.arange(150, dtype=torch.float).reshape(3, 2, 5, 5),
    )
    alignment = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    grouped_alignment = torch.arange(16, dtype=torch.float).reshape(2, 2, 4)

    information = build_information_output(
        request=request,
        batch_size=3,
        llm_attentions=llm_attentions,
        prepare_output=prepare_output,
        alignment_info={
            "alignment": alignment,
            "grouped_alignment": grouped_alignment,
            "grouped_video_mask": torch.tensor([[1, 1], [1, 0]]),
            "epsilon": 0.1,
        },
        packed_visual_attention=packed_attention,
        valid_pseudo=torch.tensor([True, False, True]),
    )

    assert isinstance(information, InformationOutput)
    torch.testing.assert_close(
        information.global_pooling_attention,
        torch.tensor([[0.2, 0.3, 0.5]]),
    )
    assert information.visual_lengths.tolist() == [3]
    assert information.visual_position_ids.tolist() == [[0, 1, 2]]
    assert information.valid_pseudo_indices.tolist() == [2]
    torch.testing.assert_close(
        information.alignment_info["alignment"], alignment[1:2]
    )
    torch.testing.assert_close(
        information.alignment_info["grouped_alignment"],
        grouped_alignment[1:2],
    )
    assert information.alignment_info["grouped_video_mask"].tolist() == [[1, 0]]
    assert len(information.llm_attentions) == 1
    assert information.llm_attentions[0].shape == (1, 5, 5)


def test_information_output_detach_to_cpu_preserves_structure_and_dtype():
    source = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    information = InformationOutput(
        alignment_info={
            "alignment": source * 2,
            "nested": [source * 3, (source * 4,)],
            "epsilon": 0.1,
        },
        global_pooling_attention=source * 5,
        llm_attentions=(source * 6,),
    )

    cpu_information = information.detach_to_cpu()

    assert cpu_information is not information
    assert cpu_information.alignment_info["epsilon"] == 0.1
    assert isinstance(cpu_information.alignment_info["nested"], list)
    assert isinstance(cpu_information.alignment_info["nested"][1], tuple)
    for tensor in (
        cpu_information.alignment_info["alignment"],
        cpu_information.alignment_info["nested"][0],
        cpu_information.alignment_info["nested"][1][0],
        cpu_information.global_pooling_attention,
        cpu_information.llm_attentions[0],
    ):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None

    # Conversion returns a new tree and leaves the original autograd values intact.
    assert information.global_pooling_attention.requires_grad is True
