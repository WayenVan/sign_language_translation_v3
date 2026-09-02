from dataclasses import dataclass
from typing import NamedTuple
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from typing import Optional

from .info_utils import InformationOutput


@dataclass
class VisualBackboneOutput(ModelOutput):
    visual_features: Optional[torch.Tensor] = (
        None  # [visual_length_1+visual_length_2..., feature_dim] visual features, might contains spatial dimensions
    )
    pooled_visual_features: Optional[torch.Tensor] = (
        None  # [visual_length_1+visual_length_2..., feature_dim] pooled visual features
    )
    visual_length: Optional[torch.Tensor] = (
        None  # [batch_size] length of visual feautres for each sample in the batch
    )
    extras: Optional[dict] = None  # any extra information


@dataclass
class VisualAdapterOutput(ModelOutput):
    # NOTE: this should only contains visual tokens
    visual_features: torch.Tensor  # [visual_length_1+visual_length_2..., feature_dim] adapted visual features, might contains spatial dimensions
    visual_length: Optional[torch.Tensor] = (
        None  # [batch_size] length of visual feautres for each sample in the batch
    )
    position_ids: Optional[torch.Tensor] = (  # Adapter-defined visual positions.
        None  # Optional packed visual-token positions; defaults to 0..length-1 per sample.
    )
    # Detached single-element tensors the trainer averages across steps, merged
    # into the model's own logging_scalars under a "visual_adapter/" prefix.
    # Kept apart from ``extras`` -- which carries arbitrary tensors -- because
    # being scalar is the one property here that has to be enforced. Values stay
    # on device: calling .item() per step would synchronize the accelerator, so
    # the conversion belongs where logging actually happens.
    logging_scalars: Optional[dict[str, torch.Tensor]] = None
    extras: Optional[dict] = None  # any extra information


class PrepareForCausalLMOutput(NamedTuple):
    input_ids: torch.Tensor  # [B, L]
    inputs_embeds: torch.Tensor  # [B, L, D]
    visual_mask: torch.Tensor  # [B, L]
    visual_lengths: Optional[torch.Tensor] = None  # [B]
    packed_visual_position_ids: Optional[torch.Tensor] = None  # [sum(Lv)]
    # Pre-scale visual features/lengths, i.e. before `* self.visual_scale`,
    # kept for the CTC head so its input isn't affected by that LLM-embedding
    # scale factor.
    ctc_visual_features: Optional[torch.Tensor] = None  # [sum(Lv), D]
    ctc_visual_lengths: Optional[torch.Tensor] = None  # [B]
    # Carried through so SltModel.forward can merge them into its own
    # logging_scalars; the adapter output itself does not reach that far.
    visual_adapter_logging_scalars: Optional[dict[str, torch.Tensor]] = None


@dataclass
class SltCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal-LM outputs augmented with detached logging information."""

    logging_scalars: Optional[dict[str, torch.Tensor]] = None
    information: Optional[InformationOutput] = None
