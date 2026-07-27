from dataclasses import dataclass
from typing import NamedTuple
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from typing import Optional


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
    extras: Optional[dict] = None  # any extra information


class PrepareForCausalLMOutput(NamedTuple):
    input_ids: torch.Tensor  # [B, L]
    inputs_embeds: torch.Tensor  # [B, L, D]
    visual_mask: torch.Tensor  # [B, L]
    visual_features: torch.Tensor  # [Bn, L, D]
    visual_length: torch.Tensor  # [B]


@dataclass
class SltCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal-LM outputs augmented with the individual training loss terms."""

    main_loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
