from dataclasses import dataclass
from typing import NamedTuple
import torch
from transformers.utils import ModelOutput
from typing import Optional


@dataclass
class VisualBackboneOutput(ModelOutput):
    visual_features: torch.Tensor  # [visual_length_1+visual_length_2..., ...,feature_dim] raw visual features, might contains spatial dimensions
    pooled_visual_features: Optional[torch.Tensor] = (
        None  # [visual_length_1+visual_length_2..., feature_dim] pooled visual features
    )
    visual_length: Optional[torch.Tensor] = (
        None  # [batch_size] length of visual feautres for each sample in the batch
    )


@dataclass
class VisualAdapterOutput(ModelOutput):
    # NOTE: this should only contains visual tokens
    visual_features: torch.Tensor  # [visual_length_1+visual_length_2..., feature_dim] adapted visual features, might contains spatial dimensions
    visual_length: Optional[torch.Tensor] = (
        None  # [batch_size] length of visual feautres for each sample in the batch
    )


class PrepareForCausalLMOutput(NamedTuple):
    input_ids: torch.Tensor  # [B, L]
    inputs_embeds: torch.Tensor  # [B, L, D]
    visual_mask: torch.Tensor  # [B, L]
