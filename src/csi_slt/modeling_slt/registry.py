from torch import nn
from typing import Dict, Type


from .visual_backbones.dinov2_backbone import DinoV2Backbone
from .visual_adapters.token_sampler_adapter import TokenSampleAdapter
from .visual_adapters.patch_shuffle_adapter import TemporalShuffleAdapter
from .visual_backbones.pretrained_backbone import PretrainedBackbone

VISUAL_BACKBONES: Dict[str, nn.Module] = {
    "dinov2": DinoV2Backbone,
    "pretrained": PretrainedBackbone,
}
VISUAL_ADAPTERS: Dict[str, nn.Module] = {
    "token_sampler": TokenSampleAdapter,
    "temporal_shuffle": TemporalShuffleAdapter,
}
