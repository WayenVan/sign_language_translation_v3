from torch import nn
from typing import Dict


from .visual_backbones.dinov2_backbone import DinoV2Backbone
from .visual_backbones.dinov3_backbone import DinoV3Backbone
from .visual_adapters.token_sampler_adapter import TokenSampleAdapter
from .visual_adapters.patch_shuffle import TemporalShuffleAdapter
from .visual_backbones.pretrained_backbone import PretrainedBackbone
from .visual_adapters.temporal_merge_adapter import TemporalMergeAdapter
from .visual_adapters.token_sampler_adapter_v2 import (
    TokenSampleAdapterV2,
)
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter import DINOFrameAdapter
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross import (
    DINOFrameAdapterCross,
)
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2 import (
    DINOFrameAdapterCrossV2,
)
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2_global import (
    DINOFrameAdapterCrossV2Global,
)
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2_shuffle import (
    DINOFrameAdapterCrossV2Shuffle,
)
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v2_grouped_shuffle import (
    DINOFrameAdapterCrossV2GroupedShuffle,
)
from csi_slt.modeling_slt.visual_adapters.dinoframe_adapter_cross_v3 import (
    DINOFrameAdapterCrossV3,
)
from csi_slt.modeling_slt.visual_backbones.c_radio_v4_backbone import CRadioV4Backbone
from csi_slt.modeling_slt.visual_backbones.siglip2_backbone import Siglip2Backbone

VISUAL_BACKBONES: Dict[str, nn.Module] = {
    "dinov2": DinoV2Backbone,
    "dinov3": DinoV3Backbone,
    "pretrained": PretrainedBackbone,
    "c_radio_v4": CRadioV4Backbone,
    "siglip2": Siglip2Backbone,
}
VISUAL_ADAPTERS: Dict[str, nn.Module] = {
    "token_sampler": TokenSampleAdapter,
    "temporal_shuffle": TemporalShuffleAdapter,
    "temporal_merge": TemporalMergeAdapter,
    "token_sampler_v2": TokenSampleAdapterV2,
    "dinoframe": DINOFrameAdapter,
    "dinoframe_cross": DINOFrameAdapterCross,
    "dinoframe_cross_v2": DINOFrameAdapterCrossV2,  # WARN: Two-token V2 adapter.
    "dinoframe_cross_v2_global": DINOFrameAdapterCrossV2Global,
    "dinoframe_cross_v2_shuffle": DINOFrameAdapterCrossV2Shuffle,
    "dinoframe_cross_v2_grouped_shuffle": DINOFrameAdapterCrossV2GroupedShuffle,
    "dinoframe_cross_v3": DINOFrameAdapterCrossV3,
    "c_radio_v4": CRadioV4Backbone,
}
