import logging

from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig
from transformers.models.dinov2_with_registers.configuration_dinov2_with_registers import (
    Dinov2WithRegistersConfig,
)
from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)
from transformers.models.dinov3_vit import DINOv3ViTModel

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput

logger = logging.getLogger(__name__)


class CRadioV4Backbone(nn.Module):
    def __init__(
        self,
        config: dict,
        c_radio_v4: PreTrainedModel | None = None,
    ):
        super().__init__()
        self.id = config.get("id")

        if self.id is None:
            raise ValueError("id must be provided in config for DinoV3Backbone")

        self.output_layer = config.get("output_layer", -1)
        self.config = config

        if c_radio_v4 is None:
            self.c_radio_v4_config = AutoConfig.from_pretrained(self.id)
            self.visual_encoder = AutoModel.from_config(self.c_radio_v4_config)
        else:
            self.visual_encoder = c_radio_v4
            self.c_radio_v4_config = c_radio_v4.config

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, t_lengths=None) -> VisualBackboneOutput:
        """
        video: [B, C, H, W]
        """
        B, C, H, W = x.shape
        feats = self.visual_encoder(x, output_hidden_states=True).hidden_states[
            self.output_layer
        ]

        return VisualBackboneOutput(
            visual_features=feats[
                :, 1 + self.c_radio_v4_config.num_register_tokens :, :
            ],  # [B, T-1, C]
            pooled_visual_features=feats[:, 0, :],  # [B, C]
            visual_length=t_lengths,
        )

    @classmethod
    def from_pretrained_backbone(cls, config: dict, dtype="auto"):
        id = config.get("id")

        if id is None:
            raise ValueError("id must be provided in config for CRadioV4Backbone")

        c_radio_v4 = AutoModel.from_pretrained(id, dtype=dtype)
        logger.info(f"Loaded pretrained CRadioV4 model from {id}")

        return cls(config=config, c_radio_v4=c_radio_v4)


if __name__ == "__main__":
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    # model = DinoV3Backbone.from_pretrained_backbone(
    #     "facebook/dinov3-with-registers-base"
    # )
    model = CRadioV4Backbone.from_pretrained_backbone(
        config={"id": "facebook/dinov3-with-registers-base", "output_layer": -1}
    ).cuda()

    x = torch.randn(2, 3, 224, 224).cuda()
    out = model(x)
    print(out.visual_features.shape)
