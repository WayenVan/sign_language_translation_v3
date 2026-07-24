import logging

from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from torch import nn
from transformers.models.dinov2_with_registers.configuration_dinov2_with_registers import (
    Dinov2WithRegistersConfig,
)
from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput

logger = logging.getLogger(__name__)


class DinoV2Backbone(nn.Module):
    def __init__(
        self,
        config: dict,
        dinov2: Dinov2WithRegistersModel | None = None,
    ):
        super().__init__()
        self.id = config.get("id")

        if self.id is None:
            raise ValueError("id must be provided in config for DinoV2Backbone")

        self.output_layer = config.get("output_layer", -1)
        self.config = config

        if dinov2 is None:
            self.dinov2_config = Dinov2WithRegistersConfig.from_pretrained(self.id)
            self.visual_encoder = Dinov2WithRegistersModel._from_config(
                self.dinov2_config
            )
        else:
            self.visual_encoder = dinov2
            self.dinov2_config = dinov2.config

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
                :, 1 + self.dinov2_config.num_register_tokens :, :
            ],  # [B, T-1, C]
            pooled_visual_features=feats[:, 0, :],  # [B, C]
            visual_length=t_lengths,
        )

    @classmethod
    def from_pretrained_backbone(cls, config: dict, dtype="auto"):
        id = config.get("id")

        if id is None:
            raise ValueError("id must be provided in config for DinoV2Backbone")

        dinov2 = Dinov2WithRegistersModel.from_pretrained(id, dtype=dtype)
        logger.info(f"Loaded pretrained DinoV2 model from {id}")

        return cls(config=config, dinov2=dinov2)


if __name__ == "__main__":
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    # model = DinoV2Backbone.from_pretrained_backbone(
    #     "facebook/dinov2-with-registers-base"
    # )
    model = DinoV2Backbone.from_pretrained_backbone(
        config={"id": "facebook/dinov2-with-registers-base", "output_layer": -1}
    ).cuda()

    x = torch.randn(2, 3, 224, 224).cuda()
    out = model(x)
    print(out.visual_features.shape)
