import logging

from torch import nn
from transformers.models.dinov3_vit import (
    DINOv3ViTConfig,
    DINOv3ViTModel,
)

from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized

logger = logging.getLogger(__name__)


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        config: dict,
        dinov3: DINOv3ViTModel | None = None,
    ):
        super().__init__()
        self.id = config.get("id")

        if self.id is None:
            raise ValueError("id must be provided in config for DinoV3Backbone")

        self.output_layer = config.get("output_layer", -1)
        self.config = config

        if dinov3 is None:
            self.dinov3_config = DINOv3ViTConfig.from_pretrained(self.id)
            self.visual_encoder = DINOv3ViTModel(self.dinov3_config)
        else:
            self.visual_encoder = dinov3
            self.dinov3_config = dinov3.config

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, t_lengths=None) -> VisualBackboneOutput:
        """
        x: Packed video frames with shape [F, C, H, W].
        """
        feats = self.visual_encoder(x, output_hidden_states=True).hidden_states[
            self.output_layer
        ]
        first_patch_token = 1 + self.dinov3_config.num_register_tokens

        return VisualBackboneOutput(
            visual_features=feats[:, first_patch_token:, :],  # [F, N, C]
            pooled_visual_features=feats[:, 0, :],  # [F, C]
            visual_length=t_lengths,
        )

    @classmethod
    def from_pretrained_backbone(cls, config: dict, dtype="auto"):
        id = config.get("id")

        if id is None:
            raise ValueError("id must be provided in config for DinoV3Backbone")

        dinov3 = DINOv3ViTModel.from_pretrained(id, dtype=dtype)
        logger.info(f"Loaded pretrained DinoV3 model from {id}")

        return cls(config=config, dinov3=dinov3)


if __name__ == "__main__":
    import torch

    model = DinoV3Backbone.from_pretrained_backbone(
        config={
            "id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "output_layer": -1,
        }
    ).cuda()

    x = torch.randn(2, 3, 224, 224).cuda()
    out = model(x)
    print(out.visual_features.shape)
