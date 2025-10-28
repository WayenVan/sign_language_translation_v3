from csi_sign_pretrain.modeling_sign_visual.sign_pt_model import (
    SignVisualModelForPretrain,
)
from csi_sign_pretrain.configuration_sign_visual.configuration import SignPretrainConfig

import torch.nn as nn
from ..output_utils import VisualBackboneOutput
import torch.nn.functional as F
from transformers import logging

logger = logging.get_logger(__name__)


class PretrainedBackbone(nn.Module):
    def __init__(self, norm_output=True, ckpt_path=None, **cfg_kwargs):
        super().__init__()
        self.config: SignPretrainConfig = SignPretrainConfig(**cfg_kwargs)
        self.backbone = SignVisualModelForPretrain(self.config).backbone

        self.norm_output: bool = norm_output

        if ckpt_path is not None:
            self._load_from_ckpt(ckpt_path)

        # if self.config.backbone_type == "convnext":
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

    def _load_from_ckpt(self, ckpt_path):
        try:
            _pretrained_model = SignVisualModelForPretrain.from_pretrained(ckpt_path)

            # if self.config.to_dict() != _pretrained_model.config.to_dict():
            #     raise ValueError(
            #         "The config of the pretrained model does not match the provided config."
            #     )
            # NOTE: verify_config
            src_config = _pretrained_model.config
            tgt_config = self.config
            for key in ["hidden_size", "backbone_type"]:
                if getattr(src_config, key) != getattr(tgt_config, key):
                    raise ValueError(
                        f"The config of the pretrained model does not match the provided config. Mismatch found in key: {key} (src: {getattr(src_config, key)}, tgt: {getattr(tgt_config, key)})"
                    )

            self.backbone.load_state_dict(
                _pretrained_model.backbone.state_dict(), strict=True
            )
        except Exception as e:
            logger.error(
                f"❌ Loading pretrained backbone of SignVisualPretrainedModel from {ckpt_path} failed with ERROR::\n\n {e} \n\n. Skipping..."
            )

    def forward(self, x, t_lengths=None):
        feats = self.backbone(x)
        if self.norm_output:
            feats = F.normalize(feats, p=2, dim=-1)
        return VisualBackboneOutput(
            pooled_visual_features=feats,
            visual_length=t_lengths,
            visual_features=None,
        )
