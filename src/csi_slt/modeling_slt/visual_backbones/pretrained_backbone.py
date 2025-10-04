from csi_sign_pretrain.modeling_sign_visual.sign_pt_model import (
    SignVisualModelForPretrain,
)

import torch.nn as nn
from ..output_utils import VisualBackboneOutput
import torch.nn.functional as F


class PretrainedBackbone(nn.Module):
    def __init__(self, ckpt_path, norm_output=True):
        super().__init__()
        self.backbone = SignVisualModelForPretrain.from_pretrained(ckpt_path).backbone
        self.norm_output = norm_output

    def forward(self, x, t_lengths=None):
        feats = self.backbone(x)
        if self.norm_output:
            feats = F.normalize(feats, p=2, dim=-1)
        return VisualBackboneOutput(
            pooled_visual_features=feats,
            visual_length=t_lengths,
            visual_features=None,
        )
