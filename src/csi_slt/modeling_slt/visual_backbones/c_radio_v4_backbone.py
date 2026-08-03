import logging
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel

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
        self._input_values_validated = False

        if c_radio_v4 is None:
            self.c_radio_v4_config = AutoConfig.from_pretrained(
                self.id, trust_remote_code=True
            )
            self.visual_encoder = AutoModel.from_config(self.c_radio_v4_config)
        else:
            self.visual_encoder = c_radio_v4
            self.c_radio_v4_config = c_radio_v4.config

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, t_lengths=None) -> VisualBackboneOutput:
        """
        video: Packed frames with shape [F, C, H, W].
        """
        self._validate_inputs(x, t_lengths)
        radio_output = self.visual_encoder(x)

        summary = radio_output.summary
        features = radio_output.features

        return VisualBackboneOutput(
            visual_features=features,  # [B, T, C]
            pooled_visual_features=summary,  # [B, C]
            visual_length=t_lengths,
        )

    def _validate_inputs(self, x: torch.Tensor, t_lengths: torch.Tensor | None) -> None:
        """Cheaply validate packed, unnormalized C-RADIO image inputs.

        C-RADIO applies its own input conditioner and therefore expects floating
        point RGB values in the [0, 1] range rather than externally normalized
        image tensors. Scanning tensor values synchronizes the accelerator, so
        that check is performed only for the first batch handled by this module.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 4:
            raise ValueError(
                f"x must have packed-frame shape [F, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one frame")
        if x.shape[1] != 3:
            raise ValueError(f"x must contain 3 RGB channels, got {x.shape[1]}")
        if not x.is_floating_point():
            raise TypeError(f"x must use a floating-point dtype, got {x.dtype}")

        if not self._input_values_validated:
            min_value_tensor, max_value_tensor = torch.aminmax(x.detach())
            min_value = float(min_value_tensor.item())
            max_value = float(max_value_tensor.item())
            if not math.isfinite(min_value) or not math.isfinite(max_value):
                raise ValueError("x must contain only finite values")
            if min_value < 0.0 or max_value > 1.0:
                raise ValueError(
                    "C-RADIO expects unnormalized RGB values in [0, 1], but got "
                    f"range [{min_value:.6g}, {max_value:.6g}]. Disable external "
                    "normalization (for example, set do_normalize=false)."
                )
            self._input_values_validated = True

        if t_lengths is None:
            return
        if not isinstance(t_lengths, torch.Tensor):
            raise TypeError(
                "t_lengths must be a torch.Tensor or None, got "
                f"{type(t_lengths).__name__}"
            )
        if t_lengths.ndim != 1 or t_lengths.numel() == 0:
            raise ValueError("t_lengths must be a non-empty 1D tensor")
        if t_lengths.is_floating_point() or t_lengths.is_complex():
            raise TypeError(
                f"t_lengths must use an integer dtype, got {t_lengths.dtype}"
            )

    @classmethod
    def from_pretrained_backbone(cls, config: dict, dtype="auto"):
        id = config.get("id")

        if id is None:
            raise ValueError("id must be provided in config for CRadioV4Backbone")

        c_radio_v4 = AutoModel.from_pretrained(id, dtype=dtype, trust_remote_code=True)
        logger.info(f"Loaded pretrained CRadioV4 model from {id}")

        return cls(config=config, c_radio_v4=c_radio_v4)


if __name__ == "__main__":
    # model = DinoV3Backbone.from_pretrained_backbone(
    #     "facebook/dinov3-with-registers-base"
    # )
    model = CRadioV4Backbone.from_pretrained_backbone(
        config={"id": "nvidia/C-RADIOv4-SO400M", "output_layer": -1}
    ).cuda()

    x = torch.rand(2, 3, 224, 224).cuda()
    out = model(x)
    print(out.visual_features.shape)
