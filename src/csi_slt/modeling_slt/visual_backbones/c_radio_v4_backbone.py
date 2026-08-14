import logging
import math
from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel

from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized
from csi_slt.modeling_slt.output_utils import VisualBackboneOutput

logger = logging.getLogger(__name__)


class _LayerAttentionFusion(nn.Module):
    """Fuse same-shaped layer outputs with content-dependent layer attention.

    Attention is computed independently for every leading token position from
    channel-wise mean, RMS, and absolute mean descriptors. The final scoring
    layer starts at zero, making the initial fusion exactly a uniform mean.
    """

    def __init__(self, num_layers: int, hidden_dim: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("layer attention requires at least two layers")
        if hidden_dim <= 0:
            raise ValueError("layer_fusion_hidden_dim must be positive")

        self.num_layers = num_layers
        self.score_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.layer_bias = nn.Parameter(torch.zeros(num_layers))
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def forward(
        self, tensors: list[torch.Tensor], name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(tensors) != self.num_layers:
            raise RuntimeError(
                f"expected {self.num_layers} C-RADIO {name} layers, "
                f"got {len(tensors)}"
            )
        reference_shape = tensors[0].shape
        if any(tensor.shape != reference_shape for tensor in tensors[1:]):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise RuntimeError(f"cannot fuse C-RADIO {name} tensors with shapes {shapes}")

        # [F, L, ..., C]. Scores are produced per frame and per optional token.
        stacked = torch.stack(tensors, dim=1)
        mean = stacked.mean(dim=-1)
        rms = stacked.square().mean(dim=-1).add(1e-6).sqrt()
        absolute_mean = stacked.abs().mean(dim=-1)
        descriptors = torch.stack((mean, rms, absolute_mean), dim=-1)
        descriptors = (descriptors - descriptors.mean(dim=1, keepdim=True)) / (
            descriptors.var(dim=1, keepdim=True, unbiased=False).add(1e-6).sqrt()
        )

        bias_shape = (1, self.num_layers, *([1] * (stacked.ndim - 3)))
        logits = self.score_mlp(descriptors).squeeze(-1)
        logits = logits + self.layer_bias.view(bias_shape)
        weights = logits.softmax(dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class CRadioV4Backbone(nn.Module):
    def __init__(
        self,
        config: dict,
        c_radio_v4: PreTrainedModel | None = None,
    ):
        super().__init__()
        self.id = config.get("id")

        if self.id is None:
            raise ValueError("id must be provided in config for CRadioV4Backbone")

        self.output_layers = self._normalize_output_layers(
            config.get("output_layer", [-1, -2, -3, -4])
        )
        # Preserve the public attribute used by existing configuration/debug code.
        self.output_layer = self.output_layers
        self.config = config
        self._input_values_validated = False
        freeze_visual_encoder = config.get("freeze_visual_encoder", True)
        if not isinstance(freeze_visual_encoder, bool):
            raise TypeError("freeze_visual_encoder must be a boolean")
        self.freeze_visual_encoder = freeze_visual_encoder

        if c_radio_v4 is None:
            self.c_radio_v4_config = AutoConfig.from_pretrained(
                self.id, trust_remote_code=True
            )
            self.visual_encoder = AutoModel.from_config(self.c_radio_v4_config)
        else:
            self.visual_encoder = c_radio_v4
            self.c_radio_v4_config = c_radio_v4.config

        if self.freeze_visual_encoder:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()

        if len(self.output_layers) > 1:
            fusion_hidden_dim = config.get("layer_fusion_hidden_dim", 32)
            if isinstance(fusion_hidden_dim, bool) or not isinstance(
                fusion_hidden_dim, int
            ):
                raise TypeError("layer_fusion_hidden_dim must be an integer")
            self.summary_layer_fusion = _LayerAttentionFusion(
                len(self.output_layers), fusion_hidden_dim
            )
            self.feature_layer_fusion = _LayerAttentionFusion(
                len(self.output_layers), fusion_hidden_dim
            )
        else:
            self.summary_layer_fusion = None
            self.feature_layer_fusion = None

        # Preserve the encoder's initialization/checkpoint and the fusion
        # modules' uniform-mean initialization when attached to SltModel.
        mark_module_tree_as_initialized(self)

    def train(self, mode: bool = True):
        """Train fusion modules while keeping a frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze_visual_encoder:
            self.visual_encoder.eval()
        return self

    def forward(self, x, t_lengths=None) -> VisualBackboneOutput:
        """
        video: Packed frames with shape [F, C, H, W].
        """
        self._validate_inputs(x, t_lengths)
        radio_outputs = self._forward_intermediates(x)
        summaries = [output.summary for output in radio_outputs]
        feature_layers = [output.features for output in radio_outputs]
        extras = None
        if len(radio_outputs) == 1:
            summary = summaries[0]
            features = feature_layers[0]
        else:
            summary, summary_layer_weights = self.summary_layer_fusion(
                summaries, "summary"
            )
            features, feature_layer_weights = self.feature_layer_fusion(
                feature_layers, "features"
            )
            extras = {
                "summary_layer_weights": summary_layer_weights,
                "feature_layer_weights": feature_layer_weights,
            }

        return VisualBackboneOutput(
            visual_features=features,  # [B, T, C]
            pooled_visual_features=summary,  # [B, C]
            visual_length=t_lengths,
            extras=extras,
        )

    @staticmethod
    def _normalize_output_layers(output_layer: int | Sequence[int]) -> tuple[int, ...]:
        """Normalize one or more intermediate block indices."""
        if isinstance(output_layer, bool):
            raise TypeError("output_layer must be an integer or a sequence of integers")
        if isinstance(output_layer, int):
            return (output_layer,)
        if isinstance(output_layer, (str, bytes)) or not isinstance(
            output_layer, Sequence
        ):
            raise TypeError("output_layer must be an integer or a sequence of integers")

        layers = tuple(output_layer)
        if not layers:
            raise ValueError("output_layer must contain at least one layer index")
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
            raise TypeError("every output_layer entry must be an integer")
        if len(set(layers)) != len(layers):
            raise ValueError("output_layer must not contain duplicate layer indices")
        return layers

    def _forward_intermediates(self, x: torch.Tensor):
        radio_model = getattr(self.visual_encoder, "radio_model", self.visual_encoder)
        forward_intermediates = getattr(radio_model, "forward_intermediates", None)
        if forward_intermediates is None:
            raise TypeError(
                "C-RADIO encoder must expose forward_intermediates to fuse layers"
            )
        outputs = forward_intermediates(
            x,
            indices=list(self.output_layers),
            return_prefix_tokens=True,
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
            aggregation="sparse",
        )
        if len(outputs) != len(self.output_layers):
            raise RuntimeError(
                "C-RADIO returned an unexpected number of intermediate layers: "
                f"expected {len(self.output_layers)}, got {len(outputs)}"
            )
        return outputs

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
        config={"id": "nvidia/C-RADIOv4-SO400M"}
    ).cuda()

    x = torch.rand(2, 3, 224, 224).cuda()
    out = model(x)
    print(out.visual_features.shape)
