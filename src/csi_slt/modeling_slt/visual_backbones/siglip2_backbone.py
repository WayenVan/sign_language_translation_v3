import logging
import math
from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoConfig, SiglipVisionModel
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from csi_slt.modeling_slt.misc import mark_module_tree_as_initialized
from csi_slt.modeling_slt.output_utils import VisualBackboneOutput

logger = logging.getLogger(__name__)


class _LayerAttentionFusion(nn.Module):
    """Fuse same-shaped SigLIP layer outputs with content-dependent weights."""

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
        # Begin as an exact uniform mean while retaining content-dependent
        # gradients for the first optimizer step.
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def forward(
        self, tensors: list[torch.Tensor], name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(tensors) != self.num_layers:
            raise RuntimeError(
                f"expected {self.num_layers} SigLIP 2 {name} layers, got {len(tensors)}"
            )
        reference_shape = tensors[0].shape
        if any(tensor.shape != reference_shape for tensor in tensors[1:]):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise RuntimeError(
                f"cannot fuse SigLIP 2 {name} tensors with shapes {shapes}"
            )

        # [F, L, ..., C]. Score layers independently at every leading token
        # position using compact channel statistics.
        stacked = torch.stack(tensors, dim=1)
        descriptors = torch.stack(
            (
                stacked.mean(dim=-1),
                stacked.square().mean(dim=-1).add(1e-6).sqrt(),
                stacked.abs().mean(dim=-1),
            ),
            dim=-1,
        )
        descriptors = (descriptors - descriptors.mean(dim=1, keepdim=True)) / (
            descriptors.var(dim=1, keepdim=True, unbiased=False).add(1e-6).sqrt()
        )

        bias_shape = (1, self.num_layers, *([1] * (stacked.ndim - 3)))
        logits = self.score_mlp(descriptors).squeeze(-1)
        logits = logits + self.layer_bias.view(bias_shape)
        weights = logits.softmax(dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1), weights


class Siglip2Backbone(nn.Module):
    """Expose SigLIP 2 patch and attention-pooled features to SLT adapters."""

    def __init__(
        self,
        config: dict,
        siglip2: SiglipVisionModel | None = None,
    ) -> None:
        super().__init__()
        self.id = config.get("id")
        if self.id is None:
            raise ValueError("id must be provided in config for Siglip2Backbone")

        self.output_layers = self._normalize_output_layers(
            config.get("output_layer", -1)
        )
        self.output_layer = self.output_layers
        self.config = config
        self._input_values_validated = False

        freeze_visual_encoder = config.get("freeze_visual_encoder", True)
        if not isinstance(freeze_visual_encoder, bool):
            raise TypeError("freeze_visual_encoder must be a boolean")
        self.freeze_visual_encoder = freeze_visual_encoder

        interpolate_pos_encoding = config.get("interpolate_pos_encoding", False)
        if not isinstance(interpolate_pos_encoding, bool):
            raise TypeError("interpolate_pos_encoding must be a boolean")
        self.interpolate_pos_encoding = interpolate_pos_encoding

        if siglip2 is None:
            pretrained_config = AutoConfig.from_pretrained(self.id)
            vision_config = getattr(
                pretrained_config, "vision_config", pretrained_config
            )
            if not isinstance(vision_config, SiglipVisionConfig):
                raise TypeError(
                    "the configured checkpoint must provide a SiglipVisionConfig"
                )
            self.siglip2_config = vision_config
            self.visual_encoder = SiglipVisionModel(vision_config)
        else:
            self.visual_encoder = siglip2
            self.siglip2_config = siglip2.config

        if not self.visual_encoder.vision_model.use_head:
            raise ValueError(
                "Siglip2Backbone requires the vision attention-pooling head"
            )

        if self.freeze_visual_encoder:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad = False
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

        # Protect pretrained/injected encoder weights and the deliberate
        # uniform-mean fusion initialization from SltModel.post_init().
        mark_module_tree_as_initialized(self)

    def train(self, mode: bool = True):
        """Train optional fusion modules while a frozen encoder stays in eval."""
        super().train(mode)
        if self.freeze_visual_encoder:
            self.visual_encoder.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        t_lengths: torch.Tensor | None = None,
    ) -> VisualBackboneOutput:
        """Encode packed video frames shaped ``[F, 3, H, W]``."""
        self._validate_inputs(x, t_lengths)
        outputs = self.visual_encoder(
            pixel_values=x,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("SigLIP 2 did not return hidden states")

        selected_features = self._select_hidden_states(outputs.hidden_states)
        # HF hidden_states are captured before the final post-layer norm. Apply
        # the native SigLIP normalization and attention-pooling head to every
        # selected layer so both single- and multi-layer outputs share the same
        # representation contract as the ordinary final model output.
        vision_model = self.visual_encoder.vision_model
        feature_layers = [
            vision_model.post_layernorm(features) for features in selected_features
        ]
        summaries = [vision_model.head(features) for features in feature_layers]

        extras = None
        if len(feature_layers) == 1:
            features = feature_layers[0]
            summary = summaries[0]
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
            visual_features=features,
            pooled_visual_features=summary,
            visual_length=t_lengths,
            extras=extras,
        )

    def _select_hidden_states(
        self, hidden_states: tuple[torch.Tensor, ...]
    ) -> list[torch.Tensor]:
        num_states = len(hidden_states)
        normalized_indices = []
        for layer in self.output_layers:
            normalized_layer = layer if layer >= 0 else num_states + layer
            if not 0 <= normalized_layer < num_states:
                raise IndexError(
                    f"SigLIP 2 output layer {layer} is out of range for "
                    f"{num_states} hidden states"
                )
            normalized_indices.append(normalized_layer)
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("output_layer entries resolve to duplicate layers")
        return [hidden_states[index] for index in normalized_indices]

    @staticmethod
    def _normalize_output_layers(output_layer: int | Sequence[int]) -> tuple[int, ...]:
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
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) for layer in layers
        ):
            raise TypeError("every output_layer entry must be an integer")
        if len(set(layers)) != len(layers):
            raise ValueError("output_layer must not contain duplicate layer indices")
        return layers

    def _validate_inputs(self, x: torch.Tensor, t_lengths: torch.Tensor | None) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 4:
            raise ValueError(
                f"x must have packed-frame shape [F, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one frame")
        if x.shape[1] != self.siglip2_config.num_channels:
            raise ValueError(
                f"x must contain {self.siglip2_config.num_channels} channels, "
                f"got {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must use a floating-point dtype, got {x.dtype}")

        if not self.interpolate_pos_encoding:
            image_size = self.siglip2_config.image_size
            expected_size = (
                (image_size, image_size)
                if isinstance(image_size, int)
                else tuple(image_size)
            )
            if tuple(x.shape[-2:]) != expected_size:
                raise ValueError(
                    "SigLIP 2 input spatial size must match its configured "
                    f"image size {expected_size}, got {tuple(x.shape[-2:])}; "
                    "resize inputs or enable interpolate_pos_encoding"
                )

        if not self._input_values_validated:
            min_value, max_value = torch.aminmax(x.detach())
            if not math.isfinite(float(min_value.item())) or not math.isfinite(
                float(max_value.item())
            ):
                raise ValueError("x must contain only finite values")
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
        if bool((t_lengths <= 0).any()):
            raise ValueError("all t_lengths entries must be positive")
        if int(t_lengths.sum().item()) != x.shape[0]:
            raise ValueError("t_lengths.sum() must equal the packed frame count")

    @classmethod
    def from_pretrained_backbone(cls, config: dict, dtype="auto"):
        model_id = config.get("id")
        if model_id is None:
            raise ValueError("id must be provided in config for Siglip2Backbone")

        siglip2 = SiglipVisionModel.from_pretrained(model_id, dtype=dtype)
        logger.info("Loaded pretrained SigLIP 2 vision model from %s", model_id)
        return cls(config=config, siglip2=siglip2)
