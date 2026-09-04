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

        self.output_layer = self._normalize_output_layer(
            config.get("output_layer", -8)
        )
        self.attention_layer = self._validate_layer(
            "attention_layer", config.get("attention_layer", -25)
        )
        self.config = config
        self._input_values_validated = False
        if "freeze_visual_encoder" in config:
            logger.warning(
                "Ignoring retired C-RADIO freeze_visual_encoder=%r; configure "
                "engine.trainability.visual_backbone.parameter_mode and "
                "runtime_mode instead",
                config["freeze_visual_encoder"],
            )
        # Construction starts from the safest standalone state. The training
        # engine later owns both requires_grad and runtime-mode decisions.
        self.runtime_mode = "eval"

        if c_radio_v4 is None:
            self.c_radio_v4_config = AutoConfig.from_pretrained(
                self.id, trust_remote_code=True
            )
            self.visual_encoder = AutoModel.from_config(
                self.c_radio_v4_config, trust_remote_code=True
            )
        else:
            self.visual_encoder = c_radio_v4
            self.c_radio_v4_config = c_radio_v4.config

        # Keep standalone construction safe until the engine applies the
        # explicit visual-backbone trainability and runtime plan.
        self.visual_encoder.requires_grad_(False)
        self.set_runtime_mode("eval")

        # Preserve the encoder's initialization/checkpoint and the fusion
        # modules' uniform-mean initialization when attached to SltModel.
        mark_module_tree_as_initialized(self)

    def train(self, mode: bool = True):
        """Apply the engine-selected encoder runtime mode explicitly."""
        super().train(mode)
        # Whole-model eval always wins. During training, runtime_mode controls
        # stochastic encoder behavior independently of parameter gradients.
        self.visual_encoder.train(mode and self.runtime_mode == "train")
        return self

    def set_runtime_mode(self, runtime_mode: str) -> None:
        """Set ``train``/``eval`` behavior without changing requires_grad."""
        if runtime_mode not in ("eval", "train"):
            raise ValueError(
                f"C-RADIO runtime_mode must be 'eval' or 'train', got {runtime_mode!r}"
            )
        self.runtime_mode = runtime_mode
        self.visual_encoder.train(self.training and runtime_mode == "train")

    def forward(
        self,
        x,
        t_lengths=None,
        *,
        return_attention_maps: bool = False,
    ) -> VisualBackboneOutput:
        """
        video: Packed frames with shape [F, C, H, W].
        """
        self._validate_inputs(x, t_lengths)
        if not isinstance(return_attention_maps, bool):
            raise TypeError("return_attention_maps must be a boolean")
        captured_attention_input = None
        attention_module = None
        hook_handle = None
        if return_attention_maps:
            attention_module = self._get_attention_module(self.attention_layer)

            def capture_attention_input(_module, args):
                nonlocal captured_attention_input
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError(
                        "C-RADIO attention module did not receive a tensor input"
                    )
                captured_attention_input = args[0]

            hook_handle = attention_module.register_forward_pre_hook(
                capture_attention_input
            )

        try:
            radio_output = self._forward_intermediates(
                x, include_attention_layer=return_attention_maps
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        extras = None
        summary = radio_output.summary
        features = radio_output.features

        if return_attention_maps:
            if captured_attention_input is None or attention_module is None:
                raise RuntimeError(
                    "the requested C-RADIO attention layer was not executed"
                )
            cls_attention = self._compute_cls_patch_attention(
                attention_module,
                captured_attention_input,
                patch_count=features.shape[1],
            )
            extras = {} if extras is None else extras
            extras["attention_maps"] = cls_attention
            extras["attention_layer"] = self.attention_layer

        return VisualBackboneOutput(
            visual_features=features,  # [B, T, C]
            pooled_visual_features=summary,  # [B, C]
            visual_length=t_lengths,
            extras=extras,
        )

    def _get_attention_module(self, attention_layer: int) -> nn.Module:
        """Find one ViT attention block by the same index used for intermediates."""
        radio_model = getattr(self.visual_encoder, "radio_model", self.visual_encoder)
        attention_modules = [
            module
            for module in radio_model.modules()
            if isinstance(getattr(module, "qkv", None), nn.Module)
            and isinstance(getattr(module, "num_heads", None), int)
        ]
        if not attention_modules:
            raise TypeError(
                "C-RADIO encoder does not expose ViT attention modules with qkv "
                "and num_heads attributes"
            )
        try:
            return attention_modules[attention_layer]
        except IndexError as error:
            raise IndexError(
                f"attention_layer {attention_layer} is out of range for the "
                f"{len(attention_modules)} discovered C-RADIO attention layers"
            ) from error

    @staticmethod
    def _compute_cls_patch_attention(
        attention_module: nn.Module,
        hidden_states: torch.Tensor,
        *,
        patch_count: int,
    ) -> torch.Tensor:
        """Reconstruct per-head CLS-to-patch attention from a timm-style block."""
        qkv = attention_module.qkv(hidden_states)
        if not isinstance(qkv, torch.Tensor) or qkv.ndim != 3:
            raise RuntimeError(
                "C-RADIO attention qkv projection must return [F, N, 3 * C]"
            )
        frame_count, token_count, triple_width = qkv.shape
        num_heads = attention_module.num_heads
        if triple_width % (3 * num_heads) != 0:
            raise RuntimeError(
                f"qkv width {triple_width} is not divisible by 3 * {num_heads} heads"
            )
        head_dim = triple_width // (3 * num_heads)
        qkv = qkv.reshape(frame_count, token_count, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key = qkv[0], qkv[1]

        q_norm = getattr(attention_module, "q_norm", None)
        k_norm = getattr(attention_module, "k_norm", None)
        if isinstance(q_norm, nn.Module):
            query = q_norm(query)
        if isinstance(k_norm, nn.Module):
            key = k_norm(key)

        scale = getattr(attention_module, "scale", head_dim**-0.5)
        attention = (query * scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        if patch_count <= 0 or patch_count >= token_count:
            raise RuntimeError(
                f"invalid patch token count {patch_count} for {token_count} total tokens"
            )
        # C-RADIO/timm places CLS and optional register tokens before patches.
        return attention[:, :, 0, token_count - patch_count :]

    @staticmethod
    def _validate_layer(name: str, layer: object) -> int:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"{name} must be an integer")
        return layer

    @classmethod
    def _normalize_output_layer(cls, output_layer: object) -> int:
        """Accept legacy one-item layer lists while keeping one output layer."""
        if isinstance(output_layer, Sequence) and not isinstance(
            output_layer, (str, bytes)
        ):
            if len(output_layer) != 1:
                raise ValueError(
                    "output_layer must contain exactly one layer when provided "
                    "as a sequence"
                )
            output_layer = output_layer[0]
        return cls._validate_layer("output_layer", output_layer)

    def _forward_intermediates(
        self, x: torch.Tensor, *, include_attention_layer: bool = False
    ):
        radio_model = getattr(self.visual_encoder, "radio_model", self.visual_encoder)
        forward_intermediates = getattr(radio_model, "forward_intermediates", None)
        if forward_intermediates is None:
            raise TypeError(
                "C-RADIO encoder must expose forward_intermediates to fuse layers"
            )
        indices = [self.output_layer]
        if include_attention_layer and self.attention_layer != self.output_layer:
            indices.append(self.attention_layer)
        outputs = forward_intermediates(
            x,
            indices=indices,
            return_prefix_tokens=True,
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
            aggregation="sparse",
        )
        if len(outputs) != len(indices):
            raise RuntimeError(
                "C-RADIO returned an unexpected number of intermediate layers: "
                f"expected {len(indices)}, got {len(outputs)}"
            )
        return outputs[0]

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
