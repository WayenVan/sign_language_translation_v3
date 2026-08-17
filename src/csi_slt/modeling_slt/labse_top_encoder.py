import logging
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.masking_utils import create_bidirectional_mask

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    packed_to_padded,
    padded_to_packed,
)
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput

logger = logging.getLogger(__name__)


class LaBSETopEncoder(nn.Module):
    """Run hidden states through the final Transformer layers of LaBSE.

    The complete pretrained checkpoint is loaded first so that selecting the
    final ``num_layers`` preserves their original checkpoint mapping. Word,
    token-type, position-embedding and pooler modules are intentionally
    discarded.

    Inputs are expected to have already been projected to ``hidden_size``.
    Freezing this module prevents updates to LaBSE while still allowing
    gradients to flow through it to an upstream visual adapter.
    """

    def __init__(
        self,
        config: dict,
        labse: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(labse, nn.Module):
            raise TypeError("labse must be an externally constructed nn.Module")

        self.config = self._validate_config(config)
        self.id = self.config["id"]
        num_layers = self.config["num_layers"]
        freeze = self.config["freeze"]

        base_model = labse.base_model
        if not hasattr(base_model, "encoder") or not hasattr(
            base_model.encoder, "layer"
        ):
            raise TypeError("the LaBSE checkpoint must expose encoder.layer")

        encoder_layers = base_model.encoder.layer
        total_layers = len(encoder_layers)
        if num_layers > total_layers:
            raise ValueError(
                f"num_layers ({num_layers}) exceeds the checkpoint's "
                f"{total_layers} encoder layers"
            )

        self.labse_config = labse.config
        self.config["model_config"] = self.labse_config.to_dict()
        self.hidden_size = int(self.labse_config.hidden_size)
        self.num_layers = num_layers
        self.source_layer_indices = tuple(
            range(total_layers - num_layers, total_layers)
        )
        self.freeze = freeze
        self.layers = nn.ModuleList(list(encoder_layers[-num_layers:]))

        if freeze:
            self.requires_grad_(False)
            self.eval()

        # Prevent an enclosing Hugging Face model's post_init() from replacing
        # the selected pretrained weights.
        mark_module_tree_as_initialized(self)

    @classmethod
    def from_pretrained_encoder(cls, config: dict, dtype="auto"):
        """Load the configured LaBSE checkpoint before selecting its top layers."""
        config = cls._validate_config(config)
        model_id = config["id"]

        labse = AutoModel.from_pretrained(model_id, dtype=dtype)
        logger.info("Loaded pretrained LaBSE encoder from %s", model_id)
        return cls(config=config, labse=labse)

    @classmethod
    def from_encoder_config(cls, config: dict):
        """Construct an encoder skeleton without loading pretrained weights."""
        config = cls._validate_config(config)
        serialized_model_config = config.get("model_config")
        if not isinstance(serialized_model_config, dict):
            raise ValueError(
                "config.encoder.model_config is required when constructing "
                "LaBSETopEncoder without pretrained weights"
            )
        serialized_model_config = dict(serialized_model_config)
        model_type = serialized_model_config.pop("model_type", None)
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("config.encoder.model_config must contain model_type")
        labse_config = AutoConfig.for_model(model_type, **serialized_model_config)
        labse = AutoModel.from_config(labse_config)
        return cls(config=config, labse=labse)

    @staticmethod
    def _validate_config(config: dict) -> dict:
        """Validate and normalize constructor options without loading a model."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        normalized = {
            "num_layers": 4,
            "freeze": True,
            **config,
        }
        model_id = normalized.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(
                "id must be a non-empty string in config for LaBSETopEncoder"
            )

        num_layers = normalized["num_layers"]
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not isinstance(normalized["freeze"], bool):
            raise TypeError("freeze must be a boolean")

        return normalized

    def train(self, mode: bool = True):
        """Keep frozen LaBSE layers deterministic while surrounding modules train."""
        super().train(mode)
        if self.freeze:
            super().train(False)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``[batch, time, hidden_size]`` projected visual features."""
        self._validate_inputs(hidden_states, attention_mask)
        extended_attention_mask = self._extend_attention_mask(
            attention_mask, hidden_states
        )
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            # Transformers 4.x returns a tuple here, while 5.x returns the
            # hidden-state tensor directly.
            hidden_states = (
                layer_outputs[0]
                if isinstance(layer_outputs, tuple)
                else layer_outputs
            )

        return hidden_states

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, time, hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be a floating-point tensor")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"hidden_states width must be {self.hidden_size}, got "
                f"{hidden_states.size(-1)}"
            )
        if attention_mask is None:
            return
        if attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "attention_mask must have shape [batch, time] matching "
                f"hidden_states, got {tuple(attention_mask.shape)}"
            )
        valid_values = torch.logical_or(attention_mask == 0, attention_mask == 1)
        if not bool(valid_values.all()):
            raise ValueError("attention_mask values must be 0 or 1")

    def _extend_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=hidden_states.device)
        return create_bidirectional_mask(
            config=self.labse_config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )


class LaBSESemanticEncoder(nn.Module):
    """Apply a top-layer LaBSE encoder and projector to packed visual tokens."""

    def __init__(
        self,
        config: dict,
        encoder: LaBSETopEncoder,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, LaBSETopEncoder):
            raise TypeError("encoder must be an externally constructed LaBSETopEncoder")

        self.config = self._validate_config(config)
        self.encoder = encoder
        self.config["encoder"] = dict(encoder.config)
        projector_config = self.config["projector"]
        position_config = self.config["position_embedding"]
        self.input_dim = encoder.hidden_size
        self.hidden_dim = projector_config["hidden_dim"]
        self.output_dim = projector_config["output_dim"]
        self.max_position_embeddings = position_config["max_positions"]
        self.input_layernorm = nn.LayerNorm(self.input_dim)
        self.position_embeddings = nn.Embedding(
            self.max_position_embeddings, self.input_dim
        )
        nn.init.normal_(
            self.position_embeddings.weight,
            mean=0.0,
            std=position_config["init_std"],
        )
        self.residual_gate = nn.Parameter(
            torch.tensor(self.config["residual_gate_init"], dtype=torch.float32)
        )
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    @classmethod
    def from_pretrained_encoder(cls, config: dict, dtype="auto"):
        """Load the nested LaBSE top encoder and construct its projector."""
        config = cls._validate_config(config)
        encoder = LaBSETopEncoder.from_pretrained_encoder(
            config["encoder"], dtype=dtype
        )
        return cls(config=config, encoder=encoder)

    @classmethod
    def from_encoder_config(cls, config: dict):
        """Construct the nested encoder skeleton for checkpoint loading."""
        config = cls._validate_config(config)
        encoder = LaBSETopEncoder.from_encoder_config(config["encoder"])
        return cls(config=config, encoder=encoder)

    @staticmethod
    def _validate_config(config: dict) -> dict:
        """Validate and normalize nested encoder/projector configuration."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        encoder_config = config.get("encoder")
        projector_config = config.get("projector")
        if not isinstance(encoder_config, dict):
            raise TypeError("config.encoder must be a dictionary")
        if not isinstance(projector_config, dict):
            raise TypeError("config.projector must be a dictionary")

        encoder_config = LaBSETopEncoder._validate_config(encoder_config)
        projector_config = dict(projector_config)
        output_dim = projector_config.get("output_dim")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int):
            raise TypeError("config.projector.output_dim must be an integer")
        if output_dim <= 0:
            raise ValueError("config.projector.output_dim must be positive")

        hidden_dim = projector_config.get("hidden_dim", output_dim)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("config.projector.hidden_dim must be an integer")
        if hidden_dim <= 0:
            raise ValueError("config.projector.hidden_dim must be positive")
        projector_config["hidden_dim"] = hidden_dim

        position_config = config.get("position_embedding", {})
        if not isinstance(position_config, dict):
            raise TypeError("config.position_embedding must be a dictionary")
        position_config = {
            "max_positions": 1024,
            "init_std": 0.02,
            **position_config,
        }
        max_positions = position_config["max_positions"]
        if isinstance(max_positions, bool) or not isinstance(max_positions, int):
            raise TypeError(
                "config.position_embedding.max_positions must be an integer"
            )
        if max_positions <= 0:
            raise ValueError(
                "config.position_embedding.max_positions must be positive"
            )
        init_std = position_config["init_std"]
        if isinstance(init_std, bool) or not isinstance(init_std, (int, float)):
            raise TypeError("config.position_embedding.init_std must be a real number")
        if not math.isfinite(init_std) or init_std < 0.0:
            raise ValueError(
                "config.position_embedding.init_std must be finite and non-negative"
            )
        position_config["init_std"] = float(init_std)

        residual_gate_init = config.get("residual_gate_init", -4.0)
        if isinstance(residual_gate_init, bool) or not isinstance(
            residual_gate_init, (int, float)
        ):
            raise TypeError("config.residual_gate_init must be a real number")
        if not math.isfinite(residual_gate_init):
            raise ValueError("config.residual_gate_init must be finite")

        return {
            **config,
            "encoder": encoder_config,
            "projector": projector_config,
            "position_embedding": position_config,
            "residual_gate_init": float(residual_gate_init),
        }

    def forward(self, visual_output: VisualAdapterOutput) -> VisualAdapterOutput:
        """Contextualize packed visual tokens and project them to the LLM width."""
        features, lengths = self._validate_visual_output(visual_output)
        padded_features, attention_mask = packed_to_padded(features, lengths)
        sequence_length = padded_features.size(1)
        if sequence_length > self.max_position_embeddings:
            raise ValueError(
                f"visual sequence length {sequence_length} exceeds semantic "
                f"position embedding capacity {self.max_position_embeddings}"
            )
        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=padded_features.device,
        ).unsqueeze(0)
        encoder_inputs = self.input_layernorm(
            padded_features + self.position_embeddings(position_ids)
        )
        encoded_features = self.encoder(encoder_inputs, attention_mask)
        residual_weight = torch.sigmoid(self.residual_gate).to(
            dtype=encoded_features.dtype
        )
        encoded_features = encoder_inputs + residual_weight * (
            encoded_features - encoder_inputs
        )
        packed_features, encoded_lengths = padded_to_packed(
            encoded_features, attention_mask
        )
        if not torch.equal(
            encoded_lengths,
            lengths.to(device=encoded_lengths.device, dtype=torch.long),
        ):
            raise RuntimeError("semantic encoder changed the packed visual lengths")
        projected_features = self.projector(packed_features)

        return VisualAdapterOutput(
            visual_features=projected_features,
            visual_length=visual_output.visual_length,
            position_ids=visual_output.position_ids,
            extras=visual_output.extras,
        )

    def _validate_visual_output(
        self, visual_output: VisualAdapterOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(visual_output, VisualAdapterOutput):
            raise TypeError("visual_output must be a VisualAdapterOutput")
        features = visual_output.visual_features
        lengths = visual_output.visual_length
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("visual_features must have shape [sum(T), hidden_size]")
        if not features.is_floating_point():
            raise TypeError("visual_features must be a floating-point tensor")
        if features.size(1) != self.input_dim:
            raise ValueError(
                f"visual_features width must be {self.input_dim}, got "
                f"{features.size(1)}"
            )
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("visual_length must be a torch.Tensor")
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if lengths.is_floating_point() or lengths.is_complex():
            raise TypeError("visual_length must use an integer dtype")
        if bool((lengths <= 0).any()):
            raise ValueError("all visual_length entries must be positive")
        if int(lengths.sum().item()) != features.size(0):
            raise ValueError("visual_length.sum() must equal the packed token count")
        return features, lengths
