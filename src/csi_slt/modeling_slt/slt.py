import math
import warnings
from copy import deepcopy
from typing import Callable, Optional

import torch
from torch import nn
from transformers import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_utils import PreTrainedModel

from csi_slt.modeling_slt.info_utils import (
    InformationRequest,
    build_information_output,
)
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from ..configuration_slt.configuration import SltConfig
from .output_utils import (
    PrepareForCausalLMOutput,
    SltCausalLMOutputWithPast,
    VisualAdapterOutput,
    VisualBackboneOutput,
)
from .misc import (
    mark_module_tree_as_initialized,
    packed_to_padded,
)
from .registry import (
    VISUAL_ADAPTERS,
    VISUAL_BACKBONES,
    VISUAL_SEMANTIC_ENCODERS,
)

logger = logging.get_logger(__name__)


def _serialize_lora_config(config: LoraConfig) -> dict:
    """Convert a PEFT config to values supported by JSON serialization."""
    return {
        key: list(value) if isinstance(value, set) else value
        for key, value in config.to_dict().items()
    }


def get_llm_cls_by_model_name(model_name):
    """Return the supported causal language-model class for ``model_name``."""
    if "qwen" in model_name.lower():
        from transformers.models.qwen3 import Qwen3ForCausalLM

        model_cls = Qwen3ForCausalLM
    elif "gemma-3" in model_name.lower():
        from transformers.models.gemma3 import (
            Gemma3ForCausalLM,
            Gemma3ForConditionalGeneration,
        )

        model_cls = (
            Gemma3ForCausalLM
            if "1b" in model_name.lower()
            else Gemma3ForConditionalGeneration
        )
    elif "gemma-4" in model_name.lower():
        from transformers.models.gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )

        model_cls = Gemma4UnifiedForConditionalGeneration

    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")
    return model_cls


def token_type_ids_mask_function(
    token_type_ids: Optional[torch.Tensor],
) -> Optional[Callable]:
    """Create a mask that allows video tokens to attend to other video tokens."""
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        is_video_block_q = token_type_ids[batch_idx, q_idx] == 1
        is_video_block_kv = token_type_ids[batch_idx, kv_idx] == 1

        return is_video_block_q & is_video_block_kv

    return inner_mask


class SltModel(PreTrainedModel, GenerationMixin):
    config_class = SltConfig
    MAX_TOKEN_LENGTH = 1024

    # _tied_weights_keys = {"llm.lm_head.weight": "model.embed_tokens.weight"}
    _keep_in_fp32_modules = ["visual_adapter"]

    # The bidirectional video-token overlay is an arbitrary 4D mask. SDPA and
    # FlexAttention can express one; FlashAttention's kernel API only knows
    # "causal + varlen padding", and its mask builder ignores the mask function
    # outright (``masking_utils.flash_attention_mask``), which would silently
    # degrade the model to plain causal attention. Declaring the capability here
    # makes Transformers reject an unusable request at load time instead.
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn = False
    # Attention implementations whose mask builder can carry the overlay.
    _BIDIRECTIONAL_MASK_IMPLEMENTATIONS = ("sdpa", "eager", "flex_attention")

    def __init__(
        self,
        config: SltConfig,
        llm: Optional[nn.Module] = None,
        visual_backbone: Optional[nn.Module] = None,
        visual_semantic_encoder: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        for component_name, component in (
            ("llm", llm),
            ("visual_backbone", visual_backbone),
        ):
            if component is not None and not getattr(
                component, "_is_hf_initialized", False
            ):
                logger.warning(
                    "Externally supplied %s is not marked as initialized and "
                    "may be initialized by SltModel.post_init(). If it already "
                    "contains trained or loaded weights, call "
                    "mark_module_tree_as_initialized(%s) before constructing "
                    "SltModel.",
                    component_name,
                    component_name,
                )

        # Always construct the model structure here to support meta-device models.
        self.llm = llm
        self.visual_backbone = visual_backbone
        self.visual_semantic_encoder = visual_semantic_encoder

        # Initialize the visual backbone when it was not supplied by the caller.
        if self.visual_backbone is None:
            backbone_cls = VISUAL_BACKBONES.get(config.visual_backbone_type, None)
            if backbone_cls is None:
                raise ValueError(
                    f"Unsupported visual backbone type: "
                    f"{config.visual_backbone_type}. Supported types are: "
                    f"{list(VISUAL_BACKBONES.keys())}"
                )
            self.visual_backbone = backbone_cls(config.visual_backbone_config)

        # The visual adapter is always constructed from the SLT configuration.
        adapter_cls = VISUAL_ADAPTERS.get(config.visual_adapter_type, None)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported visual adapter type: "
                f"{config.visual_adapter_type}. Supported types are: "
                f"{list(VISUAL_ADAPTERS.keys())}"
            )
        self.visual_adapter = adapter_cls(**config.visual_adapter_kwargs)

        semantic_encoder_type = config.visual_semantic_encoder_type
        if semantic_encoder_type is None:
            if self.visual_semantic_encoder is not None:
                raise ValueError(
                    "visual_semantic_encoder was supplied while "
                    "visual_semantic_encoder_type is None"
                )
        else:
            semantic_encoder_cls = VISUAL_SEMANTIC_ENCODERS.get(semantic_encoder_type)
            if semantic_encoder_cls is None:
                raise ValueError(
                    f"Unsupported visual semantic encoder type: "
                    f"{semantic_encoder_type}. Supported types are: "
                    f"{list(VISUAL_SEMANTIC_ENCODERS.keys())}"
                )
            if self.visual_semantic_encoder is None:
                self.visual_semantic_encoder = semantic_encoder_cls.from_encoder_config(
                    config.visual_semantic_encoder_config
                )
            semantic_output_dim = getattr(
                self.visual_semantic_encoder, "output_dim", None
            )
            if semantic_output_dim != config.hidden_size:
                raise ValueError(
                    "visual semantic encoder output_dim must match the LLM "
                    f"hidden size ({config.hidden_size}), got "
                    f"{semantic_output_dim}"
                )

        # Initialize the language model when it was not supplied by the caller.
        if self.llm is None:
            llm_cls = get_llm_cls_by_model_name(config.llm_model_name_or_path)
            self.llm = llm_cls._from_config(config.llm_config)

        # Keep exactly one configuration object for the language model.
        # ``from_pretrained_components`` builds the LLM from its own hub config,
        # which leaves ``config.llm_config`` as an object no model ever
        # instantiated: runtime-resolved fields such as ``_attn_implementation``
        # stay unset on it. Since ``forward`` builds the attention mask from
        # ``config.get_text_config()`` while the LLM dispatches its kernel from
        # ``llm.config``, two objects mean the mask and the kernel can disagree.
        # Transformers itself compares these by identity (``modeling_utils``
        # ``set_attn_implementation``), so bind them together here.
        self.config.llm_config = self.llm.config
        # Re-run the setter so this config and its sub-config agree on whatever
        # the LLM actually resolved.
        self.config._attn_implementation = self.llm.config._attn_implementation

        self._configure_generation()

        # Visual-backbone freezing is owned by each backbone implementation so
        # trainable fusion/adaptation parameters are not frozen accidentally.
        for param in self.llm.parameters():
            param.requires_grad = False

        # WARN: new lora code start here
        if config.llm_lora:
            if not config.llm_lora_config:
                raise ValueError(
                    "llm_lora_config must be provided when llm_lora is True."
                )
            self.llm = get_peft_model(self.llm, LoraConfig(**config.llm_lora_config))
            # PEFT initializes the newly injected LoRA modules itself. Keep the
            # outer SltModel.post_init() from replacing that initialization.
            mark_module_tree_as_initialized(self.llm)

        if config.visual_lora:
            if not config.visual_lora_config:
                raise ValueError(
                    "visual_lora_config must be provided when visual_lora is True."
                )
            self._inject_visual_lora(LoraConfig(**config.visual_lora_config))

        # Keep the existing attribute names for checkpoint compatibility.
        self.start_video_embds = nn.Parameter(
            torch.empty(
                1, self.config.hidden_size, dtype=torch.float32, device=self.device
            ),
            requires_grad=True,
        )
        self.end_video_embeds = nn.Parameter(
            torch.empty(
                1, self.config.hidden_size, dtype=torch.float32, device=self.device
            ),
            requires_grad=True,
        )
        # Temporal position of a visual token, added before the token enters
        # the LLM (which then applies its own RoPE on top). Only the learned
        # mode owns parameters; "none" and "sincos" leave the state dict without
        # a ``visual_position_embedding.*`` entry at all.
        position_embedding_type = config.visual_position_embedding_type
        self.visual_position_embedding = (
            nn.Embedding(self.MAX_TOKEN_LENGTH, self.config.hidden_size)
            if position_embedding_type == "learned"
            else None
        )
        # Non-persistent: the table is a pure function of MAX_TOKEN_LENGTH and
        # hidden_size, so serializing it would only add weight to every
        # checkpoint and let a stale copy override the formula on reload.
        self.register_buffer(
            "visual_position_table",
            self._build_sincos_position_table(
                self.MAX_TOKEN_LENGTH, self.config.hidden_size
            )
            if position_embedding_type == "sincos"
            else None,
            persistent=False,
        )
        # Global learnable scale, initialized as an identity transform. Shape
        # (1,) rather than a scalar: FSDP2 shards along dim 0 and rejects 0-dim
        # parameters. Broadcasting against the visual features is unchanged.
        self.visual_scale = nn.Parameter(torch.tensor([1.0]))
        # CTC head over visual tokens, predicting the word-level pseudo-gloss
        # vocabulary. Only constructed when the CTC objective is enabled;
        # otherwise the model carries no extra CTC parameters.
        self.ctc_head = (
            nn.Linear(self.config.hidden_size, config.ctc_vocab_size)
            if config.ctc_enabled
            else None
        )
        # The adapter projection and learned visual positions can have a
        # different scale from the frozen LLM's token embeddings. Normalize the
        # completed visual token (projection + position) immediately before it
        # is merged into the LLM input sequence.
        # self.visual_output_norm = nn.RMSNorm(self.config.hidden_size, eps=1e-6)
        self.config.num_extra_tokens = 2  # Start and end video tokens.
        self.config.is_encoder_decoder = False
        self.config.is_decoder = True
        self.post_init()

        # NOTE: tie the weights when using LoRA initialization
        if config.llm_lora:
            self._register_llm_tied_weights()

        # Set once by the first forward that checks the materialized mask.
        self._bidirectional_mask_validated = False
        self._validate_attention_support()

    @property
    def text_config(self):
        """Config that drives both the LLM attention dispatch and mask creation.

        ``__init__`` binds ``config.llm_config`` to ``llm.config``, so both
        routes reach the same object. Reading it through the language model
        keeps the invariant visible at the one place where a mismatch matters.
        """
        return self.llm.config.get_text_config()

    def _validate_attention_support(self) -> None:
        """Reject at construction an attention that cannot carry the overlay."""
        if not self.config.video_bidirectional_attention:
            return
        implementation = self.text_config._attn_implementation
        if implementation not in self._BIDIRECTIONAL_MASK_IMPLEMENTATIONS:
            raise ValueError(
                "video_bidirectional_attention=True needs an attention "
                "implementation whose mask builder can carry a custom mask "
                f"({', '.join(self._BIDIRECTIONAL_MASK_IMPLEMENTATIONS)}), but "
                f"the language model resolved to {implementation!r}. Load with "
                'attn_implementation="sdpa" (or "flex_attention"), or set '
                "video_bidirectional_attention=False."
            )

    @staticmethod
    def _mask_allows(causal_mask, batch_index, query_index, key_index, device):
        """Read one entry of a materialized mask, whatever form it takes."""
        # FlexAttention returns a BlockMask, which carries the predicate itself
        # rather than a dense tensor.
        if hasattr(causal_mask, "mask_mod"):
            indices = [
                torch.tensor(value, device=device)
                for value in (batch_index, 0, query_index, key_index)
            ]
            return bool(causal_mask.mask_mod(*indices))
        entry = causal_mask[batch_index, 0, query_index, key_index]
        if entry.dtype == torch.bool:
            return bool(entry)
        # Eager masks are additive: blocked positions hold the dtype minimum.
        return bool(entry > torch.finfo(entry.dtype).min / 2)

    def _validate_bidirectional_mask(self, causal_mask, token_type_ids) -> None:
        """Check the overlay actually survived into the materialized mask.

        Only called for a prefill that requested the overlay. The ``None`` check
        runs every time, because ``_attn_implementation`` can still be changed
        after construction. The structural check runs once per model, in the
        spirit of ``CRadioV4Backbone._validate_inputs``, since reading mask
        entries synchronizes the accelerator.
        """
        if causal_mask is None:
            raise RuntimeError(
                "The bidirectional video attention mask was requested but "
                "create_causal_mask() returned None, so the language model "
                "would silently fall back to plain causal attention -- and to "
                "no padding mask at all. Transformers early-exits mask creation "
                "when the config driving it carries an attention implementation "
                "it cannot build a custom mask for; here it resolved to "
                f"{self.text_config._attn_implementation!r}."
            )
        if self._bidirectional_mask_validated:
            return

        # A non-None mask is not enough: the FlashAttention builder returns the
        # 2D padding mask and drops the overlay entirely. Verify that a
        # video -> later-video pair is open while video -> later-text is not.
        for batch_index in range(token_type_ids.shape[0]):
            row = token_type_ids[batch_index]
            video_positions = (row == 1).nonzero(as_tuple=True)[0]
            if video_positions.numel() < 2:
                continue
            query_index = int(video_positions[0])
            key_index = int(video_positions[-1])
            later_text = (row[key_index + 1 :] == 0).nonzero(as_tuple=True)[0]
            if later_text.numel() == 0:
                continue
            text_index = key_index + 1 + int(later_text[0])

            device = token_type_ids.device
            sees_later_video = self._mask_allows(
                causal_mask, batch_index, query_index, key_index, device
            )
            sees_later_text = self._mask_allows(
                causal_mask, batch_index, query_index, text_index, device
            )
            if not sees_later_video or sees_later_text:
                raise RuntimeError(
                    "The materialized attention mask does not implement "
                    "bidirectional video attention: video token "
                    f"{query_index} -> video token {key_index} is "
                    f"{'open' if sees_later_video else 'BLOCKED'} and video "
                    f"token {query_index} -> text token {text_index} is "
                    f"{'OPEN' if sees_later_text else 'blocked'}. Expected "
                    "open/blocked. attn_implementation="
                    f"{self.text_config._attn_implementation!r}."
                )
            self._bidirectional_mask_validated = True
            return

    def _register_llm_tied_weights(self):
        """Register the actual embedding/head paths after PEFT wraps the LLM."""
        input_embeddings = self.get_input_embeddings()
        output_embeddings = self.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            return

        module_names = {
            id(module): name
            for name, module in self.named_modules(remove_duplicate=False)
        }
        input_name = module_names.get(id(input_embeddings))
        output_name = module_names.get(id(output_embeddings))
        if input_name is None or output_name is None:
            raise RuntimeError(
                "Could not resolve the PEFT-wrapped input/output embedding paths."
            )

        self.all_tied_weights_keys[f"{output_name}.weight"] = f"{input_name}.weight"

    def _inject_visual_lora(self, peft_config: LoraConfig) -> None:
        """Inject LoRA in-place without changing the visual encoder interface."""
        visual_encoder = getattr(self.visual_backbone, "visual_encoder", None)
        if visual_encoder is None:
            raise TypeError(
                f"{type(self.visual_backbone).__name__} does not expose "
                "visual_encoder and cannot use visual LoRA"
            )
        inject_adapter_in_model(peft_config=peft_config, model=visual_encoder)
        mark_module_tree_as_initialized(visual_encoder)

    def inject_llm_lora(self, peft_config: LoraConfig) -> None:
        """Inject a new LoRA adapter into the language model."""
        if self.config.llm_lora:
            raise ValueError(
                "The checkpoint already contains LLM LoRA. "
                "Use SltModel.from_pretrained() to load it."
            )
        self.llm = get_peft_model(self.llm, peft_config)
        mark_module_tree_as_initialized(self.llm)
        self.config.llm_lora = True
        self.config.llm_lora_config = _serialize_lora_config(peft_config)
        self._register_llm_tied_weights()

    def inject_visual_lora(self, peft_config: LoraConfig) -> None:
        """Inject a new LoRA adapter into the visual encoder in-place."""
        if self.config.visual_lora:
            raise ValueError(
                "The checkpoint already contains visual LoRA. "
                "Use SltModel.from_pretrained() to load it."
            )
        self._inject_visual_lora(peft_config)
        self.config.visual_lora = True
        self.config.visual_lora_config = _serialize_lora_config(peft_config)

    @classmethod
    def from_pretrained_components(
        cls,
        config: SltConfig,
        llm_dtype="auto",
        visual_backbone_dtype="auto",
        visual_semantic_encoder_dtype="auto",
    ):
        visual_backbone_cls = VISUAL_BACKBONES.get(config.visual_backbone_type, None)
        if visual_backbone_cls is None:
            raise ValueError(
                f"Unsupported visual backbone type: "
                f"{config.visual_backbone_type}. Supported types are: "
                f"{list(VISUAL_BACKBONES.keys())}"
            )
        visual_backbone = visual_backbone_cls.from_pretrained_backbone(
            config.visual_backbone_config, dtype=visual_backbone_dtype
        )

        visual_semantic_encoder = None
        semantic_encoder_type = config.visual_semantic_encoder_type
        if semantic_encoder_type is not None:
            semantic_encoder_cls = VISUAL_SEMANTIC_ENCODERS.get(semantic_encoder_type)
            if semantic_encoder_cls is None:
                raise ValueError(
                    f"Unsupported visual semantic encoder type: "
                    f"{semantic_encoder_type}. Supported types are: "
                    f"{list(VISUAL_SEMANTIC_ENCODERS.keys())}"
                )
            visual_semantic_encoder = semantic_encoder_cls.from_pretrained_encoder(
                config.visual_semantic_encoder_config,
                dtype=visual_semantic_encoder_dtype,
            )
            # Save the source architecture so SltModel.from_pretrained can
            # reconstruct the semantic encoder without contacting its source.
            config.visual_semantic_encoder_config = dict(visual_semantic_encoder.config)

        llm_cls = get_llm_cls_by_model_name(config.llm_model_name_or_path)
        llm = llm_cls.from_pretrained(config.llm_model_name_or_path, dtype=llm_dtype)
        # Ensure lm_head and input embeddings are tied even when the source model
        # did not tie them.
        llm.tie_weights(recompute_mapping=True)

        # This factory explicitly loaded both components from pretrained
        # checkpoints. Protect them from the outer SltModel.post_init(); the
        # ordinary constructor makes no such assumption about supplied modules.
        mark_module_tree_as_initialized(llm)
        mark_module_tree_as_initialized(visual_backbone)

        logger.info("force retie the lm_head to the input embeddings!!!!!!!!")

        return cls(
            config=config,
            llm=llm,
            visual_backbone=visual_backbone,
            visual_semantic_encoder=visual_semantic_encoder,
        )

        # fmt: on

    @classmethod
    def from_pretrained_components_with_lora(
        cls,
        config: SltConfig,
        peft_config: LoraConfig,
        llm_dtype="auto",
        visual_backbone_dtype="auto",
        visual_semantic_encoder_dtype="auto",
    ):
        model = cls.from_pretrained_components(
            config=config,
            llm_dtype=llm_dtype,
            visual_backbone_dtype=visual_backbone_dtype,
            visual_semantic_encoder_dtype=visual_semantic_encoder_dtype,
        )

        # --------------------------
        # appli freze policy for visual encoder, let the encoder itself decide which parameters to freeze,
        # so that the trainable parameters are not frozen accidentally.
        # -----------------------
        model.visual_backbone.apply_freeze_policy()

        if model.config.llm_lora:
            raise ValueError(
                "The checkpoint already contains LoRA. "
                "Use SltModel.from_pretrained() to load it."
            )

        model.llm = get_peft_model(model.llm, peft_config)
        # The checkpoint weights were already loaded and PEFT initialized the
        # newly injected LoRA modules; the complete wrapped LLM is now ready.
        mark_module_tree_as_initialized(model.llm)

        # setup config
        model.config.llm_lora = True
        model.config.llm_lora_config = {
            key: list(value) if isinstance(value, set) else value
            for key, value in peft_config.to_dict().items()
        }
        model._register_llm_tied_weights()

        return model

    @classmethod
    def from_pretrained_with_new_lora(
        cls,
        peft_config: LoraConfig | None = None,
        checkpoint_dir: str | None = None,
        model_dtype="auto",
        *,
        llm_lora_config: LoraConfig | None = None,
        visual_lora_config: LoraConfig | None = None,
    ):
        """Load a checkpoint and inject LoRA (deprecated compatibility API)."""
        warnings.warn(
            "SltModel.from_pretrained_with_new_lora() is deprecated; call "
            "SltModel.from_pretrained() and then inject_llm_lora() and/or "
            "inject_visual_lora() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if checkpoint_dir is None:
            raise TypeError("checkpoint_dir must be provided")
        if peft_config is not None and llm_lora_config is not None:
            raise ValueError(
                "Pass either the legacy peft_config argument or "
                "llm_lora_config, not both"
            )
        if llm_lora_config is None:
            llm_lora_config = peft_config
        if llm_lora_config is None and visual_lora_config is None:
            raise ValueError("At least one LoRA config must be provided")

        model: SltModel = cls.from_pretrained(checkpoint_dir, dtype=model_dtype)

        if llm_lora_config is not None:
            model.inject_llm_lora(llm_lora_config)
        if visual_lora_config is not None:
            model.inject_visual_lora(visual_lora_config)

        return model

    def _configure_generation(self):
        text_config = self.config.get_text_config()
        generation_config = self.llm.generation_config
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(text_config)
        else:
            generation_config = deepcopy(generation_config)

        # Preserve model-specific generation token ids (notably Gemma 4's
        # multiple EOS ids) and only fall back to the text config when the
        # generation config does not define one.
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(generation_config, name, None) is None:
                value = getattr(text_config, name, None)
                if value is not None:
                    setattr(generation_config, name, value)

        generation_config.do_sample = False
        generation_config.top_k = None
        generation_config.top_p = None
        generation_config.temperature = None

        self.generation_config = generation_config
        self.has_sliding_layers = "sliding_attention" in text_config.layer_types

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize modules owned by SltModel through Hugging Face post_init."""
        super()._init_weights(module)

        if (
            self.visual_position_embedding is not None
            and module is self.visual_position_embedding
        ):
            nn.init.trunc_normal_(module.weight, std=0.02)

        # Raw parameters are not handled by the generic module-type rules.
        # smart_apply visits children before their parent, so the LLM token
        # embeddings are initialized before this root-model branch runs.
        if module is self:
            mean = self.llm.get_input_embeddings().weight.mean(dim=0, keepdim=True)
            self.start_video_embds.copy_(mean)
            self.end_video_embeds.copy_(mean)

    @staticmethod
    def _build_sincos_position_table(
        max_positions: int, hidden_size: int
    ) -> torch.Tensor:
        """Build the fixed sinusoidal table of ``visual_position_embedding_type``.

        The classic interleaved formulation, with one difference: every row is
        L2-normalized. Raw sinusoids give each row a norm of ``sqrt(D/2)`` --
        about 32 at ``D=2048``, some 35x the learned table's initialization
        (``0.02 * sqrt(D) = 0.905``) and 20x the LLM's own token embeddings --
        which would drown the visual content the row is added to. Unit rows put
        the two modes on the same scale, so switching between them measures the
        encoding rather than its amplitude.
        """
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        positions = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        angles = positions * frequencies
        table = torch.zeros(max_positions, hidden_size, dtype=torch.float32)
        table[:, 0::2] = torch.sin(angles)
        # An odd hidden size leaves the last sine without its cosine partner.
        table[:, 1::2] = torch.cos(angles[:, : hidden_size // 2])
        return table / table.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def visual_position_embedding_forward(
        self,
        video_feats: torch.Tensor,
        video_length: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Add per-video positional embeddings to flattened visual features.

        Which encoding is added -- a trainable table, nothing at all, or a fixed
        sinusoidal table -- is selected by
        ``config.visual_position_embedding_type``.

        Args:
            video_feats: Concatenated video features with shape ``[BT, D]``.
            video_length: Per-video lengths with shape ``[B]``.
        """
        # New adapters may provide repeated/custom temporal positions;
        # legacy adapters keep the original 0..length-1 behavior.
        if position_ids is None:
            batch_size = video_length.shape[0]
            position_ids = torch.cat(
                [
                    torch.arange(video_length[index], device=video_feats.device)
                    for index in range(batch_size)
                ]
            )
        else:
            position_ids = position_ids.to(device=video_feats.device, dtype=torch.long)
            if position_ids.ndim != 1 or position_ids.numel() != video_feats.shape[0]:
                raise ValueError(
                    "visual position_ids must be 1D and match the number of "
                    f"visual tokens, got {tuple(position_ids.shape)} for "
                    f"{video_feats.shape[0]} tokens"
                )
        # The shape checks above hold for every mode; only a table lookup is
        # bounded by MAX_TOKEN_LENGTH, so "none" stays valid for any length.
        if self.config.visual_position_embedding_type == "none":
            return video_feats  # [BT, D]

        if (
            position_ids.numel()
            and int(position_ids.max().item()) >= self.MAX_TOKEN_LENGTH
        ):
            raise ValueError(
                f"visual position id must be smaller than {self.MAX_TOKEN_LENGTH}"
            )
        if self.config.visual_position_embedding_type == "learned":
            position_embeddings = self.visual_position_embedding(position_ids)
        else:
            position_embeddings = self.visual_position_table[position_ids].to(
                dtype=video_feats.dtype
            )
        return video_feats + position_embeddings  # [BT, D]

    def _apply_llm_embedding_scale(
        self, visual_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Match injected visual embeddings to the LLM token-embedding scale.

        Some LLMs scale token embeddings inside their embedding module. Gemma
        4, for example, multiplies them by ``sqrt(hidden_size)``. Visual
        embeddings bypass that module when injected through ``inputs_embeds``,
        so apply the same scale explicitly. Backends such as Qwen that expose
        no embedding scale retain their original behavior.
        """
        embedding_layer = self.llm.get_input_embeddings()
        embedding_scale = getattr(embedding_layer, "embed_scale", None)
        if embedding_scale is None:
            return visual_embeddings

        if isinstance(embedding_scale, torch.Tensor):
            embedding_scale = embedding_scale.to(
                device=visual_embeddings.device,
                dtype=visual_embeddings.dtype,
            )
        return visual_embeddings * embedding_scale

    def get_visual_feats(
        self,
        video: torch.Tensor,
        video_length: torch.Tensor,
        permute_video_tokens: bool = False,
        return_visual_backbone_extras: bool = False,
    ) -> VisualAdapterOutput:
        """Encode video frames and adapt them to the language-model space.

        Args:
            video: Concatenated video frames with shape ``[BT, C, H, W]``.
            video_length: Per-video frame counts with shape ``[B]``.
        """
        _, _, _, _ = video.shape
        _ = video_length.shape[0]

        visual_backbone_output: VisualBackboneOutput = self.visual_backbone(
            video, video_length
        )  # [BT, CLS+HW+REGISTIRY, C]
        visual_adapter_output: VisualAdapterOutput = self.visual_adapter(
            visual_backbone_output, permute_video_tokens=permute_video_tokens
        )  # [BT,  D]
        if self.visual_semantic_encoder is not None:
            visual_adapter_output = self.visual_semantic_encoder(visual_adapter_output)

        if return_visual_backbone_extras:
            return visual_adapter_output, visual_backbone_output.extras
        return visual_adapter_output

    def prepare_for_casual_lm(
        self,
        text_input_ids: torch.Tensor,  # [B, L] [<pad>, ..., <bos>, .... <start_of_image>, ...]
        video: torch.Tensor,  # [BT, C, H, W]
        video_length: torch.Tensor,  # [B], length of each video in the batch
        permute_video_tokens: Optional[bool] = False,
        return_visual_adapter_extras: bool = False,
        return_visual_backbone_extras: bool = False,
    ):
        batch_size = video_length.shape[0]

        visual_result = self.get_visual_feats(
            video,
            video_length,
            permute_video_tokens=permute_video_tokens,
            return_visual_backbone_extras=return_visual_backbone_extras,
        )
        if return_visual_backbone_extras:
            visual_output, visual_backbone_extras = visual_result
        else:
            visual_output = visual_result
            visual_backbone_extras = None

        visual_feats = visual_output.visual_features
        visual_lengths = visual_output.visual_length
        if visual_lengths is None:
            raise ValueError("video_length is required for prepare_for_casual_lm")
        visual_position_ids = visual_output.position_ids
        if visual_position_ids is None:
            visual_position_ids = torch.cat(
                [
                    torch.arange(length, device=visual_feats.device)
                    for length in visual_lengths
                ]
            )
        visual_feats = self.visual_position_embedding_forward(
            visual_feats,
            visual_lengths,
            visual_position_ids,
        )  # [BT, D]
        # visual_feats = self.visual_output_norm(visual_feats)

        # Keep the pre-scale features/lengths for the CTC head, so its input
        # is unaffected by the LLM-embedding scale factor applied below.
        ctc_visual_features = visual_feats
        ctc_visual_lengths = visual_lengths

        # Scale adapted visual features before injecting them into the LLM.
        visual_feats = visual_feats * self.visual_scale

        _, hidden_size = visual_feats.shape
        visual_feats_by_video = torch.split(
            visual_feats, visual_lengths.tolist(), dim=0
        )

        visual_token_mask = text_input_ids.eq(self.config.video_soft_token_id).long()
        text_visual_lengths = visual_token_mask.sum(dim=1)

        assert (text_visual_lengths == visual_lengths + 2).all(), (
            f"The length of text and video must be the same, but got text_visual_lengths: {text_visual_lengths} and visual_lengths: {visual_lengths}"
        )

        extended_visual_feats = []
        for batch_index in range(batch_size):
            video_positions = visual_token_mask[batch_index].nonzero(as_tuple=True)[0]
            start_video_pos = video_positions[0]
            end_video_pos = video_positions[-1]
            extended_visual_feat = torch.cat(
                [
                    torch.zeros(start_video_pos, hidden_size, device=self.device),
                    self.start_video_embds,
                    visual_feats_by_video[batch_index],
                    self.end_video_embeds,
                    torch.zeros(
                        text_input_ids.shape[1] - end_video_pos - 1,
                        hidden_size,
                        device=self.device,
                    ),
                ]
            )
            extended_visual_feats.append(extended_visual_feat)

        extended_visual_feats = torch.stack(
            extended_visual_feats, dim=0
        ).contiguous()  # [B, L, D]
        extended_visual_feats = self._apply_llm_embedding_scale(
            extended_visual_feats
        )
        text_embeds = self.llm.get_input_embeddings()(text_input_ids).contiguous()
        # Keep multimodal inputs in the LLM embedding dtype (e.g. bf16) after
        # the fp32 position embedding and RMSNorm computation.
        extended_visual_feats = extended_visual_feats.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.where(
            visual_token_mask.bool().unsqueeze(-1),  # [B, L, 1]
            extended_visual_feats,  # [B, L, D]
            text_embeds,  # [B, L, D]
        )

        prepare_output = PrepareForCausalLMOutput(
            input_ids=text_input_ids,  # [B, L]
            inputs_embeds=inputs_embeds,  # [B, L, D]
            visual_mask=visual_token_mask,  # [B, L]
            visual_lengths=visual_lengths,  # [B]
            packed_visual_position_ids=visual_position_ids,
            ctc_visual_features=ctc_visual_features,  # [sum(Lv), D]
            ctc_visual_lengths=ctc_visual_lengths,  # [B]
        )
        if return_visual_adapter_extras and return_visual_backbone_extras:
            return prepare_output, visual_output.extras, visual_backbone_extras
        if return_visual_adapter_extras:
            return prepare_output, visual_output.extras
        if return_visual_backbone_extras:
            return prepare_output, visual_backbone_extras
        return prepare_output

    @staticmethod
    def _compute_branch_attention_diversity_loss(
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize overlap between queries within one attention branch.

        Args:
            attention_weights: Per-head probabilities shaped ``[G, H, N, S]``.
        """
        if attention_weights.ndim != 4:
            raise ValueError(
                "attention_weights must have shape [G, H, N, S], got "
                f"{tuple(attention_weights.shape)}"
            )
        if attention_weights.shape[0] == 0 or attention_weights.shape[-1] == 0:
            raise ValueError("attention batch and source dimensions must be non-empty")

        num_queries = attention_weights.shape[2]
        if num_queries < 2:
            return attention_weights.sum() * 0.0

        # [G, H, N, S] -> [G, N, S], then normalize each query distribution.
        mean_attention = attention_weights.mean(dim=1)
        normalized_attention = nn.functional.normalize(
            mean_attention, p=2, dim=-1, eps=1e-12
        )
        query_similarity = torch.bmm(
            normalized_attention, normalized_attention.transpose(1, 2)
        )  # [G, N, N]
        diagonal = query_similarity.diagonal(dim1=1, dim2=2).sum()
        off_diagonal = query_similarity.sum() - diagonal
        denominator = attention_weights.shape[0] * num_queries * (num_queries - 1)
        return off_diagonal / denominator

    @classmethod
    def _compute_adapter_attention_diversity_loss(
        cls,
        adapter_extras: Optional[dict],
    ) -> Optional[torch.Tensor]:
        """Sum diversity loss over every attention branch exposed by an adapter."""
        if not adapter_extras:
            return None
        attention_weights = [
            adapter_extras[key]
            for key in (
                "temporal_attention_weights",
                "patch_attention_weights",
            )
            if adapter_extras.get(key) is not None
        ]
        if not attention_weights:
            return None
        if not all(isinstance(weights, torch.Tensor) for weights in attention_weights):
            raise TypeError("adapter attention weights must be torch.Tensor values")

        return sum(
            (
                cls._compute_branch_attention_diversity_loss(weights)
                for weights in attention_weights
            ),
            start=attention_weights[0].new_zeros(()),
        )

    @staticmethod
    def _compute_causal_lm_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean next-token cross entropy over non-ignored labels."""
        if logits.ndim != 3:
            raise ValueError(
                f"logits must have shape [B, L, V], got {tuple(logits.shape)}"
            )
        if labels.shape != logits.shape[:2]:
            raise ValueError(
                f"labels shape {tuple(labels.shape)} must match logits batch and "
                f"sequence dimensions {tuple(logits.shape[:2])}"
            )
        if logits.size(1) < 2:
            raise ValueError("causal language-model loss requires sequence length >= 2")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _compute_ctc_loss(
        self,
        ctc_visual_features: torch.Tensor,  # [sum(Lv), D]
        ctc_visual_lengths: torch.Tensor,  # [B]
        pseudo_gloss_ids: torch.Tensor,  # [sum(pseudo_gloss_length)]
        pseudo_gloss_length: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Compute the CTC loss between visual tokens and packed pseudo-gloss targets."""
        logits = self.ctc_head(ctc_visual_features)  # [sum(Lv), V]
        # Compute in fp32 for numerical stability under mixed-precision training.
        log_probs = nn.functional.log_softmax(logits.float(), dim=-1)
        padded_log_probs, _ = packed_to_padded(
            log_probs, ctc_visual_lengths
        )  # [B, T, V]
        log_probs = padded_log_probs.transpose(0, 1)  # [T, B, V], required by ctc_loss
        return nn.functional.ctc_loss(
            log_probs,
            pseudo_gloss_ids,
            ctc_visual_lengths,
            pseudo_gloss_length,
            blank=self.config.ctc_blank_id,
            reduction="mean",
            zero_infinity=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L] [<pad>, ..., <bos>, .... <video_soft_token>, ...]
        pixel_values: Optional[torch.Tensor] = None,  # [BT, C, H, W]
        pixel_values_length: Optional[
            torch.Tensor
        ] = None,  # [B], length of each video in the batch
        attention_mask: Optional[torch.Tensor] = None,  # [B, L]
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,  # [B, L]
        labels: Optional[torch.Tensor] = None,  # [B, L]
        # Accepted temporarily so existing processor batches remain compatible
        # after removal of the old contrastive/alignment objectives.
        # Packed (no padding) pseudo-gloss token ids plus per-sample lengths,
        # e.g. for a future CTC loss.
        pseudo_gloss_ids: Optional[torch.Tensor] = None,  # [sum(pseudo_gloss_length)]
        pseudo_gloss_length: Optional[torch.Tensor] = None,  # [B]
        semantic_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        # ------------ NOTE: special kwars for experimental features ------------
        permute_video_tokens: Optional[bool] = False,
        information_request: Optional[InformationRequest] = None,
        **llm_forward_kwargs: dict,
    ):
        if information_request is None:
            information_request = InformationRequest()
        elif not isinstance(information_request, InformationRequest):
            raise TypeError("information_request must be an InformationRequest")

        if information_request.enabled:
            logger.warning_once(
                "InformationRequest is enabled for this model. Extracting LLM "
                "attention tensors may "
                "increase computation and memory usage and reduce model throughput."
            )

        return_full_llm_attentions = bool(
            llm_forward_kwargs.get(
                "output_attentions",
                getattr(self.llm.config, "output_attentions", False),
            )
        )
        if information_request.llm_attentions:
            llm_forward_kwargs["output_attentions"] = True

        # Without explicit lengths, pixel_values must represent one video.
        if pixel_values_length is None and pixel_values is not None:
            assert input_ids.shape[0] == 1, (
                "When pixel_values_length is not provided, input_ids batch size must be 1."
            )
            pixel_values_length = torch.tensor(
                [pixel_values.shape[0]], dtype=torch.long, device=pixel_values.device
            )

        # Each length must align with the configured temporal downsampling ratio.
        # WARN: Divisibility is only meaningful for temporal downsampling.
        # V2 can emit multiple tokens per input frame (video_token_scale > 1).
        if pixel_values_length is not None and self.config.video_token_scale <= 1.0:
            assert (
                pixel_values_length % int(1.0 / self.config.video_token_scale) == 0
            ).all(), (
                "The length of pixel_values_length must be a multiple of (1/video_token_scale)."
            )

        past_key_values: Cache | None = llm_forward_kwargs.pop("past_key_values", None)
        inputs_embeds = llm_forward_kwargs.pop("inputs_embeds", None)

        prepare_output = None
        visual_adapter_extras = None
        visual_backbone_extras = None
        if inputs_embeds is None:
            if pixel_values is not None:
                visual_prepare_result = self.prepare_for_casual_lm(
                    input_ids,
                    pixel_values,
                    pixel_values_length,
                    permute_video_tokens=permute_video_tokens,
                    return_visual_adapter_extras=True,
                    return_visual_backbone_extras=(
                        information_request.visual_backbone_extras
                    ),
                )
                if information_request.visual_backbone_extras:
                    (
                        prepare_output,
                        visual_adapter_extras,
                        visual_backbone_extras,
                    ) = visual_prepare_result
                else:
                    prepare_output, visual_adapter_extras = visual_prepare_result
                inputs_embeds = prepare_output.inputs_embeds
            else:
                assert input_ids.shape[1] == 1, (
                    "When inputs_embeds is None, input_ids sequence length must be 1."
                )
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.llm.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Generation may have already converted the attention mask into a mapping.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.text_config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if (
                hasattr(self.llm.config, "use_bidirectional_attention")
                and self.llm.config.use_bidirectional_attention
            ):
                logger.warn(
                    "The LLM is configured to use bidirectional, which is not fully supported by our current implementation. The causal mask will be disabled, but the model may still not work as expected."
                )

            # This heuristic cannot identify an eagerly initialized cache
            # (for example, compiled prefill) without inspecting data values,
            # which would not be compile-compatible.
            is_prefill = (
                not use_cache
                or past_key_values is None
                or not past_key_values.is_initialized
                or pixel_values is not None
            )

            # Allow bidirectional attention between video tokens during prefill.
            # Decoding steps need no overlay: the single new query token only
            # attends to the past, and the video block was already resolved
            # bidirectionally when the cache was filled.
            wants_bidirectional = (
                token_type_ids is not None
                and is_prefill
                and self.config.video_bidirectional_attention
            )
            if wants_bidirectional:
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device),
                )
                if self.has_sliding_layers:
                    sliding_mask_kwargs["or_mask_function"] = (
                        token_type_ids_mask_function(
                            token_type_ids.to(cache_position.device),
                        )
                    )

            full_attention_mask = create_causal_mask(**mask_kwargs)
            if wants_bidirectional:
                self._validate_bidirectional_mask(full_attention_mask, token_type_ids)
            causal_mask_mapping = {"full_attention": full_attention_mask}
            if self.has_sliding_layers:
                sliding_attention_mask = create_sliding_window_causal_mask(
                    **sliding_mask_kwargs,
                )
                if wants_bidirectional:
                    self._validate_bidirectional_mask(
                        sliding_attention_mask, token_type_ids
                    )
                causal_mask_mapping["sliding_attention"] = sliding_attention_mask

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **llm_forward_kwargs,
        )

        ce_loss = (
            self._compute_causal_lm_loss(outputs.logits, labels)
            if labels is not None
            else None
        )
        loss = ce_loss

        attention_diversity_loss = None
        if (
            loss is not None
            and self.training
            and self.config.attention_diversity_loss_weight > 0.0
        ):
            attention_diversity_loss = self._compute_adapter_attention_diversity_loss(
                visual_adapter_extras
            )
            if attention_diversity_loss is not None:
                loss = (
                    loss
                    + self.config.attention_diversity_loss_weight
                    * attention_diversity_loss
                )

        ctc_loss = None
        if (
            loss is not None
            and self.training
            and self.config.ctc_enabled
            and self.config.ctc_loss_weight > 0.0
        ):
            if (
                pseudo_gloss_ids is None
                or pseudo_gloss_length is None
                or prepare_output is None
            ):
                raise ValueError(
                    "ctc_enabled with ctc_loss_weight > 0 requires pseudo_gloss_ids, "
                    "pseudo_gloss_length, and pixel_values to be provided."
                )
            ctc_loss = self._compute_ctc_loss(
                prepare_output.ctc_visual_features,
                prepare_output.ctc_visual_lengths,
                pseudo_gloss_ids,
                pseudo_gloss_length,
            )
            loss = loss + self.config.ctc_loss_weight * ctc_loss

        information = None
        if information_request.enabled:
            information = build_information_output(
                request=information_request,
                batch_size=input_ids.shape[0],
                llm_attentions=outputs.attentions,
                prepare_output=prepare_output,
                visual_backbone_extras=visual_backbone_extras,
            )

        logging_scalars = (
            {
                # Final objective: CE + attention diversity (+ CTC).
                "main_loss": loss.detach(),
                "ce_loss": ce_loss.detach(),
                # CE has no configurable weight (implicitly 1.0), unlike the
                # attention-diversity/CTC terms below, so this equals ce_loss.
                # Logged anyway for parity with those terms' raw/weighted pairs.
                "ce_weighted_loss": ce_loss.detach(),
            }
            if loss is not None
            else None
        )
        if attention_diversity_loss is not None:
            weighted_attention_diversity_loss = (
                self.config.attention_diversity_loss_weight * attention_diversity_loss
            )
            logging_scalars.update(
                {
                    "attention_diversity_loss": attention_diversity_loss.detach(),
                    "attention_diversity_weighted_loss": (
                        weighted_attention_diversity_loss.detach()
                    ),
                    "attention_diversity_loss_weight": loss.new_tensor(
                        self.config.attention_diversity_loss_weight
                    ),
                }
            )
        if ctc_loss is not None:
            weighted_ctc_loss = self.config.ctc_loss_weight * ctc_loss
            logging_scalars.update(
                {
                    "ctc_loss": ctc_loss.detach(),
                    "ctc_weighted_loss": weighted_ctc_loss.detach(),
                    "ctc_loss_weight": loss.new_tensor(self.config.ctc_loss_weight),
                }
            )

        return SltCausalLMOutputWithPast(
            loss=loss,
            logging_scalars=logging_scalars,
            information=information,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if return_full_llm_attentions else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        pixel_values_length=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Extend the standard generation inputs with video tensors during prefill.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Cached decoding no longer contains video placeholder tokens, so video
        # tensors are only passed on the first generation step.
        if is_first_iteration:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_length"] = pixel_values_length

        return model_inputs

    @property
    def dummy_inputs(self):
        num_frames = 32 * int(1.0 / self.config.video_token_scale)
        video_token = self.config.video_soft_token_id
        num_video_tokens = int(self.config.video_token_scale * num_frames) + 2

        # fmt: off
        input_ids= torch.tensor(
            [[ 0, 0, 0, 1, 2, 3,] + [video_token] * num_video_tokens + [ 4, 5, 9, 7, ]],
            dtype=torch.long,
            device=self.device,
        )
        seq_len = input_ids.shape[1]
        return {
            "input_ids": input_ids,
            "pixel_values": torch.ones(
                (num_frames, 3, 224, 224), dtype=torch.float32, device=self.device
            ),
            "pixel_values_length": torch.tensor(
                [num_frames], dtype=torch.long, device=self.device
            ),
            "attention_mask": torch.ones(1, seq_len , dtype=torch.long, device=self.device),
            "labels": torch.ones( 1, seq_len, dtype=torch.long, device=self.device),
            "position_ids": torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0),
        }

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm
