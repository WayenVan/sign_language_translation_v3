import math
from copy import deepcopy
from enum import Enum
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
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
from peft import LoraConfig, inject_adapter_in_model
from ..configuration_slt.configuration import SltConfig
from .ctc_codebook import CTCCodebookBridge
from .output_utils import (
    CTCEncoderOutput,
    PrepareForCausalLMOutput,
    SltCausalLMOutputWithPast,
    SltCTCOutput,
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
)

logger = logging.get_logger(__name__)

ForwardMode = Literal["ctc_only", "joint"]
_FORWARD_MODES = frozenset({"ctc_only", "joint"})


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


def _prefixed_logging_scalars(
    scalars: Optional[dict], prefix: str
) -> dict[str, torch.Tensor]:
    """Validate a component's logging scalars and namespace them.

    Components emit bare names so they need to know nothing about the model's
    namespace; the prefix is applied here, where collisions with ``ce_loss`` and
    friends would otherwise happen. Values must already be detached
    single-element tensors: the trainer averages them across steps, and a live
    graph or a full feature map here would hold memory for a whole logging
    interval before anything complained.
    """
    if not scalars:
        return {}
    prefixed = {}
    for name, value in scalars.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError(
                f"{prefix}/{name} must be a single-element tensor, got "
                f"{type(value).__name__}"
            )
        if value.requires_grad:
            raise ValueError(f"{prefix}/{name} must be detached before logging")
        prefixed[f"{prefix}/{name}"] = value
    return prefixed


def _ctc_head_blank_frequency_scalars(
    ctc_logits: torch.Tensor, blank_id: int
) -> dict[str, torch.Tensor]:
    """Blank-frequency diagnostics computed straight from CTC head logits.

    A pure function of the classifier's own output -- it needs no codebook
    state, so ``ctc_only`` forwards (which never build a codebook
    distribution at all) and joint training both call this directly and
    report the same numbers on the same footing. Deliberately independent of
    the codebook's selection mode/temperature: a plain temperature-1 softmax
    here keeps the reading from drifting with whatever temperature or Gumbel
    noise the embedding path happens to use that step.
    """
    if ctc_logits.shape[0] == 0:
        zero = ctc_logits.new_zeros((), dtype=torch.float32)
        return {"blank_probability_mean": zero, "blank_argmax_ratio": zero}
    probabilities = F.softmax(ctc_logits.float(), dim=-1)
    blank_probability_mean = probabilities[:, blank_id].mean()
    predicted_ids = ctc_logits.argmax(dim=-1)
    blank_argmax_ratio = predicted_ids.eq(blank_id).float().mean()
    return {
        "blank_probability_mean": blank_probability_mean.detach(),
        "blank_argmax_ratio": blank_argmax_ratio.detach(),
    }


def _load_pretrained_submodule_components(model: nn.Module) -> None:
    """Let submodules pull in weights that live outside this checkpoint.

    Some components are built empty by ``__init__`` and filled from their own
    pretrained source -- the hand-patch scorer, whose coefficients were fitted
    offline. Loading them in ``__init__`` instead would make every
    ``from_pretrained`` read weights it is about to overwrite, and would tie a
    finished checkpoint to a directory that may no longer exist. So this factory,
    the one path that does load from external sources, drives it.

    Duck-typed: any module exposing ``load_pretrained_components()`` is called.
    Define the hook on whichever module owns the loading and not also on its
    parent, or the same weights are read twice.
    """
    for name, module in model.named_modules():
        loader = getattr(module, "load_pretrained_components", None)
        if callable(loader):
            loader()
            mark_module_tree_as_initialized(module)
            logger.info("Loaded pretrained components for %s", name or "<root>")


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
        # Training-time policy owns this value. Start from deterministic eval
        # so standalone construction is safe before a plan is applied.
        self.llm_runtime_mode = "eval"
        self.visual_adapter_runtime_mode = "eval"

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

        validate_llm_lora_config_presence(
            enabled=config.llm_lora,
            config=config.llm_lora_config,
        )
        if config.llm_lora:
            self._inject_llm_lora(LoraConfig(**config.llm_lora_config))

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
            nn.Embedding(self.MAX_TOKEN_LENGTH, self.config.ctc_hidden_size)
            if position_embedding_type == "learned"
            else None
        )
        # Non-persistent: the table is a pure function of MAX_TOKEN_LENGTH and
        # ctc_hidden_size, so serializing it would only add weight to every
        # checkpoint and let a stale copy override the formula on reload.
        self.register_buffer(
            "visual_position_table",
            self._build_sincos_position_table(
                self.MAX_TOKEN_LENGTH, self.config.ctc_hidden_size
            )
            if position_embedding_type == "sincos"
            else None,
            persistent=False,
        )
        # CTC is the mandatory discrete interface between visual tokens and
        # the language model. Its classifier and semantic codebook stay
        # separate because they optimize different geometry.
        self.ctc_head = nn.Linear(
            self.config.ctc_hidden_size,
            config.ctc_vocab_size,
        )
        self.ctc_codebook = CTCCodebookBridge(
            ctc_vocab_size=config.ctc_vocab_size,
            llm_hidden_size=self.llm.get_input_embeddings().embedding_dim,
            blank_id=config.ctc_blank_id,
            training_mode=config.ctc_codebook_training_mode,
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

        # ``post_init`` initializes parameters but does not own module runtime
        # modes. Keep a frozen/default LLM deterministic until the engine's
        # explicit trainability plan selects otherwise.
        self.set_llm_runtime_mode("eval")
        self.set_visual_adapter_runtime_mode("eval")

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

    def train(self, mode: bool = True):
        """Apply engine-selected LLM and visual-adapter runtime modes."""
        super().train(mode)
        # Whole-model eval always wins: `follow` is permission to enter
        # training mode with the model, never a mode of its own. During
        # training this stays independent of which base or LoRA parameters
        # receive gradients.
        self.llm.train(mode and self.llm_runtime_mode == "follow")
        visual_adapter = getattr(self, "visual_adapter", None)
        if isinstance(visual_adapter, nn.Module):
            visual_adapter.train(
                mode and self.visual_adapter_runtime_mode == "follow"
            )
        return self

    def set_llm_runtime_mode(self, runtime_mode: str) -> None:
        """Set Qwen train/eval behavior without changing requires_grad."""
        if runtime_mode not in ("eval", "follow"):
            raise ValueError(
                f"LLM runtime_mode must be 'eval' or 'follow', got {runtime_mode!r}"
            )
        self.llm_runtime_mode = runtime_mode
        self.llm.train(self.training and runtime_mode == "follow")

    def set_visual_adapter_runtime_mode(self, runtime_mode: str) -> None:
        """Set visual-adapter train/eval behavior without changing gradients."""
        if runtime_mode not in ("eval", "follow"):
            raise ValueError(
                "Visual adapter runtime_mode must be 'eval' or 'follow', got "
                f"{runtime_mode!r}"
            )
        self.visual_adapter_runtime_mode = runtime_mode
        self.visual_adapter.train(self.training and runtime_mode == "follow")

    @torch.no_grad()
    def initialize_ctc_codebook(
        self,
        llm_token_ids_by_ctc_id: Sequence[Sequence[int]],
        *,
        blank_init_token_id: int,
    ) -> None:
        """Initialize the CTC codebook once from this model's LLM embeddings.

        Tokenizers stay outside the model boundary: the construction workflow
        resolves each CTC token to LLM sub-token ids and passes only those ids
        here. The initialized weights and initialization marker are then saved
        by the ordinary Hugging Face checkpoint path.
        """
        self.ctc_codebook.initialize_from_llm_embeddings(
            llm_embeddings=self.llm.get_input_embeddings(),
            llm_token_ids_by_ctc_id=llm_token_ids_by_ctc_id,
            llm_pad_token_id=blank_init_token_id,
        )

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

    def _inject_llm_lora(self, peft_config: LoraConfig) -> None:
        """Inject LoRA into the native LLM without wrapping its top level."""
        injected_llm = inject_adapter_in_model(
            peft_config=peft_config,
            model=self.llm,
        )
        if injected_llm is not self.llm:
            raise RuntimeError("PEFT replaced the native LLM during in-place injection")
        # PEFT initializes the new adapter parameters. Protect them from the
        # outer SltModel.post_init(), which runs after checkpoint topology has
        # been reconstructed.
        mark_module_tree_as_initialized(self.llm)

    def inject_llm_lora(self, peft_config: LoraConfig) -> None:
        """Inject a new LoRA adapter into the language model."""
        if self.config.llm_lora:
            raise ValueError(
                "The checkpoint already contains LLM LoRA. "
                "Use SltModel.from_pretrained() to load it."
            )
        self._inject_llm_lora(peft_config)
        self.config.llm_lora = True
        self.config.llm_lora_config = _serialize_lora_config(peft_config)

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

        llm_cls = get_llm_cls_by_model_name(config.llm_model_name_or_path)
        llm = llm_cls.from_pretrained(config.llm_model_name_or_path, dtype=llm_dtype)

        # This factory explicitly loaded both components from pretrained
        # checkpoints. Protect them from the outer SltModel.post_init(); the
        # ordinary constructor makes no such assumption about supplied modules.
        mark_module_tree_as_initialized(llm)
        mark_module_tree_as_initialized(visual_backbone)

        model = cls(
            config=config,
            llm=llm,
            visual_backbone=visual_backbone,
        )
        _load_pretrained_submodule_components(model)
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

        backbone_kwargs = {}
        if getattr(
            self.visual_adapter, "requires_visual_backbone_attention", False
        ):
            backbone_kwargs["return_attention_maps"] = True
        visual_backbone_output: VisualBackboneOutput = self.visual_backbone(
            video, video_length, **backbone_kwargs
        )  # [BT, CLS+HW+REGISTIRY, C]
        visual_adapter_output: VisualAdapterOutput = self.visual_adapter(
            visual_backbone_output, permute_video_tokens=permute_video_tokens
        )  # [BT,  D]

        if return_visual_backbone_extras:
            return visual_adapter_output, visual_backbone_output.extras
        return visual_adapter_output

    def _encode_ctc(
        self,
        video: torch.Tensor,
        video_length: torch.Tensor,
        *,
        permute_video_tokens: bool = False,
        return_visual_backbone_extras: bool = False,
    ) -> CTCEncoderOutput:
        """Run the one shared video-to-CTC path for every forward mode."""
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

        visual_features = visual_output.visual_features
        visual_lengths = visual_output.visual_length
        if visual_lengths is None:
            raise ValueError("visual adapter output must include visual_length")
        position_ids = visual_output.position_ids
        if position_ids is None:
            position_ids = torch.cat(
                [
                    torch.arange(length, device=visual_features.device)
                    for length in visual_lengths
                ]
            )
        visual_features = self.visual_position_embedding_forward(
            visual_features,
            visual_lengths,
            position_ids,
        )
        return CTCEncoderOutput(
            logits=self.ctc_head(visual_features),
            lengths=visual_lengths,
            packed_position_ids=position_ids,
            visual_adapter_logging_scalars=visual_output.logging_scalars,
            visual_backbone_extras=visual_backbone_extras,
        )

    def prepare_for_casual_lm(
        self,
        text_input_ids: torch.Tensor,  # [B, L] [<pad>, ..., <bos>, .... <start_of_image>, ...]
        video: torch.Tensor,  # [BT, C, H, W]
        video_length: torch.Tensor,  # [B], length of each video in the batch
        permute_video_tokens: Optional[bool] = False,
        return_visual_backbone_extras: bool = False,
        ctc_codebook_temperature: Optional[float] = None,
    ):
        batch_size = video_length.shape[0]

        ctc_output = self._encode_ctc(
            video,
            video_length,
            permute_video_tokens=permute_video_tokens,
            return_visual_backbone_extras=return_visual_backbone_extras,
        )
        temperature = (
            self.config.ctc_codebook_default_temperature
            if ctc_codebook_temperature is None
            else ctc_codebook_temperature
        )
        codebook_output = self.ctc_codebook(
            ctc_output.logits,
            ctc_output.lengths,
            temperature=temperature,
        )
        visual_feats = codebook_output.embeddings
        visual_lengths = ctc_output.lengths

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
            packed_visual_position_ids=ctc_output.packed_position_ids,
            ctc_logits=ctc_output.logits,
            ctc_lengths=visual_lengths,
            ctc_codebook_logging_scalars=codebook_output.logging_scalars,
            visual_adapter_logging_scalars=(
                ctc_output.visual_adapter_logging_scalars
            ),
        )
        if return_visual_backbone_extras:
            return prepare_output, ctc_output.visual_backbone_extras
        return prepare_output

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
        ctc_logits: torch.Tensor,  # [sum(Lv), V]
        ctc_lengths: torch.Tensor,  # [B]
        pseudo_gloss_ids: torch.Tensor,  # [sum(pseudo_gloss_length)]
        pseudo_gloss_length: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Compute CTC loss from the logits already consumed by the codebook."""
        # Compute in fp32 for numerical stability under mixed-precision training.
        log_probs = nn.functional.log_softmax(ctc_logits.float(), dim=-1)
        padded_log_probs, _ = packed_to_padded(
            log_probs, ctc_lengths
        )  # [B, T, V]
        log_probs = padded_log_probs.transpose(0, 1)  # [T, B, V], required by ctc_loss
        return nn.functional.ctc_loss(
            log_probs,
            pseudo_gloss_ids,
            ctc_lengths,
            pseudo_gloss_length,
            blank=self.config.ctc_blank_id,
            reduction="mean",
            zero_infinity=True,
        )

    def _forward_ctc_only(
        self,
        *,
        pixel_values: torch.Tensor,
        pixel_values_length: torch.Tensor,
        pseudo_gloss_ids: Optional[torch.Tensor],
        pseudo_gloss_length: Optional[torch.Tensor],
        permute_video_tokens: bool,
    ) -> SltCTCOutput:
        """Run Phase-A CTC without constructing codebook or LLM inputs."""
        if (pseudo_gloss_ids is None) != (pseudo_gloss_length is None):
            raise ValueError(
                "ctc_only requires pseudo_gloss_ids and pseudo_gloss_length "
                "to be provided together"
            )
        ctc_output = self._encode_ctc(
            pixel_values,
            pixel_values_length,
            permute_video_tokens=permute_video_tokens,
        )
        ctc_loss = (
            self._compute_ctc_loss(
                ctc_output.logits,
                ctc_output.lengths,
                pseudo_gloss_ids,
                pseudo_gloss_length,
            )
            if pseudo_gloss_ids is not None
            else None
        )
        logging_scalars = _prefixed_logging_scalars(
            ctc_output.visual_adapter_logging_scalars,
            "visual_adapter",
        )
        # Blank-frequency health check, computable straight from ctc_head's
        # own output. ctc_only never builds a codebook distribution, but
        # this doesn't need one -- see _ctc_head_blank_frequency_scalars.
        logging_scalars.update(
            _prefixed_logging_scalars(
                _ctc_head_blank_frequency_scalars(
                    ctc_output.logits, self.config.ctc_blank_id
                ),
                "ctc_head",
            )
        )
        if ctc_loss is not None:
            logging_scalars.update(
                {
                    "main_loss": ctc_loss.detach(),
                    "ctc_loss": ctc_loss.detach(),
                }
            )
        return SltCTCOutput(
            loss=ctc_loss,
            logits=ctc_output.logits,
            lengths=ctc_output.lengths,
            logging_scalars=logging_scalars or None,
        )

    @staticmethod
    def _validate_forward_mode(forward_mode: str) -> None:
        if forward_mode not in _FORWARD_MODES:
            raise ValueError(
                f"forward_mode must be one of {sorted(_FORWARD_MODES)}, "
                f"got {forward_mode!r}"
            )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # [B, L]
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
        ctc_codebook_temperature: Optional[float] = None,
        forward_mode: ForwardMode = "joint",
        information_request: Optional[InformationRequest] = None,
        **llm_forward_kwargs: dict,
    ):
        self._validate_forward_mode(forward_mode)
        # Without explicit lengths, pixel_values must represent one video.
        if pixel_values_length is None and pixel_values is not None:
            assert input_ids is None or input_ids.shape[0] == 1, (
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

        if forward_mode == "ctc_only":
            if pixel_values is None or pixel_values_length is None:
                raise ValueError(
                    "ctc_only requires pixel_values and pixel_values_length"
                )
            return self._forward_ctc_only(
                pixel_values=pixel_values,
                pixel_values_length=pixel_values_length,
                pseudo_gloss_ids=pseudo_gloss_ids,
                pseudo_gloss_length=pseudo_gloss_length,
                permute_video_tokens=bool(permute_video_tokens),
            )

        if input_ids is None:
            raise ValueError("joint forward requires input_ids")

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

        past_key_values: Cache | None = llm_forward_kwargs.pop("past_key_values", None)
        inputs_embeds = llm_forward_kwargs.pop("inputs_embeds", None)

        prepare_output = None
        visual_backbone_extras = None
        if inputs_embeds is None:
            if pixel_values is not None:
                visual_prepare_result = self.prepare_for_casual_lm(
                    input_ids,
                    pixel_values,
                    pixel_values_length,
                    permute_video_tokens=permute_video_tokens,
                    return_visual_backbone_extras=(
                        information_request.visual_backbone_extras
                    ),
                    ctc_codebook_temperature=ctc_codebook_temperature,
                )
                if information_request.visual_backbone_extras:
                    prepare_output, visual_backbone_extras = visual_prepare_result
                else:
                    prepare_output = visual_prepare_result
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

        ctc_loss = None
        if (
            loss is not None
            and self.training
            and self.config.ctc_loss_weight > 0.0
        ):
            if (
                pseudo_gloss_ids is None
                or pseudo_gloss_length is None
                or prepare_output is None
            ):
                raise ValueError(
                    "ctc_loss_weight > 0 requires pseudo_gloss_ids, "
                    "pseudo_gloss_length, and pixel_values to be provided."
                )
            ctc_loss = self._compute_ctc_loss(
                prepare_output.ctc_logits,
                prepare_output.ctc_lengths,
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
                # Final objective: CE (+ CTC).
                "main_loss": loss.detach(),
                "ce_loss": ce_loss.detach(),
                # CE has no configurable weight (implicitly 1.0), unlike the
                # CTC term below, so this equals ce_loss. Logged anyway for
                # parity with that term's raw/weighted pair.
                "ce_weighted_loss": ce_loss.detach(),
            }
            if loss is not None
            else None
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
        if logging_scalars is not None and prepare_output is not None:
            logging_scalars.update(
                _prefixed_logging_scalars(
                    prepare_output.ctc_codebook_logging_scalars, "ctc_codebook"
                )
            )
            logging_scalars.update(
                _prefixed_logging_scalars(
                    _ctc_head_blank_frequency_scalars(
                        prepare_output.ctc_logits, self.config.ctc_blank_id
                    ),
                    "ctc_head",
                )
            )
            logging_scalars.update(
                _prefixed_logging_scalars(
                    prepare_output.visual_adapter_logging_scalars, "visual_adapter"
                )
            )

        return SltCausalLMOutputWithPast(
            loss=loss,
            logging_scalars=logging_scalars,
            information=information,
            ctc_logits=(
                prepare_output.ctc_logits if prepare_output is not None else None
            ),
            ctc_lengths=(
                prepare_output.ctc_lengths if prepare_output is not None else None
            ),
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
        ctc_codebook_temperature=None,
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
            model_inputs["ctc_codebook_temperature"] = ctc_codebook_temperature

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


# ---------------------------------------------------------------------------
# LLM LoRA configuration validation
#
# Keep checkpoint/configuration policy out of SltModel's execution path. The
# model calls these helpers only at topology boundaries: construction and a
# request to resume or add LoRA.
# ---------------------------------------------------------------------------

_LORA_CONFIG_METADATA_KEYS = {
    "auto_mapping",
    "base_model_name_or_path",
    "peft_version",
    "revision",
}


def validate_llm_lora_config_presence(*, enabled: bool, config: dict) -> None:
    """Require the LoRA presence flag and reconstruction config to agree."""
    if not isinstance(enabled, bool):
        raise TypeError("llm_lora must be a bool")
    if not isinstance(config, dict):
        raise TypeError("llm_lora_config must be a dict")
    if enabled and not config:
        raise ValueError("llm_lora_config must be provided when llm_lora is True")
    if not enabled and config:
        raise ValueError("llm_lora_config must be empty when llm_lora is False")


def validate_requested_llm_lora_config(
    checkpoint_config: dict,
    requested_config: LoraConfig,
) -> None:
    """Reject a resume request that differs from checkpoint LoRA topology."""
    if not isinstance(checkpoint_config, dict) or not checkpoint_config:
        raise ValueError("checkpoint does not contain an LLM LoRA configuration")
    if not isinstance(requested_config, LoraConfig):
        raise TypeError("requested_config must be a LoraConfig")

    checkpoint = _canonical_lora_config(checkpoint_config)
    requested = _canonical_lora_config(requested_config)
    if checkpoint == requested:
        return

    differing_fields = sorted(
        key
        for key in checkpoint.keys() | requested.keys()
        if checkpoint.get(key) != requested.get(key)
    )
    differences = ", ".join(
        f"{key}: checkpoint={checkpoint.get(key)!r}, "
        f"requested={requested.get(key)!r}"
        for key in differing_fields
    )
    raise ValueError(
        "Requested LLM LoRA config does not match the checkpoint config"
        + (f" ({differences})" if differences else "")
    )


def _canonical_lora_config(config: dict | LoraConfig) -> dict:
    """Fill PEFT defaults and normalize containers before config comparison."""
    lora_config = config if isinstance(config, LoraConfig) else LoraConfig(**config)
    values = {
        key: value
        for key, value in lora_config.to_dict().items()
        if key not in _LORA_CONFIG_METADATA_KEYS
    }
    return _canonicalize_lora_value(values)


def _canonicalize_lora_value(value):
    """Convert PEFT enums and unordered containers to stable Python values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            key: _canonicalize_lora_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize_lora_value(item) for item in value]
        return sorted(normalized, key=repr)
    if isinstance(value, (list, tuple)):
        return [_canonicalize_lora_value(item) for item in value]
    return value
