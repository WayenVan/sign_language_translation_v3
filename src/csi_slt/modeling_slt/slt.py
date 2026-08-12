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

from csi_slt.modeling_slt.cross_modal_contrastive_loss import CrossModalContrastiveLoss

from peft import get_peft_model, LoraConfig
from ..configuration_slt.configuration import SltConfig
from .output_utils import (
    PrepareForCausalLMOutput,
    SltCausalLMOutputWithPast,
    VisualAdapterOutput,
    VisualBackboneOutput,
)
from .misc import mark_module_tree_as_initialized
from .registry import VISUAL_ADAPTERS, VISUAL_BACKBONES

logger = logging.get_logger(__name__)


def get_llm_cls_by_model_name(model_name):
    """Return the supported causal language-model class for ``model_name``."""
    if "qwen" in model_name.lower():
        from transformers.models.qwen3 import Qwen3ForCausalLM

        model_cls = Qwen3ForCausalLM
    elif "gemma" in model_name.lower():
        from transformers.models.gemma3 import (
            Gemma3ForCausalLM,
            Gemma3ForConditionalGeneration,
        )

        model_cls = (
            Gemma3ForCausalLM
            if "1b" in model_name.lower()
            else Gemma3ForConditionalGeneration
        )
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

        self._configure_generation()

        # freeze the visual backbone and language model parameters by default; only the
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
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
        self.visual_position_embedding = nn.Embedding(
            self.MAX_TOKEN_LENGTH, self.config.hidden_size
        )
        # 全局可学习标量，初始不改变特征
        self.visual_scale = nn.Parameter(torch.tensor(1.0))
        # The adapter projection and learned visual positions can have a
        # different scale from the frozen LLM's token embeddings. Normalize the
        # completed visual token (projection + position) immediately before it
        # is merged into the LLM input sequence.
        # self.visual_output_norm = nn.RMSNorm(self.config.hidden_size, eps=1e-6)
        # Keep one registered instance so its learnable temperature is included
        # in model parameters, checkpoints, and the optimizer.
        self.contrastive_loss_fct = CrossModalContrastiveLoss(
            gather_with_grad=True,
            text_queue_size=config.contrastive_text_queue_size,
        )

        self.config.num_extra_tokens = 2  # Start and end video tokens.
        self.config.is_encoder_decoder = False
        self.config.is_decoder = True

        self.post_init()

        # NOTE: tie the weights when using LoRA initialization
        if config.llm_lora:
            self._register_llm_tied_weights()

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

    @classmethod
    def from_pretrained_components(
        cls, config: SltConfig, llm_dtype="auto", visual_backbone_dtype="auto"
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
        # Ensure lm_head and input embeddings are tied even when the source model
        # did not tie them.
        llm.tie_weights(recompute_mapping=True)

        # This factory explicitly loaded both components from pretrained
        # checkpoints. Protect them from the outer SltModel.post_init(); the
        # ordinary constructor makes no such assumption about supplied modules.
        mark_module_tree_as_initialized(llm)
        mark_module_tree_as_initialized(visual_backbone)

        logger.info("force retie the lm_head to the input embeddings!!!!!!!!")

        return cls(config=config, llm=llm, visual_backbone=visual_backbone)

        # fmt: on

    @classmethod
    def from_pretrained_with_new_lora(
        cls, peft_config, checkpoint_dir: str, model_dtype="auto"
    ):
        model: SltModel = cls.from_pretrained(checkpoint_dir, dtype=model_dtype)

        for p in model.llm.parameters():
            p.requires_grad = False
        for p in model.visual_backbone.parameters():
            p.requires_grad = False

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

    def _configure_generation(self):
        text_config = self.config.get_text_config()
        generation_config = self.llm.generation_config
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(text_config)
        else:
            generation_config = deepcopy(generation_config)

        # Token identities come from the canonical text config. Keep a list of
        # EOS ids intact because GenerationConfig supports multiple stop tokens.
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
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

        if module is self.visual_position_embedding:
            nn.init.trunc_normal_(module.weight, std=0.02)

        # Raw parameters are not handled by the generic module-type rules.
        # smart_apply visits children before their parent, so the LLM token
        # embeddings are initialized before this root-model branch runs.
        if module is self:
            mean = self.llm.get_input_embeddings().weight.mean(dim=0, keepdim=True)
            self.start_video_embds.copy_(mean)
            self.end_video_embeds.copy_(mean)

    def visual_position_embedding_forward(
        self,
        video_feats: torch.Tensor,
        video_length: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Add per-video positional embeddings to flattened visual features.

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
        if (
            position_ids.numel()
            and int(position_ids.max().item()) >= self.MAX_TOKEN_LENGTH
        ):
            raise ValueError(
                f"visual position id must be smaller than {self.MAX_TOKEN_LENGTH}"
            )
        position_embeddings = self.visual_position_embedding(position_ids)
        return video_feats + position_embeddings  # [BT, D]

    def get_visual_feats(
        self,
        video: torch.Tensor,
        video_length: torch.Tensor,
        permute_video_tokens: bool = False,
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

        return visual_adapter_output

    def _encode_labels_for_contrastive(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode each label sequence for the contrastive text branch.

        ``labels`` uses ``-100`` for positions excluded from the language-model
        loss. ``embedding_mean`` mean-pools the valid token embeddings, while
        ``decoder_last`` runs the frozen decoder and selects the last valid
        token's contextual hidden state.

        Args:
            labels: Token ids with shape ``[B, T]``. Ignored positions are
                represented by ``-100``.

        Returns:
            Mean-pooled text features with shape ``[B, D]``.
        """
        if labels.ndim != 2:
            raise ValueError(f"labels must have shape [B, T], got {labels.shape}")

        valid_mask = labels.ne(-100)
        if torch.any(valid_mask.sum(dim=-1) == 0):
            raise ValueError("every label sequence must contain a valid token")

        pad_token_id = self.config.get_text_config().pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = labels.masked_fill(~valid_mask, pad_token_id)
        mode = self.config.contrastive_text_encoding_mode
        if mode == "embedding_mean":
            token_features = self.llm.get_input_embeddings()(input_ids)
            weights = valid_mask.unsqueeze(-1).to(dtype=token_features.dtype)
            return (token_features * weights).sum(dim=1) / weights.sum(dim=1)

        attention_mask = valid_mask.long()
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(~valid_mask, 0)

        with torch.no_grad():
            decoder_outputs = self.llm.get_decoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )

        token_positions = torch.arange(labels.shape[1], device=labels.device)
        last_valid_indices = (
            token_positions.masked_fill(~valid_mask, -1).max(dim=-1).values
        )
        batch_indices = torch.arange(labels.shape[0], device=labels.device)
        return decoder_outputs.last_hidden_state[batch_indices, last_valid_indices]

    @staticmethod
    def _encode_visual_features_for_contrastive(
        visual_features: torch.Tensor,
        visual_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool packed visual tokens into one feature per video."""
        if visual_features.ndim != 2:
            raise ValueError(
                "visual_features must have shape [sum(lengths), D], got "
                f"{tuple(visual_features.shape)}"
            )
        if visual_lengths.ndim != 1 or visual_lengths.numel() == 0:
            raise ValueError("visual_lengths must be a non-empty 1D tensor")
        if visual_lengths.is_floating_point() or visual_lengths.is_complex():
            raise TypeError(
                f"visual_lengths must use an integer dtype, got {visual_lengths.dtype}"
            )

        lengths = visual_lengths.to(device=visual_features.device, dtype=torch.long)
        if bool((lengths <= 0).any()):
            raise ValueError("all visual lengths must be positive")
        if int(lengths.sum().item()) != visual_features.shape[0]:
            raise ValueError(
                "visual_lengths.sum() must equal the packed visual token count"
            )

        batch_indices = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=visual_features.device), lengths
        )
        pooled_features = visual_features.new_zeros(
            lengths.numel(), visual_features.shape[-1]
        )
        pooled_features.index_add_(0, batch_indices, visual_features)
        return pooled_features / lengths.to(visual_features.dtype).unsqueeze(-1)

    def prepare_for_casual_lm(
        self,
        text_input_ids: torch.Tensor,  # [B, L] [<pad>, ..., <bos>, .... <start_of_image>, ...]
        video: torch.Tensor,  # [BT, C, H, W]
        video_length: torch.Tensor,  # [B], length of each video in the batch
        permute_video_tokens: Optional[bool] = False,
    ):
        batch_size = video_length.shape[0]

        visual_output = self.get_visual_feats(
            video, video_length, permute_video_tokens=permute_video_tokens
        )

        visual_feats = visual_output.visual_features
        visual_feats = self.visual_position_embedding_forward(
            visual_feats,
            visual_output.visual_length,
            visual_output.position_ids,
        )  # [BT, D]
        # visual_feats = self.visual_output_norm(visual_feats)

        # NOTE: before injuecting into the llm
        contrastive_visual_feats = visual_feats

        # NOTE: scale hte feature
        visual_feats = visual_feats * self.visual_scale

        _, hidden_size = visual_feats.shape
        visual_lengths = visual_output.visual_length

        if visual_lengths is None:
            raise ValueError("video_length is required for prepare_for_casual_lm")

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
        text_embeds = self.llm.get_input_embeddings()(text_input_ids).contiguous()
        # Keep multimodal inputs in the LLM embedding dtype (e.g. bf16) after
        # the fp32 position embedding and RMSNorm computation.
        extended_visual_feats = extended_visual_feats.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.where(
            visual_token_mask.bool().unsqueeze(-1),  # [B, L, 1]
            extended_visual_feats,  # [B, L, D]
            text_embeds,  # [B, L, D]
        )

        return PrepareForCausalLMOutput(
            input_ids=text_input_ids,  # [B, L]
            inputs_embeds=inputs_embeds,  # [B, L, D]
            visual_mask=visual_token_mask,  # [B, L]
            contrastive_features=contrastive_visual_feats,  # [sum(Lv), D], before visual_scale
            contrastive_lengths=visual_lengths,  # [B]
            global_contrastive_features=visual_output.global_visual_features,
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
        pseudo_gloss_input_ids: Optional[torch.Tensor] = None,  # [B, L]
        pseudo_gloss_attention_mask: Optional[torch.Tensor] = None,  # [B, L]
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        # ------------ NOTE: special kwars for experimental features ------------
        permute_video_tokens: Optional[bool] = False,
        **llm_forward_kwargs: dict,
    ):
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

        if inputs_embeds is None:
            if pixel_values is not None:
                prepare_output = self.prepare_for_casual_lm(
                    input_ids,
                    pixel_values,
                    pixel_values_length,
                    permute_video_tokens=permute_video_tokens,
                )
                inputs_embeds = prepare_output.inputs_embeds
            else:
                assert input_ids.shape[1] == 1, (
                    "When inputs_embeds is None, input_ids sequence length must be 1."
                )
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(
                self.llm.config,
            )

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
                "config": self.config.get_text_config(),
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
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
            if token_type_ids is not None and is_prefill:
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device),
                )
                if self.has_sliding_layers:
                    sliding_mask_kwargs["or_mask_function"] = (
                        token_type_ids_mask_function(
                            token_type_ids.to(cache_position.device),
                        )
                    )

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(
                        **sliding_mask_kwargs,
                    )
                )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **llm_forward_kwargs,
        )

        loss = None
        main_loss = None
        contrastive_loss = None
        if labels is not None:
            # Shift so that tokens before position n predict token n.
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss_fct = nn.CrossEntropyLoss()
            main_loss = main_loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Keep this as a tensor so it can be logged consistently even when
            # the contrastive objective is disabled.
            contrastive_loss = torch.zeros_like(main_loss)
            if self.config.contrastive_loss_weight > 0.0:
                if prepare_output.global_contrastive_features is not None:
                    visual_features = prepare_output.global_contrastive_features
                else:
                    visual_features = self._encode_visual_features_for_contrastive(
                        prepare_output.contrastive_features,
                        prepare_output.contrastive_lengths,
                    )
                text_features = self._encode_labels_for_contrastive(labels).detach()
                contrastive_loss = self.contrastive_loss_fct(
                    visual_features=visual_features,
                    text_features=text_features,
                )

            loss = main_loss + self.config.contrastive_loss_weight * contrastive_loss

        return SltCausalLMOutputWithPast(
            loss=loss,
            main_loss=main_loss,
            contrastive_loss=contrastive_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        pixel_values_length=None,
        cache_position=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        # Extend the standard generation inputs with video tensors during prefill.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # Cached decoding no longer contains video placeholder tokens, so video
        # tensors are only passed on the first generation step.
        if cache_position[0] == 0:
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
