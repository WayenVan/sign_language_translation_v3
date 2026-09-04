from transformers.configuration_utils import PretrainedConfig
from typing import Any, Dict, Optional, Union
from transformers import AutoConfig


# How the temporal position of each visual token is encoded before the token is
# injected into the language model. The language model already applies RoPE over
# the whole sequence, so this is a *second*, adapter-side position signal.
VISUAL_POSITION_EMBEDDING_TYPES = ("learned", "none", "sincos")
CTC_CODEBOOK_TRAINING_MODES = ("soft", "straight_through", "argmax")


class SltConfig(PretrainedConfig):
    """Configuration for the packed-video sign-language translation models.

    An SLT input contains frames from all videos in a batch concatenated along
    dimension 0, plus one text sequence per video. The visual backbone and
    adapter convert those packed frames into visual tokens. In the text prompt,
    every visual token is represented by ``video_soft_token_id``; two additional
    placeholders are required for the learned start- and end-of-video tokens.

    Consequently, the data processor, visual adapter, and
    ``video_token_scale`` must agree: for a video with ``L`` input frames, the
    processor inserts ``int(L * video_token_scale) + 2`` video placeholders,
    and the adapter must emit exactly ``int(L * video_token_scale)`` visual
    tokens. The base :class:`~csi_slt.modeling_slt.slt.SltModel` validates the
    placeholder count at runtime.

    The configuration selects component classes by their registry keys and
    forwards the corresponding ``*_config``/``*_kwargs`` dictionaries to the
    component constructors. It does not infer dimensions or temporal reduction
    from the selected components. The complete language-model configuration is
    stored in ``llm_config`` and serialized with this configuration.
    ``llm_model_name_or_path`` locates the weights and is a compatibility
    fallback for older checkpoints without an embedded LLM configuration.
    """

    model_type = "slt"
    # Avoid constructing a default SltConfig (and therefore resolving a remote
    # fallback LLM) merely to compute the serialized config diff.
    has_no_defaults_at_init = True
    # Declaring the embedded LLM configuration as a sub-config is what lets
    # Transformers propagate runtime-resolved settings (notably
    # ``_attn_implementation``, which is never serialized) from this config into
    # ``llm_config``. Without it that propagation silently stops here and
    # ``llm_config._attn_implementation`` stays ``None``, which makes
    # ``create_causal_mask`` return no mask at all.
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        video_soft_token_id: int = -1,
        video_token_scale: float = 1.0,
        llm_model_name_or_path: str = "google/gemma-3-1b-it",
        llm_config: Optional[Union[PretrainedConfig, Dict[str, Any]]] = None,
        llm_init_kwargs: Optional[Dict[str, Any]] = None,
        llm_lora: bool = False,
        llm_lora_config: Optional[Dict[str, Any]] = None,
        visual_lora: bool = False,
        visual_lora_config: Optional[Dict[str, Any]] = None,
        visual_backbone_type: str = "resnet50",
        visual_backbone_config: Optional[Dict[str, Any]] = None,
        visual_adapter_type: str = "linear",
        visual_adapter_kwargs: Optional[Dict[str, Any]] = None,
        ctc_loss_weight: float = 0.0,
        ctc_vocab_size: Optional[int] = None,
        ctc_blank_id: Optional[int] = None,
        ctc_codebook_training_mode: str = "soft",
        ctc_codebook_default_temperature: float = 1.0,
        video_bidirectional_attention: Optional[bool] = None,
        visual_position_embedding_type: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the serializable SLT configuration.

        Args:
            hidden_size: Compatibility field for older configurations. When
                provided, it must match the canonical LLM text hidden size.
            video_soft_token_id: Tokenizer id used as each video placeholder in
                ``input_ids``. The processor emits a contiguous run of these
                tokens, and the model replaces that run with the start token,
                visual tokens, and end token. It must be a valid id for the
                chosen LLM tokenizer.
            video_token_scale: Ratio of adapter output visual tokens to input
                video frames. For ``L`` input frames, prompts contain
                ``int(L * video_token_scale) + 2`` placeholders. The two extra
                placeholders are consumed by learned start/end video tokens.
                The current ``SltModel`` also requires each input frame length
                to be divisible by ``int(1 / video_token_scale)``. This value
                must match the selected adapter's effective temporal reduction;
                it does not rescale token embeddings.
            llm_model_name_or_path: Hugging Face model id or local path for the
                language-model weights. It loads the LLM config only when
                ``llm_config`` is absent.
            llm_config: Complete LLM configuration, as a ``PretrainedConfig``
                object or its serialized dictionary.
            llm_init_kwargs: Optional keyword arguments for LLM initialization.
                They are used by ``SltQwenVLModel`` when loading/constructing
                its language model. ``SltModel`` and
                ``SltModel.from_pretrained_components`` currently load the LLM
                without forwarding this dictionary.
            visual_backbone_type: Key in ``VISUAL_BACKBONES`` selecting the
                visual backbone. The current registry provides ``"dinov2"``
                and ``"pretrained"``. The historical default ``"resnet50"``
                is not a registered key and therefore cannot construct a
                current ``SltModel`` unchanged.
            visual_backbone_config: Keyword arguments passed as one dictionary
                to the selected visual backbone constructor; for pretrained
                component loading it is passed to that backbone's
                ``from_pretrained_backbone`` method. For example, the DINOv2
                backbone expects keys such as ``id`` and ``output_layer``.
            visual_adapter_type: Key in ``VISUAL_ADAPTERS`` selecting the
                visual adapter. Current keys are ``"token_sampler"``,
                ``"temporal_shuffle"``, ``"temporal_merge"``,
                ``"token_sampler_v2"``, and ``"dinoframe"``. The historical
                default ``"linear"`` is not a registered key.
            visual_adapter_kwargs: Keyword arguments unpacked into the selected
                adapter constructor. Its output width must match ``hidden_size``
                and its temporal downsampling must agree with
                ``video_token_scale``.
            ctc_loss_weight: Coefficient applied to the CTC loss when summed
                with the language-model loss. Set it to zero for a phase that
                does not optimize the CTC objective; the CTC head and codebook
                remain mandatory model components.
            ctc_vocab_size: Vocabulary size of the CTC head's output
                projection, i.e. the size of the word-level CTC tokenizer
                (including the blank token). Required for every model.
            ctc_blank_id: Token id used as the CTC blank symbol, valid within
                ``[0, ctc_vocab_size)``. Required for every model; it is
                dataset-tokenizer-specific and not assumed to be ``0``.
            ctc_codebook_training_mode: Codebook selection rule used while the
                model is in training mode: differentiable ``"soft"``,
                straight-through Gumbel ``"straight_through"``, or hard
                ``"argmax"``. Evaluation always uses argmax.
            ctc_codebook_default_temperature: Default softmax/Gumbel
                temperature used during training when forward does not provide
                a step-specific override. Ignored by argmax and evaluation.
            video_bidirectional_attention: Whether video tokens attend to each
                other in both directions during prefill, instead of only
                causally. ``None`` means "not recorded by this checkpoint" and
                resolves to ``False``: runs predating this field were trained
                with the overlay silently disabled (``llm_config`` was never
                attn-resolved, so ``create_causal_mask`` returned no mask), and
                loading them with the overlay enabled would evaluate them under
                an attention pattern they never saw. A freshly constructed
                configuration defaults to ``True``.
            visual_position_embedding_type: How the temporal position of a
                visual token is encoded before injection into the language
                model, which applies its own RoPE on top of it. ``"learned"``
                (the default, and what ``None`` resolves to, so existing
                checkpoints are unaffected) adds a trainable
                ``MAX_TOKEN_LENGTH x hidden_size`` table. ``"none"`` adds
                nothing and leaves the temporal order entirely to RoPE.
                ``"sincos"`` adds a fixed sinusoidal table, whose rows are
                L2-normalized so its magnitude matches the learned table at
                initialization; it owns no parameters, so unlike the learned
                table its tail cannot stay at initialization for positions the
                training set rarely reaches.
            **kwargs: Standard Hugging Face ``PretrainedConfig`` fields, such
                as serialization and generation metadata.
        """
        # Retired objective keys and old model-owned scheduling keys may still
        # exist in checkpoints. Ignore them instead of reintroducing inactive
        # attributes that appear to affect the current model.
        for retired_key in (
            "contrastive_dim",
            "contrastive_loss_weight",
            "contrastive_text_queue_size",
            "alignment_loss_weight",
            "alignment_eps",
            "alignment_n_iters",
            "alignment_target_relaxation",
            "alignment_null_mass_prior",
            "alignment_null_ratio_max",
            "alignment_null_temperature",
            "alignment_beta_ot",
            "alignment_beta_null",
            "alignment_beta_tv",
            "alignment_pooling_distill_weight",
            "dsid_loss_weight",
            "dsid_js_tau",
            "dsid_warmup_ratio",
            "dsid_decay_ratio",
            "visual_semantic_encoder_type",
            "visual_semantic_encoder_config",
            "attention_diversity_loss_weight",
            # CTC is mandatory in this model generation. Accept the retired
            # switch from older YAML/config files without restoring an
            # attribute that appears capable of disabling the architecture.
            "ctc_enabled",
        ):
            kwargs.pop(retired_key, None)

        # Transformers validates token ids during PretrainedConfig
        # initialization. That validation dynamically calls get_text_config(),
        # so the embedded language-model config must exist before super().__init__.
        if llm_config is None:
            llm_config = AutoConfig.from_pretrained(llm_model_name_or_path)
        elif isinstance(llm_config, dict):
            llm_config = dict(llm_config)
            model_type = llm_config.pop("model_type", None)
            if model_type is None:
                raise ValueError("Serialized llm_config must contain 'model_type'.")
            llm_config = AutoConfig.for_model(model_type, **llm_config)
        elif not isinstance(llm_config, PretrainedConfig):
            raise TypeError("llm_config must be a PretrainedConfig, a dict, or None.")

        self.llm_config = llm_config
        super().__init__(**kwargs)

        self.video_soft_token_id = video_soft_token_id
        self.llm_model_name_or_path = llm_model_name_or_path
        self.llm_init_kwargs = llm_init_kwargs if llm_init_kwargs is not None else {}
        self.llm_lora = llm_lora
        self.llm_lora_config = llm_lora_config if llm_lora_config is not None else {}
        self.visual_lora = visual_lora
        self.visual_lora_config = (
            visual_lora_config if visual_lora_config is not None else {}
        )
        self.visual_backbone_type = visual_backbone_type
        self.visual_backbone_config = (
            visual_backbone_config if visual_backbone_config is not None else {}
        )
        self.visual_adapter_type = visual_adapter_type
        self.visual_adapter_kwargs = (
            visual_adapter_kwargs if visual_adapter_kwargs is not None else {}
        )
        if isinstance(ctc_loss_weight, bool) or not isinstance(
            ctc_loss_weight, (int, float)
        ):
            raise TypeError("ctc_loss_weight must be a real number")
        if ctc_loss_weight < 0.0:
            raise ValueError("ctc_loss_weight must be non-negative")
        if isinstance(ctc_vocab_size, bool) or not isinstance(ctc_vocab_size, int):
            raise TypeError("ctc_vocab_size must be an int")
        if ctc_vocab_size <= 0:
            raise ValueError("ctc_vocab_size must be positive")
        if isinstance(ctc_blank_id, bool) or not isinstance(ctc_blank_id, int):
            raise TypeError("ctc_blank_id must be an int")
        if not 0 <= ctc_blank_id < ctc_vocab_size:
            raise ValueError("ctc_blank_id must be in [0, ctc_vocab_size)")
        self.ctc_loss_weight = float(ctc_loss_weight)
        self.ctc_vocab_size = ctc_vocab_size
        self.ctc_blank_id = ctc_blank_id
        if not isinstance(ctc_codebook_training_mode, str):
            raise TypeError("ctc_codebook_training_mode must be a str")
        if ctc_codebook_training_mode not in CTC_CODEBOOK_TRAINING_MODES:
            raise ValueError(
                "ctc_codebook_training_mode must be one of "
                f"{CTC_CODEBOOK_TRAINING_MODES}, got "
                f"{ctc_codebook_training_mode!r}"
            )
        if isinstance(ctc_codebook_default_temperature, bool) or not isinstance(
            ctc_codebook_default_temperature, (int, float)
        ):
            raise TypeError("ctc_codebook_default_temperature must be a real number")
        if ctc_codebook_default_temperature < 0.1:
            raise ValueError("ctc_codebook_default_temperature must be >= 0.1")
        self.ctc_codebook_training_mode = ctc_codebook_training_mode
        self.ctc_codebook_default_temperature = float(
            ctc_codebook_default_temperature
        )

        if video_bidirectional_attention is None:
            # A deserialized configuration always carries ``transformers_version``;
            # a configuration built in code never does. Checkpoints written before
            # this field existed therefore keep the behaviour they were trained
            # with, while new experiments opt in by default.
            video_bidirectional_attention = "transformers_version" not in kwargs
        if not isinstance(video_bidirectional_attention, bool):
            raise TypeError("video_bidirectional_attention must be a bool or None")
        self.video_bidirectional_attention = video_bidirectional_attention

        if visual_position_embedding_type is None:
            # Unlike ``video_bidirectional_attention``, the default here is the
            # historical behaviour for old and new configurations alike: every
            # run so far used the learned table, and nothing about it was
            # silently disabled, so there is no regime to distinguish.
            visual_position_embedding_type = "learned"
        if not isinstance(visual_position_embedding_type, str):
            raise TypeError("visual_position_embedding_type must be a str or None")
        if visual_position_embedding_type not in VISUAL_POSITION_EMBEDDING_TYPES:
            raise ValueError(
                "visual_position_embedding_type must be one of "
                f"{VISUAL_POSITION_EMBEDDING_TYPES}, got "
                f"{visual_position_embedding_type!r}"
            )
        self.visual_position_embedding_type = visual_position_embedding_type

        text_hidden_size = self.get_text_config().hidden_size
        if hidden_size is not None and hidden_size != text_hidden_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) does not match the LLM text "
                f"hidden_size ({text_hidden_size})."
            )
        # Compatibility alias for the SLT projection modules. Its source of
        # truth is the embedded LLM text configuration.
        self.hidden_size = text_hidden_size

        self.video_token_scale = video_token_scale
        self.num_extra_tokens = None

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        """Return the text portion of the embedded language-model config."""
        return self.llm_config.get_text_config(decoder=decoder, encoder=encoder)
