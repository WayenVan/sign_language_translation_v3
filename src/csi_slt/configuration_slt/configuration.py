from transformers.configuration_utils import PretrainedConfig
from typing import Any, Dict, Optional, Union
from transformers import AutoConfig


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
        visual_backbone_type: str = "resnet50",
        visual_backbone_config: Optional[Dict[str, Any]] = None,
        visual_adapter_type: str = "linear",
        visual_adapter_kwargs: Optional[Dict[str, Any]] = None,
        contrastive_dim: int = 512,
        contrastive_loss_weight: float = 0.25,
        contrastive_text_queue_size: int = 0,
        alignment_loss_weight: float = 1.0,
        alignment_eps: float = 0.12,
        alignment_n_iters: int = 10,
        alignment_target_relaxation: float = 0.5,
        alignment_null_mass_prior: float = 0.2,
        alignment_null_ratio_max: float = 0.2,
        alignment_null_temperature: float = 0.1,
        alignment_beta_ot: float = 1.0,
        alignment_beta_null: float = 0.1,
        alignment_beta_tv: float = 2.0,
        alignment_pooling_distill_weight: float = 0.5,
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
            contrastive_dim: Output width of the trainable visual and textual
                projection heads used by global contrastive learning.
            contrastive_loss_weight: Weight applied to the global video-text
                contrastive objective before it is added to the language-model
                loss.
            contrastive_text_queue_size: Number of detached historical text
                features used as additional video-to-text negatives. Zero
                disables the queue.
            alignment_loss_weight: Weight of local visual/pseudo-gloss OT
                alignment in the total training loss. Zero disables it.
            alignment_eps: Initial entropy coefficient for semi-unbalanced
                Sinkhorn OT. Training may update it through a scheduler.
            alignment_n_iters: Number of Sinkhorn scaling iterations.
            alignment_target_relaxation: KL strength for the relaxed target
                marginal.
            alignment_null_mass_prior: Prior transport mass assigned to NULL.
            alignment_null_ratio_max: Maximum preferred local NULL ratio.
            alignment_null_temperature: Temperature for the NULL preference
                softmax.
            alignment_beta_ot: Internal weight of the OT objective.
            alignment_beta_null: Internal weight of the NULL regularizer.
            alignment_beta_tv: Internal weight of the temporal-variation loss
                on the row-normalized real-token plan.
            alignment_pooling_distill_weight: Weight of the KL loss that uses
                detached non-NULL OT mass to supervise visual-only global
                attention. Zero disables attention distillation.
            **kwargs: Standard Hugging Face ``PretrainedConfig`` fields, such
                as serialization and generation metadata.
        """
        super().__init__(**kwargs)

        self.video_soft_token_id = video_soft_token_id
        self.llm_model_name_or_path = llm_model_name_or_path
        self.llm_init_kwargs = llm_init_kwargs if llm_init_kwargs is not None else {}
        self.llm_lora = llm_lora
        self.llm_lora_config = llm_lora_config if llm_lora_config is not None else {}
        self.visual_backbone_type = visual_backbone_type
        self.visual_backbone_config = (
            visual_backbone_config if visual_backbone_config is not None else {}
        )
        self.visual_adapter_type = visual_adapter_type
        self.visual_adapter_kwargs = (
            visual_adapter_kwargs if visual_adapter_kwargs is not None else {}
        )
        if contrastive_dim <= 0:
            raise ValueError("contrastive_dim must be positive")
        self.contrastive_dim = contrastive_dim
        self.contrastive_loss_weight = contrastive_loss_weight
        if contrastive_text_queue_size < 0:
            raise ValueError("contrastive_text_queue_size must be non-negative")
        self.contrastive_text_queue_size = contrastive_text_queue_size
        if alignment_loss_weight < 0:
            raise ValueError("alignment_loss_weight must be non-negative")
        if alignment_eps <= 0:
            raise ValueError("alignment_eps must be positive")
        if alignment_n_iters <= 0:
            raise ValueError("alignment_n_iters must be positive")
        if alignment_target_relaxation <= 0:
            raise ValueError("alignment_target_relaxation must be positive")
        if not 0 < alignment_null_mass_prior < 1:
            raise ValueError("alignment_null_mass_prior must lie in (0, 1)")
        if not 0 <= alignment_null_ratio_max <= 1:
            raise ValueError("alignment_null_ratio_max must lie in [0, 1]")
        if alignment_null_temperature <= 0:
            raise ValueError("alignment_null_temperature must be positive")
        if alignment_beta_ot < 0 or alignment_beta_null < 0 or alignment_beta_tv < 0:
            raise ValueError("alignment beta weights must be non-negative")
        if alignment_pooling_distill_weight < 0:
            raise ValueError("alignment_pooling_distill_weight must be non-negative")
        if alignment_pooling_distill_weight > 0 and alignment_loss_weight <= 0:
            raise ValueError(
                "alignment_loss_weight must be positive when alignment pooling "
                "distillation is enabled"
            )
        self.alignment_loss_weight = alignment_loss_weight
        self.alignment_eps = alignment_eps
        self.alignment_n_iters = alignment_n_iters
        self.alignment_target_relaxation = alignment_target_relaxation
        self.alignment_null_mass_prior = alignment_null_mass_prior
        self.alignment_null_ratio_max = alignment_null_ratio_max
        self.alignment_null_temperature = alignment_null_temperature
        self.alignment_beta_ot = alignment_beta_ot
        self.alignment_beta_null = alignment_beta_null
        self.alignment_beta_tv = alignment_beta_tv
        self.alignment_pooling_distill_weight = alignment_pooling_distill_weight

        # New checkpoints embed the complete LLM configuration. Loading it by
        # name is retained only for old configs that do not contain this field.
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
