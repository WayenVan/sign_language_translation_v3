from transformers.configuration_utils import PretrainedConfig
from typing import Any, Dict, Optional
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
    from the selected components. Constructing this configuration loads the LLM
    ``AutoConfig`` from ``llm_model_name_or_path``; that identifier or path must
    therefore be available even when only constructing configuration objects.

    During model construction, ``num_extra_tokens`` is set to ``2`` and the
    ``bos_token_id``, ``eos_token_id``, and ``pad_token_id`` fields are copied
    from the loaded LLM. ``num_hidden_layers`` is derived from the LLM
    ``AutoConfig`` when that configuration exposes a compatible field.
    """

    model_type = "slt"

    def __init__(
        self,
        hidden_size: int = 512,
        video_soft_token_id: int = -1,
        video_token_scale: float = 1.0,
        llm_model_name_or_path: str = "google/gemma-3-1b-it",
        llm_init_kwargs: Optional[Dict[str, Any]] = None,
        visual_backbone_type: str = "resnet50",
        visual_backbone_config: Optional[Dict[str, Any]] = None,
        visual_adapter_type: str = "linear",
        visual_adapter_kwargs: Optional[Dict[str, Any]] = None,
        contrastive_loss_weight: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize the serializable SLT configuration.

        Args:
            hidden_size: Hidden width of the language model's text embedding
                space. It determines the shapes of the learned video boundary
                embeddings and visual positional embedding in ``SltModel``;
                it must equal the LLM text hidden size and the output dimension
                of the selected visual adapter. It is not automatically checked
                against either component.
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
                language model. It is also used to load an ``AutoConfig`` during
                ``SltConfig`` construction in order to expose
                ``num_hidden_layers``.
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
            contrastive_loss_weight: Stored in the serialized configuration but
                currently not consumed by ``SltModel`` or ``SltQwenVLModel``
                when computing their loss. Changing it currently has no effect
                on training unless an external training loop reads it.
            **kwargs: Standard Hugging Face ``PretrainedConfig`` fields, such
                as serialization and generation metadata.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.video_soft_token_id = video_soft_token_id
        self.llm_model_name_or_path = llm_model_name_or_path
        self.llm_init_kwargs = llm_init_kwargs if llm_init_kwargs is not None else {}
        self.visual_backbone_type = visual_backbone_type
        self.visual_backbone_config = (
            visual_backbone_config if visual_backbone_config is not None else {}
        )
        self.visual_adapter_type = visual_adapter_type
        self.visual_adapter_kwargs = (
            visual_adapter_kwargs if visual_adapter_kwargs is not None else {}
        )
        self.contrastive_loss_weight = contrastive_loss_weight
        llm_config = AutoConfig.from_pretrained(
            llm_model_name_or_path
        )  # NOTE: using AutoConfig to support more models
        if hasattr(llm_config, "num_hidden_layers"):
            self.num_hidden_layers = llm_config.num_hidden_layers
        elif hasattr(llm_config, "text_config"):
            self.num_hidden_layers = llm_config.text_config.num_hidden_layers

        self.video_token_scale = video_token_scale
        self.num_extra_tokens = None

        self.bos_token_id = None  # to be set when laoding the tokenizer
        self.eos_token_id = None  # to be set when laoding the tokenizer
        self.pad_token_id = None  # to be set when laoding the tokenizer
