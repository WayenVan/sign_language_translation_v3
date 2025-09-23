from transformers.configuration_utils import PretrainedConfig
from typing import Any, Dict, Optional
from transformers import AutoConfig


class SltConfig(PretrainedConfig):
    """Configuration class for SLT (Sign Language Translation) model.

    This class stores the configuration for the SLT model, including settings for
    the language model, visual backbone, and adapters.
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
        visual_backbone_kwargs: Optional[Dict[str, Any]] = None,
        visual_adapter_type: str = "linear",
        visual_adapter_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize SltConfig.

        Args:
            hidden_size: Dimensionality of the model's hidden states.
            video_soft_token_id: The video soft token id when replacing <video> token in the text input.
            video_token_scale: Scaling factor for video token embeddings.
            llm_model_name_or_path: Path to the pre-trained LLM model or model identifier from huggingface.co/models.
            llm_init_kwargs: Additional keyword arguments for when using .from_pretrained() to load the LLM model.
            visual_backbone_type: Type of visual backbone to use. Options: 'resnet50', 'vit-base-patch16-224', etc.
            visual_backbone_kwargs: Additional keyword arguments for the visual backbone model.
            visual_adapter_type: Type of adapter to use for visual features. Options: 'linear', 'mlp', etc.
            visual_adapter_kwargs: Additional keyword arguments for the visual adapter.
            **kwargs: Additional arguments passed to the parent PretrainedConfig class.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.video_soft_token_id = video_soft_token_id
        self.llm_model_name_or_path = llm_model_name_or_path
        self.llm_init_kwargs = llm_init_kwargs if llm_init_kwargs is not None else {}
        self.visual_backbone_type = visual_backbone_type
        self.visual_backbone_kwargs = (
            visual_backbone_kwargs if visual_backbone_kwargs is not None else {}
        )
        self.visual_adapter_type = visual_adapter_type
        self.visual_adapter_kwargs = (
            visual_adapter_kwargs if visual_adapter_kwargs is not None else {}
        )

        llm_config = AutoConfig.from_pretrained(
            llm_model_name_or_path
        )  # NOTE: using AutoConfig to support more models
        self.num_hidden_layers = llm_config.num_hidden_layers
        self.video_token_scale = video_token_scale
        self.num_extra_tokens = None

        self.bos_token_id = None  # to be set when laoding the tokenizer
        self.eos_token_id = None  # to be set when laoding the tokenizer
        self.pad_token_id = None  # to be set when laoding the tokenizer
