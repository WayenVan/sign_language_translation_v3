from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass, field


@dataclass
class SltConfig(PretrainedConfig):
    hidden_size: int = field(
        default=512,
        metadata={"help": "Dimensionality of the model's hidden states."},
    )

    video_soft_token_id: int = field(
        default=-1,
        metadata={
            "help": "the video soft token id when replaceing <video> token in the text input"
        },
    )

    llm_model_name_or_path: str = field(
        default="google/gemma-3-1b-it",
        metadata={
            "help": "Path to the pre-trained LLM model or model identifier from huggingface.co/models"
        },
    )
    llm_init_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={
            "help": "Additional keyword arguments for when using .from_pretrained() to load the LLM model."
        },
    )

    visual_backbone_type: str = field(
        default="resnet50",
        metadata={
            "help": "Type of visual backbone to use. Options: 'resnet50', 'vit-base-patch16-224', etc."
        },
    )

    visual_backbone_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={
            "help": "Additional keyword arguments for the visual backbone model."
        },
    )

    visual_adapter_type: str = field(
        default="linear",
        metadata={
            "help": "Type of adapter to use for visual features. Options: 'linear', 'mlp', etc."
        },
    )

    visual_adapter_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={"help": "Additional keyword arguments for the visual adapter."},
    )
