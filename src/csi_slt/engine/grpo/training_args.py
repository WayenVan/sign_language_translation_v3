"""Training arguments for the experimental SLT/TRL integration.

This module is deliberately separate from :mod:`csi_slt.engine.sft.training_args` so
the existing supervised training entry points keep their current behaviour.
"""

from dataclasses import dataclass, field

try:
    from trl import GRPOConfig
except ImportError as error:  # pragma: no cover - depends on an optional package
    raise ImportError(
        "SltGRPOConfig requires TRL. Install a TRL version compatible with the "
        "project's Transformers version before importing this module."
    ) from error


@dataclass
class SltGRPOConfig(GRPOConfig):
    """GRPO arguments plus the small amount of SLT-specific policy.

    The conservative defaults are intended for bringing up the integration:
    native Transformers generation is used and the reference-policy KL is
    disabled. They can be changed once the basic policy path is verified.
    """

    use_vllm: bool = field(
        default=False,
        metadata={"help": "Keep disabled until SltModel has a vLLM adapter."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Keep disabled because the outer SltModel does not yet "
            "support Transformers gradient-checkpointing hooks."
        },
    )
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. Start at zero; internal-LoRA reference "
            "evaluation needs the custom trainer path."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Preserve videos, language metadata, and reward columns."
        },
    )
    slt_disable_auxiliary_losses: bool = field(
        default=True,
        metadata={
            "help": "Require D-SID and attention-diversity losses to be disabled."
        },
    )
    slt_use_internal_lora_reference: bool = field(
        default=False,
        metadata={
            "help": "Evaluate the reference policy by disabling model.llm's adapter."
        },
    )
