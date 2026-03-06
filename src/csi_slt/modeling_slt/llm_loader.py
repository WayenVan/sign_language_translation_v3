from transformers import AutoConfig


def load_llm(model_name, llm_init_kwargs={}, is_meta_model=False):
    if "qwen" in model_name.lower():
        from transformers.models.qwen3 import Qwen3ForCausalLM

        model_cls = Qwen3ForCausalLM
    elif "gemma" in model_name.lower():
        from transformers.models.gemma3 import (
            Gemma3ForCausalLM,
            Gemma3ForConditionalGeneration,
        )

        if "1b" in model_name.lower():
            model_cls = Gemma3ForCausalLM
        else:
            model_cls = Gemma3ForConditionalGeneration
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

    attn_implementation = llm_init_kwargs.pop(
        "attn_implementation",
        "eager",  # NOTE: default to eager since spda produce nan
    )
    config = AutoConfig.from_pretrained(model_name)

    if is_meta_model:
        model = model_cls._from_config(
            config, attn_implementation=attn_implementation, **llm_init_kwargs
        )
    else:
        model = model_cls.from_pretrained(
            model_name, attn_implementation=attn_implementation, **llm_init_kwargs
        )

    if hasattr(config, "text_config"):
        config = config.text_config

    model.tie_weights()

    return model, config
