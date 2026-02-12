from transformers import AutoConfig


def load_llm(model_name, llm_init_kwargs={}):
    if "qwen" in model_name.lower():
        from transformers.models.qwen3 import Qwen3ForCausalLM

        model_cls = Qwen3ForCausalLM
    elif "gemma" in model_name.lower():
        from transformers.models.gemma3 import Gemma3ForCausalLM

        model_cls = Gemma3ForCausalLM
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

    attn_implementation = llm_init_kwargs.pop(
        "attn_implementation",
        "eager",  # NOTE: default to eager since spda produce nan
    )
    model = model_cls.from_pretrained(
        model_name, attn_implementation=attn_implementation, **llm_init_kwargs
    )
    config = AutoConfig.from_pretrained(model_name)

    return model, config
