from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
print(tokenizer.convert_tokens_to_ids("<|video_pad|>"))

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt")

print(tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=False))
