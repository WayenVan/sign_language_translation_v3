from transformers import AutoTokenizer, BitsAndBytesConfig

from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForCausalLM,
    Gemma4UnifiedForConditionalGeneration,
)

from transformers import AutoModelForMultimodalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")
model = Gemma4UnifiedForConditionalGeneration.from_pretrained("google/gemma-4-12b-it")

# model.config

tokenizer.add_bos_token = False
tokenizer.add_eos_token = True
print(tokenizer.eos_token_id)
print(tokenizer.eos_token)
print(tokenizer.convert_tokens_to_ids("<unused0>"))
print(tokenizer.convert_tokens_to_ids("<unused1>"))

# output the tokens rather than the ids
label_ids = tokenizer(
    "Hello, my dog is cute",
    add_special_tokens=True,
).input_ids
print(label_ids)


messages = [{"role": "user", "content": "Hello, my dog is cute"}]
chat_prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

print(tokenizer.decode(chat_prompt + label_ids, skip_special_tokens=False))
tokenizer.save_pretrained("outputs/gemma_tokenizer")
