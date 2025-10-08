from transformers import AutoTokenizer, BitsAndBytesConfig

from transformers.models.gemma.tokenization_gemma import GemmaTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

tokenizer.add_bos_token = False
tokenizer.add_eos_token = True
tokenizer.eos_token = "<end_of_turn>"
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
print(tokenizer.eos_token_id)
print(tokenizer.eos_token)

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
