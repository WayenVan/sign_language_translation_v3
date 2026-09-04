import timm.models.sknet
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from datasets import load_dataset, concatenate_datasets

from transformers import PreTrainedTokenizerFast


TEXT_COLUMN = "orth"


# =========================
# 1. 定义 tokenizer
# =========================

tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

# 严格按照空格切分
tokenizer.pre_tokenizer = WhitespaceSplit()


# =========================
# 2. 定义 trainer
# =========================

special_tokens = [
    "<pad>",
    "<unk>",
    "<blank>",
]

trainer = WordLevelTrainer(
    special_tokens=special_tokens,
    min_frequency=1,
)


# =========================
# 3. 从训练语料构建 vocab
# =========================


train_set = load_dataset(
    "WayenVan/ph14t-multilang",
    split="train",
)

val_set = load_dataset(
    "WayenVan/ph14t-multilang",
    split="validation",
)

test_set = load_dataset(
    "WayenVan/ph14t-multilang",
    split="test",
)

full_set = concatenate_datasets([train_set, val_set, test_set])


def batch_iterator(dataset, column=TEXT_COLUMN, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][column]


tokenizer.train_from_iterator(
    batch_iterator(full_set),
    trainer=trainer,
    length=len(full_set),
)


# =========================
# 4. 包装成 HuggingFace tokenizer
# =========================

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<pad>",
    unk_token="<unk>",
    # 注意：
    # transformers 本身没有标准的 blank_token 参数
    # 所以 <blank> 作为 additional special token 保存
    extra_special_tokens={
        "blank_token": "<blank>",
    },
)


# =========================
# 5. 查看 vocab
# =========================

print("vocab size:", len(hf_tokenizer))

print("pad:", hf_tokenizer.pad_token_id)
print("unk:", hf_tokenizer.unk_token_id)

blank_token_id = hf_tokenizer.convert_tokens_to_ids("<blank>")
print("blank:", blank_token_id)


# =========================
# 6. 测试
# =========================

text = "JETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR"

encoded = hf_tokenizer(
    text,
    add_special_tokens=False,
)

print(encoded)
print(hf_tokenizer.convert_ids_to_tokens(encoded["input_ids"]))


# =========================
# 7. 保存
# =========================

hf_tokenizer.save_pretrained("./outputs/ctc_tokenizer_real")
