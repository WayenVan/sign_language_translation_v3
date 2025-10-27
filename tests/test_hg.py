from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
import numpy as np

import torch

import torchinfo
from torchinfo import summary


tokenizer: GemmaTokenizerFast = GemmaTokenizerFast.from_pretrained(
    "google/gemma-3-1b-it"
)


c_s: str = "你 好世界"


print(tokenizer.tokenize(c_s))
