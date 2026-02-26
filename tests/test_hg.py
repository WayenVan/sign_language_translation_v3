import numpy as np

import torch

import torchinfo
from torchinfo import summary
from transformers import AutoConfig


config = AutoConfig.from_pretrained("google/gemma-3-4b-it")


c_s: str = "你 好世界"


print(tokenizer.tokenize(c_s))
