"""Manually exercise loading Gemma weights into an empty-initialized model."""

import numpy as np

import torch

import torchinfo
from torchinfo import summary
from transformers import AutoConfig, AutoModel
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


config = AutoConfig.from_pretrained("google/gemma-3-1b-it")
with init_empty_weights():
    model = Gemma3ForCausalLM._from_config(config, attn_implementation="eager")

state_dict = AutoModel.from_pretrained("google/gemma-3-1b-it").state_dict()

model.load_state_dict(state_dict)
model.tied_weights()
