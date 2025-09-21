from transformers.trainer import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma3 import Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from safetensors import safe_open


with safe_open(
    "outputs/test_save/model.safetensors", framework="pt", device="cpu"
) as f:
    for k in f.keys():
        print(k)
