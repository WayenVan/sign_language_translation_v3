"""Manually inspect Qwen3-VL video feature extraction."""

from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel
import torch
from einops import rearrange


model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct"
).eval()


fake_video = torch.randn(
    4 + 8, 3, 224, 224
)  # (batch_size, channels, frames, height, width)

fake_video_grid = torch.tensor(
    [[4 / 2, 224 / 16, 224 / 16], [8 / 2, 224 / 16, 224 / 16]], dtype=torch.long
)

video_outputs = model.get_video_features(fake_video, fake_video_grid, return_dict=True)

out = video_outputs.last_hidden_state
oout = rearrange(
    out, "(f h w) d -> f h w d", h=fake_video_grid[0, 1], w=fake_video_grid[0, 2]
)
print(video_outputs)
