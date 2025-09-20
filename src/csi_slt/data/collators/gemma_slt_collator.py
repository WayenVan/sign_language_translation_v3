import numpy as np
import torch
from ..constants import Language
import random
from typing import List, Literal


VIDEO_SOFT_TOKEN = "<unused0>"
VIDEO_START_TOKEN = "<unused1>"


class Gemma3SLTCollator:
    """
    collator for my mbart model, which changes lots of things from the original gfslt collator
    <unused0> is used as image soft token, <unused1> is used as image sentinel
    """

    def __init__(
        self,
        tokenizer,
        video_token_scale: int = 2,
        num_extra_video_tokens: int = 2,
        mode: Literal["train", "val", "test"] = "train",
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.video_token_scale = video_token_scale
        self.num_extra_video_tokens = num_extra_video_tokens
        self.image_soft_token_id = self.tokenizer.convert_tokens_to_ids("<unused0>")

    @staticmethod
    def pad_dim_to_multiple_of_4(tensor, dim):
        current_size = tensor.size(dim)
        remainder = current_size % 4
        if remainder == 0:
            return tensor

        pad_size = 4 - remainder

        # 取这个维度的最后一个元素
        index = [slice(None)] * tensor.dim()
        index[dim] = -1
        last_element = tensor[tuple(index)].unsqueeze(dim)

        # 复制 pad_size 次
        padding = last_element.repeat_interleave(pad_size, dim=dim)
        return torch.cat([tensor, padding], dim=dim).contiguous()

    @staticmethod
    def inject_images(prompt: str, n: int) -> str:
        sentinel = VIDEO_START_TOKEN
        replacement = VIDEO_SOFT_TOKEN * n
        return prompt.replace(sentinel, replacement)

    def __call__(self, batch):
        # Collate a batch of samples.
        zbatch = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        # Unpack batch data
        names, videos, texts = (
            zbatch["id"],
            zbatch["augmented_video"],
            zbatch["text"],
        )

        # Stack all videos into single tensor
        # video (T, C, H, W) ...
        #
        videos = [self.pad_dim_to_multiple_of_4(video, dim=0) for video in videos]
        video_lengths = [video.size(0) for video in videos]

        video_tensor = torch.cat(videos, dim=0).contiguous()
        video_lengths_tensor = torch.tensor(video_lengths)

        prompts = []
        all_input_ids = []
        all_label_ids = []
        input_text = []
        for i, t in enumerate(zbatch["text"]):
            message = [{"role": "user", "language": Language.DE.value}]

            prompt = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            # inject images if needed
            if VIDEO_START_TOKEN in prompt:
                prompt = self.inject_images(
                    prompt,
                    int(video_lengths[i] * self.video_token_scale)
                    + self.num_extra_video_tokens,
                )

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = self.tokenizer(t, add_special_tokens=False).input_ids + [
                self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]

            label_ids = [-100] * len(
                prompt_ids
            ) + target_ids  # mask prompt part for loss calculation

            if self.mode == "train":
                input_ids = prompt_ids + target_ids
            else:
                # For inference, we only use the prompt
                input_ids = prompt_ids

            prompts.append(prompt)
            all_input_ids.append(input_ids)
            all_label_ids.append(label_ids)
            input_text.append(
                self.tokenizer.decode(input_ids, skip_special_tokens=False)
            )

        # pad on the left
        self.tokenizer.padding_side = "left"
        all_input_ids = self.tokenizer.pad(
            {"input_ids": all_input_ids},
            padding=True,
            return_tensors="pt",  # or "tf" / "np"
        )
        all_label_ids = self.tokenizer.pad(
            {"input_ids": all_label_ids},
            padding=True,
            return_tensors="pt",  # or "tf" / "np"
        )["input_ids"]
        all_label_ids[all_label_ids == self.tokenizer.pad_token_id] = -100

        # Prepare source input
        assert torch.all(
            all_input_ids.input_ids.eq(self.image_soft_token_id).sum(-1)
            == (
                video_lengths_tensor * self.video_token_scale
                + self.num_extra_video_tokens
            )
        ), "The number of image soft tokens does not match the expected number."

        return {
            "pixel_values": video_tensor,
            "pixel_values_length": video_lengths_tensor,
            "attention_mask": all_input_ids.attention_mask,
            "input_ids": all_input_ids.input_ids,
            "labels": all_label_ids,
            # other useful info
            "names": names,
            "target_text": texts,
            "prompts": prompts,
            "input_text": input_text,
        }
