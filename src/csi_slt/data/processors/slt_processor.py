from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
from transformers.image_processing_utils import (
    ImageProcessingMixin,
    BaseImageProcessor,
    BatchFeature,
)
from transformers.utils import TensorType, filter_out_non_signature_kwargs
import numpy as np
from transformers.tokenization_utils_base import TextInput

from typing import Union, Optional

from transformers import AutoVideoProcessor
from enum import Enum
import torch


class Language(Enum):
    DE = "German"
    EN = "English"
    ZH = "Chinese"


class SignTranslationProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"
    _auto_class = "AutoProcessor"

    def __init__(
        self,
        video_processor,
        tokenizer,
        chat_template=None,
        VIDEO_SOFT_TOKEN="<unused0>",
        VIDEO_START_TOKEN="<unused1>",
        video_padding_to_multiple_of=4,
        video_token_scale=0.5,
        num_extra_video_tokens=2,  # for video start and end tokens
        mode="train",
        **kwargs,
    ):
        self.VIDEO_SOFT_TOKEN = VIDEO_SOFT_TOKEN
        self.VIDEO_START_TOKEN = VIDEO_START_TOKEN
        self.video_padding_to_multiple_of = video_padding_to_multiple_of
        self.mode = mode
        self.video_token_scale = video_token_scale
        self.num_extra_video_tokens = num_extra_video_tokens
        super().__init__(
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template  # WARN: really needed?

        self.image_soft_token_id = self.tokenizer.convert_tokens_to_ids(
            self.VIDEO_SOFT_TOKEN
        )

    def inject_images(self, prompt: str, n: int) -> str:
        sentinel = self.VIDEO_START_TOKEN
        replacement = self.VIDEO_SOFT_TOKEN * n
        return prompt.replace(sentinel, replacement)

    def __call__(
        self,
        videos: Union[list[np.ndarray], np.ndarray],
        text: Union[list[TextInput], TextInput],
        src_lang: Union[list[str], str],
    ):
        if isinstance(text, str):
            text = [text]
        if isinstance(src_lang, str):
            src_lang = [src_lang]

        if self.mode == "train":
            video_batch_features = self.video_processor(
                videos,
                training=True,
                padding_to_multiple_of=self.video_padding_to_multiple_of,
            )
        else:
            video_batch_features = self.video_processor(
                videos,
                training=False,
                padding_to_multiple_of=self.video_padding_to_multiple_of,
            )

        video_lengths = video_batch_features.pixel_values_lengths.cpu().numpy()
        video_lengths_tensor = video_batch_features.pixel_values_lengths

        all_input_ids = []
        all_label_ids = []
        for i, t in enumerate(text):
            if src_lang[i] == "en":
                message = [{"role": "user", "language": Language.EN.value}]
            elif src_lang[i] == "de":
                message = [{"role": "user", "language": Language.DE.value}]
            elif src_lang[i] == "zh":
                message = [{"role": "user", "language": Language.ZH.value}]
            else:
                raise ValueError(f"Unsupported language: {src_lang[i]}")

            prompt = self.apply_chat_template(
                message,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            # inject images if needed
            if self.VIDEO_START_TOKEN in prompt:
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
                assert len(input_ids) == len(label_ids)
            else:
                # For inference, we only use the prompt
                input_ids = prompt_ids

            all_input_ids.append(input_ids)
            all_label_ids.append(label_ids)

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

        data = {
            "pixel_values": video_batch_features.pixel_values,
            "pixel_values_length": video_lengths_tensor,
            "attention_mask": all_input_ids.attention_mask,
            "input_ids": all_input_ids.input_ids,
            "labels": all_label_ids,
        }
        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)
