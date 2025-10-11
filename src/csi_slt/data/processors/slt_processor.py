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
        eos_token="<end_of_turn>",
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
        self.eos_token = eos_token
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

        prompts = []
        labels = []
        input_texts = []
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

            label = t + self.eos_token
            prompts.append(prompt)
            labels.append(label)

            input_text = prompt + label if self.mode == "train" else prompt
            input_texts.append(input_text)

        # pad on the left
        self.tokenizer.padding_side = "left"

        inputs_pt = self.tokenizer(
            input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )

        max_length = inputs_pt.input_ids.size(1)

        labels_pt = self.tokenizer(
            labels,
            add_special_tokens=False,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        labels_pt.input_ids[labels_pt.input_ids == self.tokenizer.pad_token_id] = -100

        # Prepare source input
        assert torch.all(
            inputs_pt.input_ids.eq(self.image_soft_token_id).sum(-1)
            == (
                video_lengths_tensor * self.video_token_scale
                + self.num_extra_video_tokens
            )
        ), "The number of image soft tokens does not match the expected number."

        # calcuate the postional ids
        pos_ids = inputs_pt.attention_mask.cumsum(-1) - 1
        pos_ids = pos_ids.clamp(min=0)

        data = {
            "pixel_values": video_batch_features.pixel_values,
            "pixel_values_length": video_lengths_tensor,
            "attention_mask": inputs_pt.attention_mask,
            "position_ids": pos_ids,
            "input_ids": inputs_pt.input_ids,
            "labels": labels_pt.input_ids,
        }
        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)
