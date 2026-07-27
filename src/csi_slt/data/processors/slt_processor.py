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
import json
import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
from csi_slt.constants import LANGUAGE_MAP, LANGUAGE_NAME_MAP


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
        prompt_paths_per_language: dict[str, str] = {},
        video_soft_token="<|video_pad|>",
        video_start_token="<|vision_start|>",
        video_padding_to_multiple_of=4,
        video_token_scale=0.5,
        num_extra_video_tokens=2,  # for video start and end tokens
        add_bos_token=False,
        add_eos_token=True,
        mode="train",
        position_shift_range=(0, 20),
        **kwargs,
    ):
        self.video_soft_token = video_soft_token
        self.video_start_token = video_start_token
        self.video_padding_to_multiple_of = video_padding_to_multiple_of
        self.mode = mode
        self.video_token_scale = video_token_scale
        self.num_extra_video_tokens = num_extra_video_tokens
        self.position_shift_range = position_shift_range
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.pad_token_id = tokenizer.pad_token_id

        if chat_template is None:
            chat_template = tokenizer.chat_template
        else:
            tokenizer.chat_template = chat_template

        super().__init__(
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

        self.video_soft_token_id = self.tokenizer.convert_tokens_to_ids(
            self.video_soft_token
        )

        # Load prompt templates from JSON files using Jinja2 Environment
        self.prompt_templates: dict[str, Template] = {
            lang: self._parse_prompt_file(path)
            for lang, path in prompt_paths_per_language.items()
        }

        # Cache for rendered prompts (per language), avoiding redundant
        # template.render() and apply_chat_template() calls within a batch.
        self._prompt_cache: dict[str, str] = {}

    @staticmethod
    def _parse_prompt_file(path: str) -> Template:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r") as f:
            content = f.read()

        env = Environment(undefined=StrictUndefined)

        # Support both plain text and JSON-encapsulated prompt templates.
        # If the file is valid JSON containing a string, use that string as the template.
        # Otherwise, treat the raw file content as the Jinja2 template.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, str):
                return env.from_string(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

        return env.from_string(content)

    def inject_images(self, prompt: str, n: int) -> str:
        sentinel = self.video_start_token
        replacement = self.video_soft_token * n
        return prompt.replace(sentinel, replacement)

    def _get_rendered_prompt_for_lang(self, lang: str) -> str:
        """Return the fully-formed (but not yet tokenized) prompt string for a given language,
        with chat template applied and image placeholders injected.  The result is cached per
        language because it does not depend on the actual video length."""
        cache = self._prompt_cache  # type: ignore[has-type]
        if lang in cache:
            return cache[lang]

        template = self.prompt_templates.get(lang)
        if template is None:
            raise ValueError(f"No prompt template found for language: {lang}")
        language_name = LANGUAGE_NAME_MAP.get(lang)
        if language_name is None:
            raise ValueError(f"Language '{lang}' not found in LANGUAGE_MAP.")

        rendered = template.render(
            video_start_token=self.video_start_token,
            language=language_name,
        )
        message = [
            {
                "role": "user",
                "content": rendered,
            }
        ]

        prompt = self.apply_chat_template(
            message,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )

        if self.add_bos_token:
            prompt = self.tokenizer.bos_token + prompt

        cache[lang] = prompt
        return prompt

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

        # NOTE: convert text to prompts and labels ids

        prompts: list[str] = []
        labels: list[str] = []
        input_texts: list[str] = []
        language_ids: list[int] = []
        for i, t in enumerate(text):
            lang = src_lang[i]  # each sample has exactly one language

            prompt = self._get_rendered_prompt_for_lang(lang)

            # inject image soft tokens according to video length
            if self.video_start_token in prompt:
                prompt = self.inject_images(
                    prompt,
                    int(video_lengths[i] * self.video_token_scale)
                    + self.num_extra_video_tokens,
                )

            label = t
            if self.add_eos_token:
                label = label + self.tokenizer.eos_token

            prompts.append(prompt)
            labels.append(label)
            language_ids.append(LANGUAGE_MAP[lang])

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
            inputs_pt.input_ids.eq(self.video_soft_token_id).sum(-1)
            == (
                video_lengths_tensor * self.video_token_scale
                + self.num_extra_video_tokens
            )
        ), "The number of image soft tokens does not match the expected number."

        # calcuate the postional ids
        pos_ids = None
        if (
            self.mode == "train"
        ):  # WARN: ONLY train need position ids, we don't need it when generating sequence, there is a bug
            pos_ids = inputs_pt.attention_mask.cumsum(-1) - 1
            pos_ids = pos_ids.clamp(min=0)
            pos_ids = torch.where(inputs_pt.attention_mask == 0, 1, pos_ids)
            # pos_ids = self.position_augmentation(pos_ids, inputs_pt.attention_mask)
            # pos_ids = torch.arange(
            #     0,
            #     inputs_pt.input_ids.shape[1],
            #     device=inputs_pt.input_ids.device,
            # ).unsqueeze(0)

        data = {
            "pixel_values": video_batch_features.pixel_values,
            "pixel_values_length": video_lengths_tensor,
            "attention_mask": inputs_pt.attention_mask,
            "input_ids": inputs_pt.input_ids,
            "labels": labels_pt.input_ids,
            "token_type_ids": (inputs_pt.input_ids == self.video_soft_token_id).long(),
            "lang_ids": torch.tensor(language_ids).long(),
        }

        if pos_ids is not None:
            data["position_ids"] = pos_ids

        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)

    def position_augmentation(self, position_ids, attention_mask):
        # 50% 进行随机偏移
        position_ids = self.random_shift_position_ids(
            position_ids, attention_mask, shift_range=self.position_shift_range, p=0.5
        )
        # # 20% 进行缩放
        # if torch.rand(1).item() < 0.2:
        #     position_ids = self..random_scale_position_ids(
        #         position_ids, attention_mask, scale_range=(0.95, 1.05)
        #     )
        return position_ids

    @staticmethod
    def random_shift_position_ids(
        position_ids, attention_mask, p=1.0, shift_range=(-4, 4), training=True
    ):
        """
        对 position_ids 增加随机偏移，模拟不同的起点位置。
        Args:
            position_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            shift_range: 偏移范围 (min, max)
            training: 是否在训练阶段启用
        """
        if not training:
            return position_ids  # 推理时不增强
        # 为每个 batch 生成一个独立的偏移量
        batch_size = position_ids.size(0)
        shifts = torch.randint(
            shift_range[0],
            shift_range[1] + 1,
            (batch_size,),
            device=position_ids.device,
        )
        # 广播到整行
        shifts = shifts.unsqueeze(1).expand_as(position_ids)

        # 以概率 p 应用偏移
        probs = torch.full((batch_size, 1), p, device=position_ids.device)
        mask = torch.bernoulli(probs).long().expand_as(position_ids)
        shifts = shifts * mask

        # 只对非 padding token 加偏移（防止 pad token pos 出界）
        augmented_pos = position_ids + shifts * attention_mask
        return augmented_pos

    @staticmethod
    def random_scale_position_ids(
        position_ids, attention_mask, scale_range=(0.9, 1.1), training=True
    ):
        """
        对 position_ids 做随机缩放，模拟序列密度变化。
        """
        if not training:
            return position_ids
        batch_size = position_ids.size(0)
        scales = torch.empty(batch_size, device=position_ids.device).uniform_(
            *scale_range
        )
        scales = scales.unsqueeze(1).expand_as(position_ids)
        # 只缩放有效部分
        scaled_pos = (position_ids.float() * scales) * attention_mask
        return scaled_pos.long()

    @staticmethod
    def jitter_position_ids(position_ids, attention_mask, noise_std=0.5, training=True):
        if not training:
            return position_ids
        noise = torch.randn_like(position_ids.float()) * noise_std
        noisy_pos = position_ids.float() + noise * attention_mask
        return noisy_pos.long()
