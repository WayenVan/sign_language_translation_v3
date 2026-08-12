from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import (
    BatchFeature,
)
from transformers.utils import TensorType
from transformers.processing_utils import ProcessingKwargs, Unpack
import numpy as np
from transformers.tokenization_utils_base import TextInput

from typing import Mapping, Union

from csi_slt.data.processors.sign_video_processor import (
    SignVideoKwargs,
    SignVideoProcessor,
)

import torch
import json
import os
from jinja2 import Environment, StrictUndefined
from csi_slt.constants import LANGUAGE_MAP, LANGUAGE_NAME_MAP


class SignTranslationProcessingKwargs(ProcessingKwargs, total=False):
    videos_kwargs: SignVideoKwargs
    _defaults = {}


class SignTranslationProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]

    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"
    _auto_class = "AutoProcessor"

    def __init__(
        self,
        video_processor: SignVideoProcessor,
        tokenizer,
        chat_template=None,
        prompt_paths_per_language: Mapping[str, str] | None = None,
        prompt_templates_per_language: Mapping[str, str] | None = None,
        video_soft_token="<|video_pad|>",
        video_start_token="<|vision_start|>",
        video_token_scale=0.5,
        num_extra_video_tokens=2,  # for video start and end tokens
        position_shift_range=(0, 20),
    ):
        if (
            prompt_paths_per_language is not None
            and prompt_templates_per_language is not None
        ):
            raise ValueError(
                "Specify either prompt_paths_per_language or "
                "prompt_templates_per_language, not both."
            )

        self.video_soft_token = video_soft_token
        self.video_start_token = video_start_token
        self.video_token_scale = video_token_scale
        self.num_extra_video_tokens = num_extra_video_tokens
        self.position_shift_range = position_shift_range

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

        if self.video_soft_token_id == tokenizer.unk_token_id or tokenizer.encode(
            self.video_soft_token, add_special_tokens=False
        ) != [self.video_soft_token_id]:
            raise ValueError(
                f"{self.video_soft_token!r} is not a valid single tokenizer token."
            )

        # Keep template sources instead of Jinja Template instances. ProcessorMixin
        # deep-copies __dict__ during save_pretrained(), while Template objects cannot
        # be deep-copied. The sources are JSON serializable and restore cleanly.
        if prompt_templates_per_language is not None:
            self.prompt_templates_per_language = dict(prompt_templates_per_language)
        else:
            self.prompt_templates_per_language = {
                lang: self._read_prompt_file(path)
                for lang, path in (prompt_paths_per_language or {}).items()
            }

        # Cache for rendered prompts (per language), avoiding redundant
        # template.render() and apply_chat_template() calls within a batch.
        self._prompt_cache: dict[str, str] = {}

    @staticmethod
    def _read_prompt_file(path: str) -> str:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Support both plain text and JSON-encapsulated prompt templates.
        # If the file is valid JSON containing a string, use that string as the template.
        # Otherwise, treat the raw file content as the Jinja2 template.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, str):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return content

    def inject_images(self, prompt: str, n: int) -> str:
        sentinel = self.video_start_token
        replacement = self.video_soft_token * n
        return prompt.replace(sentinel, replacement)

    def _get_rendered_prompt_for_lang(self, lang: str, add_bos_token) -> str:
        """Return the fully-formed (but not yet tokenized) prompt string for a given language,
        with chat template applied and image placeholders injected.  The result is cached per
        language because it does not depend on the actual video length."""
        cache = self._prompt_cache  # type: ignore[has-type]
        if lang in cache:
            return cache[lang]

        template_source = self.prompt_templates_per_language.get(lang)
        if template_source is None:
            raise ValueError(f"No prompt template found for language: {lang}")
        language_name = LANGUAGE_NAME_MAP.get(lang)
        if language_name is None:
            raise ValueError(f"Language '{lang}' not found in LANGUAGE_MAP.")

        rendered = (
            Environment(undefined=StrictUndefined)
            .from_string(template_source)
            .render(
                video_start_token=self.video_start_token,
                language=language_name,
            )
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

        if add_bos_token:
            prompt = self.tokenizer.bos_token + prompt

        cache[lang] = prompt
        return prompt

    @staticmethod
    def _validate_and_batch_inputs(
        videos: Union[list[np.ndarray], np.ndarray],
        text: Union[list[TextInput], TextInput],
        src_lang: Union[list[str], str],
        pseudo_gloss: Union[list[str], str] | None,
    ) -> tuple[list[np.ndarray], list[TextInput], list[str], list[str] | None]:
        """Validate processor inputs and normalize single samples to batches."""

        if isinstance(videos, np.ndarray):
            videos = [videos]
        elif isinstance(videos, (list, tuple)):
            videos = list(videos)
        else:
            raise TypeError("videos must be a numpy array or a list of numpy arrays")

        if isinstance(text, str):
            text = [text]
        elif isinstance(text, (list, tuple)):
            text = list(text)
        else:
            raise TypeError("text must be a string or a list of strings")

        if isinstance(src_lang, str):
            src_lang = [src_lang]
        elif isinstance(src_lang, (list, tuple)):
            src_lang = list(src_lang)
        else:
            raise TypeError("src_lang must be a string or a list of strings")

        if isinstance(pseudo_gloss, str):
            pseudo_gloss = [pseudo_gloss]
        elif isinstance(pseudo_gloss, (list, tuple)):
            pseudo_gloss = list(pseudo_gloss)
        elif pseudo_gloss is not None:
            raise TypeError("pseudo_gloss must be a string, a list of strings, or None")

        batch_size = len(videos)
        if batch_size == 0:
            raise ValueError("videos must contain at least one sample")

        batch_fields = {"text": text, "src_lang": src_lang}
        if pseudo_gloss is not None:
            batch_fields["pseudo_gloss"] = pseudo_gloss
        for name, values in batch_fields.items():
            if len(values) != batch_size:
                raise ValueError(
                    f"{name} batch size ({len(values)}) does not match "
                    f"videos batch size ({batch_size})"
                )

        if not all(isinstance(video, np.ndarray) for video in videos):
            raise TypeError("every item in videos must be a numpy array")
        if not all(isinstance(value, str) for value in text):
            raise TypeError("every item in text must be a string")
        if not all(isinstance(value, str) for value in src_lang):
            raise TypeError("every item in src_lang must be a string")
        if pseudo_gloss is not None and not all(
            isinstance(value, str) for value in pseudo_gloss
        ):
            raise TypeError("every item in pseudo_gloss must be a string")

        return videos, text, src_lang, pseudo_gloss

    def __call__(
        self,
        videos: Union[list[np.ndarray], np.ndarray],
        text: Union[list[TextInput], TextInput],
        src_lang: Union[list[str], str],
        pseudo_gloss: Union[list[str], str] | None = None,
        training: bool = True,
        add_bos_token: bool = False,
        add_eos_token: bool = True,
        **kwargs: Unpack[SignTranslationProcessingKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            SignTranslationProcessingKwargs,
            tokenizer_init_kwargs=None,
            **kwargs,
        )
        videos_kwargs = output_kwargs["videos_kwargs"]

        videos, text, src_lang, pseudo_gloss = self._validate_and_batch_inputs(
            videos,
            text,
            src_lang,
            pseudo_gloss,
        )

        if training:
            videos_kwargs["training"] = True
        else:
            videos_kwargs["training"] = False

        video_batch_features = self.video_processor(
            videos,
            **videos_kwargs,
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

            prompt = self._get_rendered_prompt_for_lang(lang, add_bos_token)

            # inject image soft tokens according to video length
            if self.video_start_token in prompt:
                prompt = self.inject_images(
                    prompt,
                    int(video_lengths[i] * self.video_token_scale)
                    + self.num_extra_video_tokens,
                )

            label = t
            if add_eos_token:
                label = label + self.tokenizer.eos_token

            prompts.append(prompt)
            labels.append(label)
            language_ids.append(LANGUAGE_MAP[lang])

            input_text = prompt + label if training else prompt
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
        if training:  # WARN: ONLY train need position ids, we don't need it when generating sequence, there is a bug
            pos_ids = inputs_pt.attention_mask.cumsum(-1) - 1
            pos_ids = pos_ids.clamp(min=0)
            pos_ids = torch.where(inputs_pt.attention_mask == 0, 1, pos_ids)

        # handle pseudo glosses, if present
        pseudo_gloss_pt = None
        if pseudo_gloss is not None:
            pseudo_gloss_pt = self.tokenizer(
                pseudo_gloss,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )

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
        if pseudo_gloss_pt is not None:
            data["pseudo_gloss_input_ids"] = pseudo_gloss_pt.input_ids
            data["pseudo_gloss_attention_mask"] = pseudo_gloss_pt.attention_mask

        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)
