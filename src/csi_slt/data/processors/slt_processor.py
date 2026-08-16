import hashlib
from dataclasses import dataclass

from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import (
    BatchFeature,
)
from transformers.utils import TensorType
from transformers.processing_utils import ProcessingKwargs, Unpack
import numpy as np
from transformers.tokenization_utils_base import TextInput

from typing import Mapping, Sequence, Union

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


@dataclass(frozen=True)
class _PromptParts:
    """A rendered prompt split around its source-content span."""

    before_source: str
    after_source: str


@dataclass(frozen=True)
class _EncodedPromptParts:
    """Token IDs around a source span, shared by all source paths."""

    before_source: list[int]
    after_source: list[int]


@dataclass(frozen=True)
class _TextPathFeatures:
    """Tokenized inputs for one source-conditioned text path."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    token_type_ids: torch.Tensor
    source_mask: torch.Tensor
    position_ids: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    prompt_token_type_ids: torch.Tensor
    prompt_source_mask: torch.Tensor


def _stable_semantic_id(value: str | int) -> int:
    """Encode a dataset semantic ID as a deterministic signed int64."""
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            raise ValueError("integer semantic IDs must fit in torch.long")
        return value
    if not isinstance(value, str):
        raise TypeError("semantic IDs must be strings or integers")
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=b"csi-slt"
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=True)


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

        # Cache rendered prompts. BOS handling is part of the cache key because
        # callers may use the same processor in both modes.
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

    def _get_rendered_prompt_for_lang(self, lang: str, add_bos_token: bool) -> str:
        """Return a rendered prompt that still contains the source sentinel."""
        cache = self._prompt_cache  # type: ignore[has-type]
        # Keep cache keys JSON-compatible because ProcessorMixin may include
        # processor attributes while building a save_pretrained() config.
        cache_key = f"{lang}:bos={int(add_bos_token)}"
        if cache_key in cache:
            return cache[cache_key]

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

        cache[cache_key] = prompt
        return prompt

    def _get_prompt_parts_for_lang(
        self, lang: str, add_bos_token: bool
    ) -> _PromptParts:
        """Split a rendered prompt around its one source-content sentinel."""
        prompt = self._get_rendered_prompt_for_lang(lang, add_bos_token)
        sentinel_count = prompt.count(self.video_start_token)
        if sentinel_count != 1:
            raise ValueError(
                "A rendered prompt must contain exactly one source sentinel "
                f"{self.video_start_token!r}; found {sentinel_count} for language "
                f"{lang!r}."
            )
        before_source, _, after_source = prompt.partition(self.video_start_token)
        return _PromptParts(before_source, after_source)

    def _encode_text_segments(self, texts: Sequence[str]) -> list[list[int]]:
        """Tokenize text segments without padding or special-token insertion."""
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = encoded.input_ids
        if len(input_ids) != len(texts):
            raise ValueError(
                "Tokenizer returned a different batch size than it received."
            )
        return [list(token_ids) for token_ids in input_ids]

    def _encode_prompt_parts(
        self, prompt_parts: Sequence[_PromptParts]
    ) -> list[_EncodedPromptParts]:
        """Tokenize invariant prompt segments once for reuse by all paths."""
        before_ids = self._encode_text_segments(
            [parts.before_source for parts in prompt_parts]
        )
        after_ids = self._encode_text_segments(
            [parts.after_source for parts in prompt_parts]
        )
        return [
            _EncodedPromptParts(before, after)
            for before, after in zip(before_ids, after_ids, strict=True)
        ]

    @staticmethod
    def _left_pad_sequences(
        sequences: Sequence[Sequence[int]], pad_value: int
    ) -> torch.Tensor:
        """Left-pad integer sequences to a dense ``torch.long`` tensor."""
        if not sequences:
            raise ValueError("Cannot pad an empty batch of token sequences.")
        max_length = max(len(sequence) for sequence in sequences)
        return torch.tensor(
            [
                [pad_value] * (max_length - len(sequence)) + list(sequence)
                for sequence in sequences
            ],
            dtype=torch.long,
        )

    @staticmethod
    def _attention_mask_for(
        sequences: Sequence[Sequence[int]], max_length: int
    ) -> torch.Tensor:
        return torch.tensor(
            [
                [0] * (max_length - len(sequence)) + [1] * len(sequence)
                for sequence in sequences
            ],
            dtype=torch.long,
        )

    def _process_text_path(
        self,
        prompt_parts: Sequence[_EncodedPromptParts],
        source_ids: Sequence[Sequence[int]],
        target_ids: Sequence[Sequence[int]],
        *,
        source_token_type: int,
    ) -> _TextPathFeatures:
        """Build one teacher-forcing path from explicit source boundaries.

        All text has already been tokenized at explicit segment boundaries.
        Concatenating those shared IDs prevents boundary merges from changing
        target length and makes non-source tokens identical across paths.
        """
        batch_size = len(prompt_parts)
        if len(source_ids) != batch_size or len(target_ids) != batch_size:
            raise ValueError(
                "prompt_parts, source_ids, and target_ids must have the same batch size"
            )
        if source_token_type not in (0, 1):
            raise ValueError("source_token_type must be either 0 or 1")

        prompt_ids: list[list[int]] = []
        full_input_ids: list[list[int]] = []
        full_labels: list[list[int]] = []
        prompt_token_types: list[list[int]] = []
        full_token_types: list[list[int]] = []
        prompt_source_masks: list[list[int]] = []
        full_source_masks: list[list[int]] = []
        for parts, source, target in zip(
            prompt_parts, source_ids, target_ids, strict=True
        ):
            before = parts.before_source
            after = parts.after_source
            source = list(source)
            target = list(target)
            prompt = before + source + after
            token_types = (
                [0] * len(before) + [source_token_type] * len(source) + [0] * len(after)
            )
            source_mask = [0] * len(before) + [1] * len(source) + [0] * len(after)
            prompt_ids.append(prompt)
            prompt_token_types.append(token_types)
            prompt_source_masks.append(source_mask)
            full_input_ids.append(prompt + target)
            full_token_types.append(token_types + [0] * len(target))
            full_source_masks.append(source_mask + [0] * len(target))
            full_labels.append([-100] * len(prompt) + target)

        input_ids = self._left_pad_sequences(full_input_ids, self.pad_token_id)
        sequence_length = input_ids.size(1)
        attention_mask = self._attention_mask_for(full_input_ids, sequence_length)
        labels_tensor = self._left_pad_sequences(full_labels, -100)
        token_type_ids = self._left_pad_sequences(full_token_types, 0)
        source_mask = self._left_pad_sequences(full_source_masks, 0)

        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.clamp_(min=0)
        position_ids.masked_fill_(attention_mask == 0, 1)

        prompt_input_ids = self._left_pad_sequences(prompt_ids, self.pad_token_id)
        prompt_length = prompt_input_ids.size(1)
        prompt_attention_mask = self._attention_mask_for(prompt_ids, prompt_length)
        prompt_token_type_ids = self._left_pad_sequences(prompt_token_types, 0)
        prompt_source_mask = self._left_pad_sequences(prompt_source_masks, 0)

        path = _TextPathFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_tensor,
            token_type_ids=token_type_ids,
            source_mask=source_mask,
            position_ids=position_ids,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_token_type_ids=prompt_token_type_ids,
            prompt_source_mask=prompt_source_mask,
        )
        self._validate_text_path(path)
        return path

    @staticmethod
    def _validate_text_path(path: _TextPathFeatures) -> None:
        """Validate invariants shared by every teacher-forcing path."""
        full_shape = path.input_ids.shape
        for name, tensor in (
            ("attention_mask", path.attention_mask),
            ("labels", path.labels),
            ("token_type_ids", path.token_type_ids),
            ("source_mask", path.source_mask),
            ("position_ids", path.position_ids),
        ):
            if tensor.shape != full_shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match "
                    f"input_ids shape {tuple(full_shape)}"
                )

        prompt_shape = path.prompt_input_ids.shape
        for name, tensor in (
            ("prompt_attention_mask", path.prompt_attention_mask),
            ("prompt_token_type_ids", path.prompt_token_type_ids),
            ("prompt_source_mask", path.prompt_source_mask),
        ):
            if tensor.shape != prompt_shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match "
                    f"prompt_input_ids shape {tuple(prompt_shape)}"
                )

        target_mask = path.labels.ne(-100)
        if bool((target_mask & path.attention_mask.eq(0)).any()):
            raise ValueError("Target labels cannot occupy padded input positions.")
        if bool((target_mask & path.source_mask.bool()).any()):
            raise ValueError("Target labels cannot occupy source positions.")
        if not torch.equal(path.labels[target_mask], path.input_ids[target_mask]):
            raise ValueError(
                "Target labels must exactly match their positions in input_ids."
            )
        for name, mask, attention_mask in (
            ("source_mask", path.source_mask, path.attention_mask),
            (
                "prompt_source_mask",
                path.prompt_source_mask,
                path.prompt_attention_mask,
            ),
        ):
            if bool(((mask != 0) & (mask != 1)).any()):
                raise ValueError(f"{name} must be binary.")
            if bool((mask.bool() & attention_mask.eq(0)).any()):
                raise ValueError(f"{name} cannot mark padded positions.")

        for batch_index in range(path.input_ids.size(0)):
            valid_input = path.input_ids[batch_index][
                path.attention_mask[batch_index].bool()
            ].tolist()
            valid_prompt = path.prompt_input_ids[batch_index][
                path.prompt_attention_mask[batch_index].bool()
            ].tolist()
            valid_target = path.labels[batch_index][target_mask[batch_index]].tolist()
            if valid_input != valid_prompt + valid_target:
                raise ValueError(
                    "Each full input must be its prompt followed by the target; "
                    f"mismatch at batch index {batch_index}."
                )

            full_source_mask = path.source_mask[batch_index][
                path.attention_mask[batch_index].bool()
            ].tolist()
            prompt_source_mask = path.prompt_source_mask[batch_index][
                path.prompt_attention_mask[batch_index].bool()
            ].tolist()
            if full_source_mask != prompt_source_mask + [0] * len(valid_target):
                raise ValueError(
                    "Full and prompt source spans do not correspond at batch "
                    f"index {batch_index}."
                )

    def _build_video_sources(self, video_lengths: Sequence[int]) -> list[list[int]]:
        """Construct video source token IDs from processed frame lengths."""
        token_counts: list[int] = []
        for video_length in video_lengths:
            raw_count = (
                int(video_length) * self.video_token_scale + self.num_extra_video_tokens
            )
            if not float(raw_count).is_integer() or raw_count < 0:
                raise ValueError(
                    "Video length and video_token_scale must produce a "
                    f"non-negative integer token count; got {raw_count}."
                )
            token_counts.append(int(raw_count))

        return [[self.video_soft_token_id] * count for count in token_counts]

    @staticmethod
    def _validate_source_content(
        path_name: str,
        path: _TextPathFeatures,
        expected_source_ids: Sequence[Sequence[int]],
    ) -> None:
        """Validate one path's source span against its exact source token IDs."""
        if path.input_ids.size(0) != len(expected_source_ids):
            raise ValueError(
                f"{path_name} source batch size does not match its path batch size."
            )

        for batch_index, expected in enumerate(expected_source_ids):
            expected = list(expected)
            observed = path.input_ids[batch_index][
                path.source_mask[batch_index].bool()
            ].tolist()
            observed_prompt = path.prompt_input_ids[batch_index][
                path.prompt_source_mask[batch_index].bool()
            ].tolist()
            if observed != expected or observed_prompt != expected:
                raise ValueError(
                    f"{path_name} source tokens do not match at batch index "
                    f"{batch_index}: expected {expected}, observed {observed}."
                )

    def _validate_video_path(
        self,
        path: _TextPathFeatures,
        expected_source_ids: Sequence[Sequence[int]],
    ) -> None:
        """Validate invariants specific to the video-conditioned path."""
        self._validate_source_content("video path", path, expected_source_ids)
        expected_token_counts = torch.tensor(
            [len(source) for source in expected_source_ids],
            dtype=torch.long,
            device=path.input_ids.device,
        )
        observed_full_counts = path.input_ids.eq(self.video_soft_token_id).sum(-1)
        observed_prompt_counts = path.prompt_input_ids.eq(self.video_soft_token_id).sum(
            -1
        )
        source_type_counts = path.token_type_ids.sum(-1)
        prompt_source_type_counts = path.prompt_token_type_ids.sum(-1)

        for name, observed in (
            ("input_ids", observed_full_counts),
            ("prompt_input_ids", observed_prompt_counts),
            ("token_type_ids", source_type_counts),
            ("prompt_token_type_ids", prompt_source_type_counts),
        ):
            if not torch.equal(observed, expected_token_counts):
                raise ValueError(
                    f"Video source length in {name} does not match the expected "
                    f"token counts: observed {observed.tolist()}, expected "
                    f"{expected_token_counts.tolist()}."
                )

    def _encode_prompts_and_targets(
        self,
        text: Sequence[str],
        src_lang: Sequence[str],
        *,
        add_bos_token: bool,
        add_eos_token: bool,
    ) -> tuple[list[_EncodedPromptParts], list[list[int]], list[int]]:
        """Encode shared prompt boundaries and target IDs for all source paths."""
        prompt_parts: list[_PromptParts] = []
        labels: list[str] = []
        language_ids: list[int] = []
        for target_text, language in zip(text, src_lang, strict=True):
            prompt_parts.append(
                self._get_prompt_parts_for_lang(language, add_bos_token)
            )
            labels.append(
                target_text + self.tokenizer.eos_token if add_eos_token else target_text
            )
            language_ids.append(LANGUAGE_MAP[language])

        return (
            self._encode_prompt_parts(prompt_parts),
            self._encode_text_segments(labels),
            language_ids,
        )

    @staticmethod
    def _validate_and_batch_inputs(
        videos: Union[list[np.ndarray], np.ndarray],
        text: Union[list[TextInput], TextInput],
        src_lang: Union[list[str], str],
        pseudo_gloss: Union[list[str], str] | None,
        semantic_ids: Union[list[str | int], str, int] | None,
    ) -> tuple[
        list[np.ndarray],
        list[TextInput],
        list[str],
        list[str] | None,
        list[str | int] | None,
    ]:
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

        if isinstance(semantic_ids, (str, int)):
            semantic_ids = [semantic_ids]
        elif isinstance(semantic_ids, (list, tuple)):
            semantic_ids = list(semantic_ids)
        elif semantic_ids is not None:
            raise TypeError("semantic_ids must be a string, integer, list, or None")

        batch_size = len(videos)
        if batch_size == 0:
            raise ValueError("videos must contain at least one sample")

        batch_fields = {"text": text, "src_lang": src_lang}
        if pseudo_gloss is not None:
            batch_fields["pseudo_gloss"] = pseudo_gloss
        if semantic_ids is not None:
            batch_fields["semantic_ids"] = semantic_ids
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

        if semantic_ids is not None:
            for value in semantic_ids:
                _stable_semantic_id(value)

        return videos, text, src_lang, pseudo_gloss, semantic_ids

    def __call__(
        self,
        videos: Union[list[np.ndarray], np.ndarray],
        text: Union[list[TextInput], TextInput],
        src_lang: Union[list[str], str],
        pseudo_gloss: Union[list[str], str] | None = None,
        semantic_ids: Union[list[str | int], str, int] | None = None,
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

        (
            videos,
            text,
            src_lang,
            pseudo_gloss,
            semantic_ids,
        ) = self._validate_and_batch_inputs(
            videos,
            text,
            src_lang,
            pseudo_gloss,
            semantic_ids,
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

        # Build the video-conditioned text path.
        encoded_prompt_parts, target_ids, language_ids = (
            self._encode_prompts_and_targets(
                text,
                src_lang,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
            )
        )
        video_source_ids = self._build_video_sources(video_lengths)
        video_path = self._process_text_path(
            encoded_prompt_parts,
            video_source_ids,
            target_ids,
            source_token_type=1,
        )

        # Validate the behavior specific to the video source path.
        self._validate_video_path(video_path, video_source_ids)

        # Keep the standalone pseudo-gloss fields for batch inspection and
        # compatibility.
        pseudo_gloss_ids = None
        pseudo_gloss_attention_mask = None
        if pseudo_gloss is not None:
            unpadded_pseudo_gloss_ids = self._encode_text_segments(pseudo_gloss)
            pseudo_gloss_ids = self._left_pad_sequences(
                unpadded_pseudo_gloss_ids, self.pad_token_id
            )
            pseudo_gloss_attention_mask = self._attention_mask_for(
                unpadded_pseudo_gloss_ids, pseudo_gloss_ids.size(1)
            )

        data = {
            "pixel_values": video_batch_features.pixel_values,
            "pixel_values_length": video_lengths_tensor,
            "attention_mask": video_path.attention_mask,
            "input_ids": video_path.input_ids,
            "labels": video_path.labels,
            "token_type_ids": video_path.token_type_ids,
            "lang_ids": torch.tensor(language_ids).long(),
            "position_ids": video_path.position_ids,
        }

        if not training:
            data["generation_input_ids"] = video_path.prompt_input_ids
            data["generation_attention_mask"] = video_path.prompt_attention_mask
            data["generation_token_type_ids"] = video_path.prompt_token_type_ids
        if pseudo_gloss_ids is not None:
            data["pseudo_gloss_input_ids"] = pseudo_gloss_ids
            data["pseudo_gloss_attention_mask"] = pseudo_gloss_attention_mask
        if semantic_ids is not None:
            data["semantic_ids"] = torch.tensor(
                [_stable_semantic_id(value) for value in semantic_ids],
                dtype=torch.long,
            )

        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)
