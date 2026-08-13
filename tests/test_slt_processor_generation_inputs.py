from types import SimpleNamespace

import numpy as np
import torch

from csi_slt.data.processors.slt_processor import SignTranslationProcessor


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(
        self,
        video_token: str,
        video_token_id: int,
        eos_token_id: int = 2,
    ) -> None:
        self.video_token = video_token
        self.video_token_id = video_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = "right"

    def _encode(self, text: str) -> list[int]:
        ids = []
        index = 0
        while index < len(text):
            if text.startswith(self.video_token, index):
                ids.append(self.video_token_id)
                index += len(self.video_token)
            elif text.startswith(self.eos_token, index):
                ids.append(self.eos_token_id)
                index += len(self.eos_token)
            else:
                ids.append(10 + ord(text[index]))
                index += 1
        return ids

    def __call__(
        self,
        texts,
        *,
        padding,
        return_tensors,
        add_special_tokens,
        max_length=None,
    ):
        del return_tensors, add_special_tokens
        encoded = [self._encode(text) for text in texts]
        target_length = max_length if padding == "max_length" else max(map(len, encoded))

        input_ids = []
        attention_mask = []
        for ids in encoded:
            padding_length = target_length - len(ids)
            if padding_length < 0:
                raise ValueError("fake tokenizer input exceeds max_length")
            pad = [self.pad_token_id] * padding_length
            mask_pad = [0] * padding_length
            if self.padding_side == "left":
                input_ids.append(pad + ids)
                attention_mask.append(mask_pad + [1] * len(ids))
            else:
                input_ids.append(ids + pad)
                attention_mask.append([1] * len(ids) + mask_pad)

        return SimpleNamespace(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )


class _FakeVideoProcessor:
    def __init__(self) -> None:
        self.training_values = []

    def __call__(self, videos, **kwargs):
        self.training_values.append(kwargs["training"])
        lengths = torch.tensor([len(video) for video in videos], dtype=torch.long)
        return SimpleNamespace(
            pixel_values=torch.zeros(int(lengths.sum()), 3, 1, 1),
            pixel_values_lengths=lengths,
        )


def _make_processor() -> SignTranslationProcessor:
    processor = object.__new__(SignTranslationProcessor)
    processor.video_soft_token = "<video>"
    processor.video_start_token = "<video-start>"
    processor.video_soft_token_id = 7
    processor.video_token_scale = 0.5
    processor.num_extra_video_tokens = 2
    processor.tokenizer = _FakeTokenizer("<video>", 7)
    processor.video_processor = _FakeVideoProcessor()
    processor._merge_kwargs = lambda *args, **kwargs: {"videos_kwargs": {}}
    processor._get_rendered_prompt_for_lang = (
        lambda lang, add_bos_token: f"prompt-{lang}:<video-start>:"
    )
    return processor


def test_eval_keeps_supervised_inputs_and_adds_prompt_only_generation_fields():
    videos = [np.zeros((4, 1, 1, 3), dtype=np.uint8)]
    common_kwargs = {
        "videos": videos,
        "text": ["answer"],
        "src_lang": ["en"],
    }
    train_processor = _make_processor()
    eval_processor = _make_processor()

    train_output = train_processor(training=True, **common_kwargs)
    eval_output = eval_processor(training=False, **common_kwargs)

    for name in (
        "input_ids",
        "attention_mask",
        "labels",
        "token_type_ids",
        "position_ids",
    ):
        assert torch.equal(train_output[name], eval_output[name])

    generation_ids = eval_output["generation_input_ids"]
    generation_length = generation_ids.shape[1]
    assert generation_length < eval_output["input_ids"].shape[1]
    assert torch.equal(eval_output["input_ids"][:, :generation_length], generation_ids)
    assert torch.equal(
        eval_output["generation_attention_mask"],
        torch.ones_like(generation_ids),
    )
    assert torch.equal(
        eval_output["generation_token_type_ids"],
        generation_ids.eq(eval_processor.video_soft_token_id).long(),
    )
    assert "generation_position_ids" not in eval_output
    assert "generation_labels" not in eval_output
    assert not any(name.startswith("generation_") for name in train_output)
    assert train_processor.video_processor.training_values == [True]
    assert eval_processor.video_processor.training_values == [False]


def test_real_eos_label_is_not_masked_when_it_shares_the_padding_id():
    processor = _make_processor()
    processor.tokenizer.eos_token_id = processor.tokenizer.pad_token_id

    output = processor(
        videos=[np.zeros((4, 1, 1, 3), dtype=np.uint8)],
        text=["answer"],
        src_lang=["en"],
        training=True,
    )

    assert output["labels"][0, -1].item() == processor.tokenizer.eos_token_id
    assert output["labels"][0, -1].item() != -100


def test_processor_encodes_dataset_semantic_ids_deterministically():
    processor = _make_processor()
    kwargs = {
        "videos": [
            np.zeros((4, 1, 1, 3), dtype=np.uint8),
            np.zeros((4, 1, 1, 3), dtype=np.uint8),
        ],
        "text": ["first", "second"],
        "src_lang": ["en", "en"],
        "pseudo_gloss": ["A", "B"],
        "semantic_ids": ["shared-name", "shared-name"],
        "training": True,
    }

    first = processor(**kwargs)["semantic_ids"]
    second = processor(**kwargs)["semantic_ids"]

    assert first.dtype == torch.long
    assert first.shape == (2,)
    assert first[0].item() == first[1].item()
    assert torch.equal(first, second)
