from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
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
        add_special_tokens,
        return_tensors=None,
        return_attention_mask=True,
        max_length=None,
    ):
        del add_special_tokens
        encoded = [self._encode(text) for text in texts]
        if padding is False:
            return SimpleNamespace(input_ids=encoded)

        target_length = (
            max_length if padding == "max_length" else max(map(len, encoded))
        )

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

        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return SimpleNamespace(**result)


class _BoundaryMergingFakeTokenizer(_FakeTokenizer):
    """A tokenizer whose output changes when prompt and label are joined."""

    def _encode(self, text: str) -> list[int]:
        ids = []
        index = 0
        while index < len(text):
            if text.startswith(":a", index):
                ids.append(999)
                index += 2
            elif text.startswith(self.video_token, index):
                ids.append(self.video_token_id)
                index += len(self.video_token)
            elif text.startswith(self.eos_token, index):
                ids.append(self.eos_token_id)
                index += len(self.eos_token)
            else:
                ids.append(10 + ord(text[index]))
                index += 1
        return ids


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
    processor.pad_token_id = processor.tokenizer.pad_token_id
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


def test_labels_are_copied_from_the_actual_target_suffix_without_boundary_merges():
    processor = _make_processor()
    processor.tokenizer = _BoundaryMergingFakeTokenizer("<video>", 7)
    processor.pad_token_id = processor.tokenizer.pad_token_id

    output = processor(
        videos=[np.zeros((4, 1, 1, 3), dtype=np.uint8)],
        text=["answer"],
        src_lang=["en"],
        training=True,
    )

    valid_labels = output["labels"][0][output["labels"][0] != -100]
    assert torch.equal(valid_labels, output["input_ids"][0, -len(valid_labels) :])
    assert valid_labels.tolist() == processor.tokenizer._encode("answer<eos>")
    assert 999 in processor.tokenizer._encode(
        "prompt-en:" + "<video>" * 4 + ":answer<eos>"
    )
    assert 999 not in output["input_ids"]


def test_variable_length_batch_keeps_each_label_aligned_with_its_input_suffix():
    processor = _make_processor()
    output = processor(
        videos=[
            np.zeros((4, 1, 1, 3), dtype=np.uint8),
            np.zeros((8, 1, 1, 3), dtype=np.uint8),
        ],
        text=["short", "a considerably longer answer"],
        src_lang=["en", "en"],
        training=True,
    )

    assert output["labels"].shape == output["input_ids"].shape
    for input_ids, labels in zip(output["input_ids"], output["labels"], strict=True):
        target_mask = labels != -100
        assert torch.equal(labels[target_mask], input_ids[target_mask])
        assert target_mask.sum() > 0


def test_rendered_prompt_requires_exactly_one_source_sentinel():
    processor = _make_processor()
    processor._get_rendered_prompt_for_lang = (
        lambda lang, add_bos_token: f"prompt-{lang}-without-source"
    )

    with pytest.raises(ValueError, match="exactly one source sentinel"):
        processor(
            videos=[np.zeros((4, 1, 1, 3), dtype=np.uint8)],
            text=["answer"],
            src_lang=["en"],
            training=True,
        )


def test_training_builds_corresponding_pseudo_gloss_and_empty_teacher_paths():
    processor = _make_processor()
    output = processor(
        videos=[np.zeros((4, 1, 1, 3), dtype=np.uint8)],
        text=["answer"],
        src_lang=["en"],
        pseudo_gloss=["GLOSS"],
        training=True,
    )

    teacher_prefixes = ("pseudo_gloss_teacher", "empty_source_teacher")
    for prefix in teacher_prefixes:
        for suffix in ("input_ids", "attention_mask", "position_ids", "labels"):
            assert f"{prefix}_{suffix}" in output
        assert output[f"{prefix}_input_ids"].shape == output[f"{prefix}_labels"].shape

    before = processor.tokenizer._encode("prompt-en:")
    after = processor.tokenizer._encode(":")
    target = processor.tokenizer._encode("answer<eos>")
    pseudo_source = processor.tokenizer._encode("GLOSS")

    assert output["input_ids"][0].tolist() == before + [7] * 4 + after + target
    assert output["pseudo_gloss_teacher_input_ids"][0].tolist() == (
        before + pseudo_source + after + target
    )
    assert output["empty_source_teacher_input_ids"][0].tolist() == (
        before + after + target
    )

    expected_target = torch.tensor(target)
    for labels in (
        output["labels"],
        output["pseudo_gloss_teacher_labels"],
        output["empty_source_teacher_labels"],
    ):
        valid_labels = labels[0][labels[0] != -100]
        assert torch.equal(valid_labels, expected_target)


def test_text_only_dsid_processing_matches_training_paths_without_video_io():
    processor = _make_processor()
    common = {
        "text": ["short", "a longer answer"],
        "src_lang": ["en", "de"],
        "pseudo_gloss": ["ONE", "TWO THREE"],
    }
    training_output = processor(
        videos=[
            np.zeros((4, 1, 1, 3), dtype=np.uint8),
            np.zeros((8, 1, 1, 3), dtype=np.uint8),
        ],
        training=True,
        **common,
    )
    video_calls_before = list(processor.video_processor.training_values)

    teacher_output = processor.process_dsid_teacher_paths(**common)

    for prefix in ("pseudo_gloss_teacher", "empty_source_teacher"):
        for suffix in ("input_ids", "attention_mask", "position_ids", "labels"):
            name = f"{prefix}_{suffix}"
            assert torch.equal(teacher_output[name], training_output[name])
    assert processor.video_processor.training_values == video_calls_before


def test_empty_pseudo_gloss_and_empty_source_paths_are_token_identical():
    processor = _make_processor()
    output = processor(
        videos=[np.zeros((4, 1, 1, 3), dtype=np.uint8)],
        text=["answer"],
        src_lang=["en"],
        pseudo_gloss=[""],
        training=True,
    )

    for suffix in ("input_ids", "attention_mask", "position_ids", "labels"):
        assert torch.equal(
            output[f"pseudo_gloss_teacher_{suffix}"],
            output[f"empty_source_teacher_{suffix}"],
        )


def test_evaluation_omits_training_only_teacher_paths():
    processor = _make_processor()
    output = processor(
        videos=[np.zeros((4, 1, 1, 3), dtype=np.uint8)],
        text=["answer"],
        src_lang=["en"],
        pseudo_gloss=["GLOSS"],
        training=False,
    )

    assert "pseudo_gloss_input_ids" in output
    assert not any("_teacher_" in name for name in output)


def test_cross_path_validation_rejects_non_source_differences():
    processor = _make_processor()
    prompt_parts = processor._encode_prompt_parts(
        [processor._get_prompt_parts_for_lang("en", False)]
    )
    target_ids = [processor.tokenizer._encode("answer<eos>")]
    video_path = processor._process_text_path(
        prompt_parts, [[7] * 4], target_ids, source_token_type=1
    )
    pseudo_path = processor._process_text_path(
        prompt_parts,
        [processor.tokenizer._encode("GLOSS")],
        target_ids,
        source_token_type=0,
    )
    empty_path = processor._process_text_path(
        prompt_parts, [[]], target_ids, source_token_type=0
    )

    corrupted_ids = pseudo_path.input_ids.clone()
    invariant_prompt_mask = (
        pseudo_path.attention_mask.bool()
        & ~pseudo_path.source_mask.bool()
        & pseudo_path.labels.eq(-100)
    )
    corrupted_ids[invariant_prompt_mask] += 1
    corrupted_pseudo_path = replace(pseudo_path, input_ids=corrupted_ids)

    with pytest.raises(ValueError, match="differ outside the source span"):
        processor._validate_corresponding_paths(
            {
                "video path": video_path,
                "pseudo path": corrupted_pseudo_path,
                "empty path": empty_path,
            },
            target_ids,
        )


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
