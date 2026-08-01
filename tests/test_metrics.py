from types import SimpleNamespace

import numpy as np
from transformers import AutoTokenizer

from csi_slt.engine.metrics import SLTMetric
from csi_slt.constants import LANGUAGE_MAP


class _FakeBertScoreMetric:
    def __init__(self):
        self.calls = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return {"f1": [0.8] * len(kwargs["predictions"])}


def _get_language_id(language: str) -> int:
    """从 LANGUAGE_MAP.inverse 中反查语言 ID。"""
    for language_id, language_name in LANGUAGE_MAP.inverse.items():
        if language_name == language:
            return int(language_id)

    raise KeyError(f"Language {language!r} is not defined in LANGUAGE_MAP.")


def test_bert_score_treats_empty_text_as_zero():
    metric = object.__new__(SLTMetric)
    metric.bert_score_model_type = "unused-test-model"
    metric._bert_score_metric = _FakeBertScoreMetric()

    score = metric._calculate_bert_score_f1(
        predictions=["valid prediction", "", "   ", "prediction"],
        references=["valid reference", "reference", "reference", ""],
    )

    assert score == 0.2
    assert len(metric._bert_score_metric.calls) == 1
    assert metric._bert_score_metric.calls[0]["predictions"] == [
        "valid prediction"
    ]
    assert metric._bert_score_metric.calls[0]["references"] == ["valid reference"]


def test_bert_score_skips_backend_when_all_text_is_empty():
    metric = object.__new__(SLTMetric)
    metric.bert_score_model_type = "unused-test-model"
    metric._bert_score_metric = _FakeBertScoreMetric()

    score = metric._calculate_bert_score_f1(
        predictions=["", "   "],
        references=["reference", "reference"],
    )

    assert score == 0.0
    assert metric._bert_score_metric.calls == []


def test_slt_metric_quick():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    processor = SimpleNamespace(tokenizer=tokenizer)
    metric = SLTMetric(processor)

    prompts = [
        "Translate:",
        "翻译：",
    ]

    predictions = [
        "hello world",
        "你好世界",
    ]

    references = [
        "hello world",
        "你好世界",
    ]

    languages = [
        "en",
        "zh",
    ]

    prediction_sequences = []
    prompt_lengths = []

    for prompt, prediction in zip(prompts, predictions):
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        prediction_ids = tokenizer.encode(
            prediction,
            add_special_tokens=False,
        )

        # 模拟模型输出：
        # prompt tokens + generated tokens
        sequence = prompt_ids + prediction_ids

        prediction_sequences.append(sequence)
        prompt_lengths.append(len(prompt_ids))

    max_prediction_length = max(len(sequence) for sequence in prediction_sequences)

    prediction_ids = np.full(
        shape=(len(prediction_sequences), max_prediction_length),
        fill_value=tokenizer.pad_token_id,
        dtype=np.int64,
    )

    prediction_lengths = np.zeros(
        len(prediction_sequences),
        dtype=np.int64,
    )

    for index, sequence in enumerate(prediction_sequences):
        prediction_ids[index, : len(sequence)] = sequence
        prediction_lengths[index] = len(sequence)

    encoded_labels = tokenizer(
        references,
        add_special_tokens=False,
        padding=True,
        return_tensors="np",
    )["input_ids"]

    # 模拟 Trainer 中 label padding 使用 -100。
    label_ids = np.where(
        encoded_labels == tokenizer.pad_token_id,
        -100,
        encoded_labels,
    )

    language_ids = np.array(
        [_get_language_id(language) for language in languages],
        dtype=np.int64,
    )

    output = SimpleNamespace(
        predictions=(
            prediction_ids,
            prediction_lengths,
            np.asarray(prompt_lengths, dtype=np.int64),
        ),
        label_ids=(
            label_ids,
            language_ids,
        ),
    )

    results = metric(output)

    print("\nSLT metric results:")
    for name, value in sorted(results.items()):
        print(f"{name}: {value}")

    assert results["num_samples"] == 2
    # assert results["overall_bleu1"] > 0.99
    # assert results["overall_rougeL"] > 0.99

    assert results["en_bleu1"] > 0.99
    assert results["zh_bleu1"] > 0.99

    assert results["all_n_tokens_generated"] > 0


if __name__ == "__main__":
    test_slt_metric_quick()
