from types import SimpleNamespace

import numpy as np
from transformers import AutoTokenizer

from csi_slt.engine.sft.metrics import CTCMetric, DecodedBatch, SLTMetric
from csi_slt.constants import LANGUAGE_MAP


class _FakeBertScoreMetric:
    def __init__(self):
        self.calls = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return {"f1": [0.8] * len(kwargs["predictions"])}


class _FakeLanguageDetector:
    def __init__(self, labels):
        self.labels = labels
        self.calls = []

    def __call__(self, predictions, **kwargs):
        self.calls.append((predictions, kwargs))
        return [{"label": label, "score": 1.0} for label in self.labels]


def _get_language_id(language: str) -> int:
    """从 LANGUAGE_MAP.inverse 中反查语言 ID。"""
    for language_id, language_name in LANGUAGE_MAP.inverse.items():
        if language_name == language:
            return int(language_id)

    raise KeyError(f"Language {language!r} is not defined in LANGUAGE_MAP.")


def test_ctc_metric_computes_corpus_wer_from_lengths():
    # blank_id=0 does not appear within either valid prediction span here
    # ([1, 2] and [4]), so collapsing is a no-op and this reduces to a plain
    # edit-distance check.
    metric = CTCMetric(blank_id=0)
    output = SimpleNamespace(
        predictions=(
            np.array([[1, 2, 0], [4, 0, 0]]),
            np.array([2, 1]),
        ),
        label_ids=(
            np.array([[1, 3, 0], [4, 5, 0]]),
            np.array([2, 2]),
        ),
    )

    results = metric(output)

    assert results == {
        "ctc_wer": 0.5,
        "ctc_token_errors": 2,
        "ctc_reference_tokens": 4,
    }


def test_ctc_metric_collapses_repeats_and_drops_blank_before_scoring():
    # Raw per-frame argmax paths, as SltTrainer now emits them: sample 0 is
    # [1, 1, 0, 2] -> collapses to [1, 2] (exact match); sample 1 is
    # [0, 0, 0] -> collapses to [] (fully wrong against a length-1 reference).
    metric = CTCMetric(blank_id=0)
    output = SimpleNamespace(
        predictions=(
            np.array([[1, 1, 0, 2], [0, 0, 0, 0]]),
            np.array([4, 3]),
        ),
        label_ids=(
            np.array([[1, 2], [5, 0]]),
            np.array([2, 1]),
        ),
    )

    results = metric(output)

    assert results == {
        "ctc_wer": 1 / 3,
        "ctc_token_errors": 1,
        "ctc_reference_tokens": 3,
    }


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


def test_lacc_reports_language_and_overall_scores_and_counts_empty_as_wrong():
    detector = _FakeLanguageDetector(["en", "de", "en"])
    metric = object.__new__(SLTMetric)
    metric._language_detector = detector
    metric.language_detector_batch_size = 16
    batch = DecodedBatch(
        predictions=["hello", "hallo", "wrong language", ""],
        references=["", "", "", ""],
        languages=["en", "de", "de", "zh"],
        total_token_counts=[1, 1, 2, 0],
        generated_token_counts=[1, 1, 2, 0],
    )

    results = metric._calculate_lacc(batch)

    assert results["en_lacc"] == 1.0
    assert results["de_lacc"] == 0.5
    assert results["zh_lacc"] == 0.0
    assert results["overall_macro_lacc"] == 0.5
    assert results["overall_weighted_lacc"] == 0.5
    assert detector.calls == [
        (
            ["hello", "hallo", "wrong language"],
            {"batch_size": 16, "truncation": True},
        )
    ]


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
