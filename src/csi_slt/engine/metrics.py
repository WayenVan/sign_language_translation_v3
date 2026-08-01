from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import evaluate
import numpy as np
from sacrebleu.metrics import BLEU

from ..constants import LANGUAGE_MAP


class PredictionOutput(Protocol):
    """SLTMetric 所需的最小输入接口。"""

    predictions: Any
    label_ids: Any


class PriodicMetric(Protocol):
    """A slow metric which is evaluated only at its configured cadence."""

    every_n_evaluations: int

    def compute(self, context: MetricContext) -> Mapping[str, float | int]:
        ...


@dataclass(slots=True)
class DecodedBatch:
    """解码后的完整评估数据。"""

    predictions: list[str]
    references: list[str]
    languages: list[str]
    total_token_counts: list[int]
    generated_token_counts: list[int]

    def __len__(self) -> int:
        return len(self.predictions)


@dataclass
class MetricContext:
    """评估指标计算的上下文。"""

    batch: DecodedBatch
    language_groups: dict[str, list[int]]


class SLTMetric:
    """
    多语言手语翻译指标。

    输入格式
    --------
    output.predictions:

        (
            prediction_ids,
            sequence_lengths,
            prompt_lengths,
        )

    output.label_ids:

        (
            label_ids,
            language_ids,
        )

    BLEU 输出
    --------
    每种语言：

        en_bleu1
        en_bleu4
        zh_bleu1
        zh_bleu4

    总体 BLEU：

        overall_macro_bleu4
        overall_weighted_bleu4

    BERTScore-F1：

        en_bert_score_f1
        zh_bert_score_f1
        overall_macro_bert_score_f1

    注意：两个 overall BLEU 都由各语言的 BLEU-4 聚合得到，
    不会把不同目标语言混合后直接计算 corpus BLEU。

    所有 BLEU、ROUGE 和 BERTScore 指标均返回 0～1。
    """

    _CJK_CHARACTER_PATTERN = (
        r"[\u3400-\u4DBF"
        r"\u4E00-\u9FFF"
        r"\uF900-\uFAFF]"
    )

    # 中文字符单独成 token；
    # 其他 Unicode 字母和数字按连续字符串分词。
    _MULTILINGUAL_TOKEN_PATTERN = re.compile(
        rf"{_CJK_CHARACTER_PATTERN}|[^\W_]+",
        flags=re.UNICODE,
    )

    _DEFAULT_CHINESE_LANGUAGE_CODES = (
        "zh",
        "zh-cn",
        "zh-tw",
        "zh-hans",
        "zh-hant",
        "zho",
        "cmn",
        "chinese",
    )

    def __init__(
        self,
        processor: Any,
        *,
        ignore_index: int = -100,
        lowercase_bleu: bool = False,
        default_bleu_tokenizer: str = "13a",
        bleu_tokenizer_by_language: Mapping[str, str] | None = None,
        bert_score_model_type: str = "bert-base-multilingual-cased",
        priodic_metrics: Sequence[PriodicMetric] | None = None,
    ) -> None:
        self.tokenizer = processor.tokenizer
        self.ignore_index = ignore_index
        self.lowercase_bleu = lowercase_bleu
        self.default_bleu_tokenizer = default_bleu_tokenizer
        self.bert_score_model_type = bert_score_model_type
        self.priodic_metrics = tuple(priodic_metrics or ())
        self._evaluation_count = 0

        for metric in self.priodic_metrics:
            interval = getattr(metric, "every_n_evaluations", None)
            if (
                not isinstance(interval, int)
                or isinstance(interval, bool)
                or interval == 0
                or interval < -1
            ):
                raise ValueError(
                    "Each priodic metric must define 'every_n_evaluations' "
                    "as a positive integer, or -1 to disable it."
                )
            if not callable(getattr(metric, "compute", None)):
                raise TypeError(
                    "Each priodic metric must provide a callable compute(context)."
                )

        if self.tokenizer.pad_token_id is None:
            raise ValueError("SLTMetric requires tokenizer.pad_token_id to be defined.")

        # 中文默认使用 SacreBLEU 的 zh tokenizer。
        self.bleu_tokenizer_by_language = {
            self._normalize_language_code(language): "zh"
            for language in self._DEFAULT_CHINESE_LANGUAGE_CODES
        }

        # 允许覆盖或添加特定语言 tokenizer。
        #
        # 例如：
        #
        # bleu_tokenizer_by_language={
        #     "ja": "ja-mecab",
        #     "ko": "ko-mecab",
        # }
        if bleu_tokenizer_by_language:
            self.bleu_tokenizer_by_language.update(
                {
                    self._normalize_language_code(language): tokenizer_name
                    for language, tokenizer_name in bleu_tokenizer_by_language.items()
                }
            )

        # key: (tokenizer_name, max_ngram_order)
        self._bleu_metric_cache: dict[tuple[str, int], BLEU] = {}

        # ROUGE 只加载一次。
        self._rouge_metric = evaluate.load("rouge")

        # 所有语言使用同一个多语言模型，使不同语言的分数可按语言聚合，
        # 同时避免 BERTScore 根据语言代码选择模型时不支持部分语言代码。
        self._bert_score_metric = evaluate.load("bertscore")

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_language_code(language: str) -> str:
        return language.strip().lower().replace("_", "-")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        进行不改变语义的基础规范化。

        - Unicode NFC 规范化；
        - 删除首尾空格；
        - 合并连续空白字符。
        """

        text = unicodedata.normalize("NFC", text)

        # WARN: 删除所有 Unicode 标点字符会使 BLEU 和 ROUGE 忽略中文、英文、
        # 德文等语言中的标点差异；这也会移除英文缩写或所有格中的撇号。
        text = "".join(
            character
            for character in text
            if not unicodedata.category(character).startswith("P")
        )

        return " ".join(text.strip().split())

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """兼容 NumPy array 和 PyTorch tensor。"""

        if isinstance(value, np.ndarray):
            return value

        if hasattr(value, "detach"):
            value = value.detach()

        if hasattr(value, "cpu"):
            value = value.cpu()

        if hasattr(value, "numpy"):
            return value.numpy()

        return np.asarray(value)

    @staticmethod
    def _validate_batch_sizes(
        expected_size: int,
        **arrays: np.ndarray,
    ) -> None:
        for name, array in arrays.items():
            if len(array) != expected_size:
                raise ValueError(
                    f"Batch-size mismatch: {name} contains "
                    f"{len(array)} samples, expected {expected_size}."
                )

    # ------------------------------------------------------------------
    # Input parsing
    # ------------------------------------------------------------------

    def _unpack_output(
        self,
        output: PredictionOutput,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        try:
            (
                prediction_ids,
                sequence_lengths,
                prompt_lengths,
            ) = output.predictions
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "output.predictions must be "
                "(prediction_ids, sequence_lengths, prompt_lengths)."
            ) from exc

        try:
            label_ids, language_ids = output.label_ids
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "output.label_ids must be (label_ids, language_ids)."
            ) from exc

        prediction_ids = self._to_numpy(prediction_ids)
        sequence_lengths = self._to_numpy(sequence_lengths).reshape(-1)
        prompt_lengths = self._to_numpy(prompt_lengths).reshape(-1)
        label_ids = self._to_numpy(label_ids)
        language_ids = self._to_numpy(language_ids).reshape(-1)

        if prediction_ids.ndim != 2:
            raise ValueError(
                "prediction_ids must have shape (batch_size, max_sequence_length)."
            )

        if label_ids.ndim != 2:
            raise ValueError(
                "label_ids must have shape (batch_size, max_label_length)."
            )

        batch_size = prediction_ids.shape[0]

        self._validate_batch_sizes(
            batch_size,
            sequence_lengths=sequence_lengths,
            prompt_lengths=prompt_lengths,
            label_ids=label_ids,
            language_ids=language_ids,
        )

        return (
            prediction_ids,
            sequence_lengths,
            prompt_lengths,
            label_ids,
            language_ids,
        )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_batch(
        self,
        output: PredictionOutput,
    ) -> DecodedBatch:
        (
            prediction_ids,
            sequence_lengths,
            prompt_lengths,
            label_ids,
            language_ids,
        ) = self._unpack_output(output)

        pad_token_id = self.tokenizer.pad_token_id
        batch_size, max_sequence_length = prediction_ids.shape

        generated_sequences: list[list[int]] = []
        total_token_counts: list[int] = []
        generated_token_counts: list[int] = []

        for index in range(batch_size):
            sequence_length = int(sequence_lengths[index])
            prompt_length = int(prompt_lengths[index])

            if not 0 <= prompt_length <= sequence_length <= max_sequence_length:
                raise ValueError(
                    f"Invalid lengths at sample {index}: "
                    f"prompt_length={prompt_length}, "
                    f"sequence_length={sequence_length}, "
                    f"max_sequence_length={max_sequence_length}."
                )

            full_sequence = prediction_ids[
                index,
                :sequence_length,
            ]

            generated_sequence = full_sequence[prompt_length:]

            generated_sequences.append(generated_sequence.tolist())

            total_token_counts.append(
                int(np.count_nonzero(full_sequence != pad_token_id))
            )

            generated_token_counts.append(
                int(np.count_nonzero(generated_sequence != pad_token_id))
            )

        clean_label_ids = label_ids.copy()
        clean_label_ids[clean_label_ids == self.ignore_index] = pad_token_id

        predictions = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True,
        )

        references = self.tokenizer.batch_decode(
            clean_label_ids.tolist(),
            skip_special_tokens=True,
        )

        languages = [
            str(LANGUAGE_MAP.inverse[int(language_id)]) for language_id in language_ids
        ]

        return DecodedBatch(
            predictions=[
                self._normalize_text(prediction) for prediction in predictions
            ],
            references=[self._normalize_text(reference) for reference in references],
            languages=languages,
            total_token_counts=total_token_counts,
            generated_token_counts=generated_token_counts,
        )

    # ------------------------------------------------------------------
    # BLEU
    # ------------------------------------------------------------------

    def _get_bleu_tokenizer_name(
        self,
        language: str,
    ) -> str:
        normalized_language = self._normalize_language_code(language)

        explicit_tokenizer = self.bleu_tokenizer_by_language.get(normalized_language)

        if explicit_tokenizer is not None:
            return explicit_tokenizer

        # 兼容其他 zh-* 形式。
        if normalized_language.startswith("zh-"):
            return "zh"

        return self.default_bleu_tokenizer

    def _get_bleu_metric(
        self,
        language: str,
        *,
        max_ngram_order: int,
    ) -> BLEU:
        tokenizer_name = self._get_bleu_tokenizer_name(language)

        cache_key = (
            tokenizer_name,
            max_ngram_order,
        )

        if cache_key not in self._bleu_metric_cache:
            self._bleu_metric_cache[cache_key] = BLEU(
                tokenize=tokenizer_name,
                lowercase=self.lowercase_bleu,
                smooth_method="exp",
                effective_order=False,
                max_ngram_order=max_ngram_order,
            )

        return self._bleu_metric_cache[cache_key]

    def _calculate_bleu_score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        language: str,
        max_ngram_order: int,
    ) -> float:
        """
        计算单一语言的 corpus BLEU-N。

        BLEU-1:
            max_ngram_order=1

        BLEU-4:
            max_ngram_order=4
        """

        if not predictions:
            return 0.0

        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length.")

        metric = self._get_bleu_metric(
            language,
            max_ngram_order=max_ngram_order,
        )

        result = metric.corpus_score(
            hypotheses=list(predictions),
            references=[list(references)],
        )

        # SacreBLEU 原始范围是 0～100。
        return float(result.score / 100.0)

    def _calculate_bleu(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        language: str,
    ) -> dict[str, float]:
        """同时计算 BLEU-1 和 BLEU-4。"""

        return {
            "bleu1": self._calculate_bleu_score(
                predictions,
                references,
                language=language,
                max_ngram_order=1,
            ),
            "bleu4": self._calculate_bleu_score(
                predictions,
                references,
                language=language,
                max_ngram_order=4,
            ),
            "bleu2": self._calculate_bleu_score(
                predictions,
                references,
                language=language,
                max_ngram_order=2,
            ),
            "bleu3": self._calculate_bleu_score(
                predictions,
                references,
                language=language,
                max_ngram_order=3,
            ),
        }

    # ------------------------------------------------------------------
    # ROUGE
    # ------------------------------------------------------------------

    @classmethod
    def _multilingual_rouge_tokenize(
        cls,
        text: str,
    ) -> list[str]:
        """
        多语言 ROUGE tokenizer。

        中文：
            你好世界
            -> ["你", "好", "世", "界"]

        英文：
            Hello, world!
            -> ["hello", "world"]

        中英混合：
            GPT-4 模型
            -> ["gpt", "4", "模", "型"]
        """

        return cls._MULTILINGUAL_TOKEN_PATTERN.findall(text.lower())

    def _calculate_rouge(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> dict[str, float]:
        if not predictions:
            return {}

        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length.")

        result = self._rouge_metric.compute(
            predictions=list(predictions),
            references=list(references),
            use_stemmer=False,
            tokenizer=self._multilingual_rouge_tokenize,
        )

        return {metric_name: float(value) for metric_name, value in result.items()}

    # ------------------------------------------------------------------
    # BERTScore
    # ------------------------------------------------------------------

    def _calculate_bert_score_f1(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        """计算一个语言分组内所有样本的平均 BERTScore-F1。

        Empty predictions or references receive a score of zero. In addition
        to being the conservative metric value, filtering them here avoids an
        incompatibility between bert-score 0.3.x and Transformers 5.x: the
        former calls a tokenizer method removed by the latter when encoding an
        empty string.
        """

        if not predictions:
            return 0.0

        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length.")

        scores = np.zeros(len(predictions), dtype=np.float64)
        valid_indices = [
            index
            for index, (prediction, reference) in enumerate(
                zip(predictions, references)
            )
            if prediction.strip() and reference.strip()
        ]

        if not valid_indices:
            return 0.0

        result = self._bert_score_metric.compute(
            predictions=[predictions[index] for index in valid_indices],
            references=[references[index] for index in valid_indices],
            model_type=self.bert_score_model_type,
        )

        f1_scores = np.asarray(result["f1"], dtype=np.float64)

        if f1_scores.size != len(valid_indices):
            raise ValueError(
                "BERTScore returned an unexpected number of F1 scores: "
                f"{f1_scores.size}, expected {len(valid_indices)}."
            )

        scores[valid_indices] = f1_scores
        return float(scores.mean())

    # ------------------------------------------------------------------
    # Metric composition
    # ------------------------------------------------------------------

    @staticmethod
    def _add_prefix(
        metrics: Mapping[str, float],
        prefix: str,
    ) -> dict[str, float]:
        return {
            f"{prefix}{metric_name}": value for metric_name, value in metrics.items()
        }

    def _calculate_language_metrics(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        language: str,
        prefix: str,
    ) -> dict[str, float]:
        metrics = self._calculate_bleu(
            predictions,
            references,
            language=language,
        )

        metrics.update(
            self._calculate_rouge(
                predictions,
                references,
            )
        )

        metrics["bert_score_f1"] = self._calculate_bert_score_f1(
            predictions,
            references,
        )

        return self._add_prefix(
            metrics,
            prefix,
        )

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_indices_by_language(
        languages: Sequence[str],
    ) -> dict[str, list[int]]:
        groups: defaultdict[str, list[int]] = defaultdict(list)

        for index, language in enumerate(languages):
            groups[language].append(index)

        return dict(groups)

    @staticmethod
    def _select_items(
        items: Sequence[str],
        indices: Sequence[int],
    ) -> list[str]:
        return [items[index] for index in indices]

    # ------------------------------------------------------------------
    # BLEU-4 aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_bleu4(
        language_bleu4_scores: Mapping[str, float],
        language_sample_counts: Mapping[str, int],
    ) -> dict[str, float]:
        """
        聚合各语言的 corpus BLEU-4。

        overall_macro_bleu4:
            每种语言权重相同。

        overall_weighted_bleu4:
            根据每种语言的样本数进行加权。
        """

        if not language_bleu4_scores:
            return {
                "overall_macro_bleu4": 0.0,
                "overall_weighted_bleu4": 0.0,
            }

        languages = list(language_bleu4_scores)

        bleu4_scores = np.asarray(
            [language_bleu4_scores[language] for language in languages],
            dtype=np.float64,
        )

        sample_counts = np.asarray(
            [language_sample_counts[language] for language in languages],
            dtype=np.float64,
        )

        if np.any(sample_counts <= 0):
            raise ValueError("Every language bucket must contain at least one sample.")

        return {
            "overall_macro_bleu4": float(bleu4_scores.mean()),
            "overall_weighted_bleu4": float(
                np.average(
                    bleu4_scores,
                    weights=sample_counts,
                )
            ),
        }

    @staticmethod
    def _aggregate_bert_score_f1(
        language_bert_score_f1_scores: Mapping[str, float],
    ) -> dict[str, float]:
        """对各语言的 BERTScore-F1 做等权宏平均。"""

        if not language_bert_score_f1_scores:
            return {"overall_macro_bert_score_f1": 0.0}

        return {
            "overall_macro_bert_score_f1": float(
                np.mean(list(language_bert_score_f1_scores.values()))
            )
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _calculate_priodic_metrics(
        self,
        context: MetricContext,
    ) -> dict[str, float | int]:
        """Run slow metrics whose cadence matches the current evaluation."""

        metrics: dict[str, float | int] = {}

        for priodic_metric in self.priodic_metrics:
            interval = priodic_metric.every_n_evaluations
            if interval == -1 or self._evaluation_count % interval != 0:
                continue

            result = priodic_metric.compute(context)
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"{type(priodic_metric).__name__}.compute() must return a mapping."
                )
            metrics.update(result)

        return metrics

    def __call__(
        self,
        output: PredictionOutput,
    ) -> dict[str, float | int]:
        self._evaluation_count += 1
        batch = self._decode_batch(output)
        language_groups = self._group_indices_by_language(batch.languages)
        context = MetricContext(
            batch=batch,
            language_groups=language_groups,
        )

        if len(batch) == 0:
            metrics: dict[str, float | int] = {
                "num_samples": 0,
                "num_languages": 0,
                "overall_macro_bleu4": 0.0,
                "overall_weighted_bleu4": 0.0,
                "overall_macro_bert_score_f1": 0.0,
                "avg_n_tokens": 0.0,
                "avg_n_tokens_generated": 0.0,
                "all_n_tokens_generated": 0,
            }
            metrics.update(self._calculate_priodic_metrics(context))
            return metrics

        metrics: dict[str, float | int] = {}

        # ROUGE 使用统一 tokenizer，因此可以直接计算
        # 整个多语言测试集的 overall ROUGE。
        overall_rouge = self._calculate_rouge(
            batch.predictions,
            batch.references,
        )

        metrics.update(
            self._add_prefix(
                overall_rouge,
                prefix="overall_",
            )
        )

        language_bleu4_scores: dict[str, float] = {}
        language_bert_score_f1_scores: dict[str, float] = {}
        language_sample_counts: dict[str, int] = {}

        for language in sorted(language_groups):
            indices = language_groups[language]

            language_predictions = self._select_items(
                batch.predictions,
                indices,
            )

            language_references = self._select_items(
                batch.references,
                indices,
            )

            # 指标键中的连字符替换为下划线。
            metric_language = self._normalize_language_code(language).replace("-", "_")

            language_metrics = self._calculate_language_metrics(
                language_predictions,
                language_references,
                language=language,
                prefix=f"{metric_language}_",
            )

            metrics.update(language_metrics)

            # Overall 指标只聚合 BLEU-4。
            language_bleu4_scores[language] = language_metrics[
                f"{metric_language}_bleu4"
            ]
            language_bert_score_f1_scores[language] = language_metrics[
                f"{metric_language}_bert_score_f1"
            ]

            language_sample_counts[language] = len(indices)

            metrics[f"{metric_language}_num_samples"] = len(indices)

        metrics.update(
            self._aggregate_bleu4(
                language_bleu4_scores,
                language_sample_counts,
            )
        )

        metrics.update(
            self._aggregate_bert_score_f1(language_bert_score_f1_scores)
        )

        metrics.update(
            {
                "num_samples": len(batch),
                "num_languages": len(language_groups),
                "avg_n_tokens": float(np.mean(batch.total_token_counts)),
                "avg_n_tokens_generated": float(np.mean(batch.generated_token_counts)),
                "all_n_tokens_generated": int(np.sum(batch.generated_token_counts)),
            }
        )
        metrics.update(self._calculate_priodic_metrics(context))

        return metrics
