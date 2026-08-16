"""Lightweight per-sample rewards for SLT reinforcement learning."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

from sacrebleu.metrics import BLEU


class SentenceBLEUReward:
    """Calculate one smoothed sentence BLEU reward per translation.

    Unlike the corpus BLEU in ``engine.metrics``, this class returns one score
    for every prediction so GRPO can compare completions from the same prompt.
    Scores use the same 0--1 range as ``SLTMetric``.

    Chinese uses SacreBLEU's ``zh`` tokenizer. German uses ``13a`` explicitly
    and remains case-sensitive by default, preserving German noun casing and
    NFC-normalized umlauts. Language aliases are accepted for both languages.
    """

    _CHINESE_LANGUAGE_CODES = frozenset(
        {
            "zh",
            "zh-cn",
            "zh-tw",
            "zh-hans",
            "zh-hant",
            "zho",
            "cmn",
            "chinese",
            "中文",
        }
    )
    _GERMAN_LANGUAGE_CODES = frozenset(
        {
            "de",
            "de-de",
            "de-at",
            "de-ch",
            "deu",
            "ger",
            "german",
            "deutsch",
        }
    )

    def __init__(
        self,
        *,
        lowercase: bool = False,
        default_tokenizer: str = "13a",
        tokenizer_by_language: Mapping[str, str] | None = None,
        max_ngram_order: int = 4,
    ) -> None:
        if max_ngram_order < 1:
            raise ValueError("max_ngram_order must be positive")

        self.lowercase = lowercase
        self.default_tokenizer = default_tokenizer
        self.max_ngram_order = max_ngram_order
        self.tokenizer_by_language = {
            **{language: "zh" for language in self._CHINESE_LANGUAGE_CODES},
            **{language: "13a" for language in self._GERMAN_LANGUAGE_CODES},
        }
        if tokenizer_by_language:
            self.tokenizer_by_language.update(
                {
                    self._normalize_language_code(language): tokenizer
                    for language, tokenizer in tokenizer_by_language.items()
                }
            )

        # SacreBLEU objects only differ by tokenizer in this reward.
        self._metric_cache: dict[str, BLEU] = {}

    # -------------------------------------------------------------------------
    # Text and language normalization.
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_language_code(language: str) -> str:
        if not isinstance(language, str):
            raise TypeError("language codes must be strings")
        return language.strip().lower().replace("_", "-")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Match the normalization used by ``SLTMetric``."""
        if not isinstance(text, str):
            raise TypeError("predictions and references must be strings")
        text = unicodedata.normalize("NFC", text)
        text = "".join(
            character
            for character in text
            if not unicodedata.category(character).startswith("P")
        )
        return " ".join(text.strip().split())

    # -------------------------------------------------------------------------
    # Language-specific SacreBLEU construction.
    # -------------------------------------------------------------------------
    def _get_tokenizer_name(self, language: str) -> str:
        language = self._normalize_language_code(language)
        if language in self.tokenizer_by_language:
            return self.tokenizer_by_language[language]
        if language.startswith("zh-"):
            return "zh"
        if language.startswith("de-"):
            return "13a"
        return self.default_tokenizer

    def _get_metric(self, language: str) -> BLEU:
        tokenizer_name = self._get_tokenizer_name(language)
        if tokenizer_name not in self._metric_cache:
            self._metric_cache[tokenizer_name] = BLEU(
                tokenize=tokenizer_name,
                lowercase=self.lowercase,
                smooth_method="exp",
                effective_order=True,
                max_ngram_order=self.max_ngram_order,
            )
        return self._metric_cache[tokenizer_name]

    # -------------------------------------------------------------------------
    # Per-sentence BLEU scoring.
    # -------------------------------------------------------------------------
    def score(self, prediction: str, reference: str, language: str) -> float:
        prediction = self._normalize_text(prediction)
        reference = self._normalize_text(reference)
        if not prediction or not reference:
            return 0.0

        result = self._get_metric(language).sentence_score(
            hypothesis=prediction,
            references=[reference],
        )
        return min(1.0, max(0.0, float(result.score / 100.0)))

    def __call__(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        languages: Sequence[str],
    ) -> list[float]:
        if not (
            len(predictions) == len(references) == len(languages)
        ):
            raise ValueError(
                "predictions, references, and languages must have the same length"
            )

        return [
            self.score(prediction, reference, language)
            for prediction, reference, language in zip(
                predictions,
                references,
                languages,
                strict=True,
            )
        ]
