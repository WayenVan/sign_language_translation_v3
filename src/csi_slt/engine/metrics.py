import evaluate
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers.trainer_utils import EvalLoopOutput
from ..constants import LANGUAGE_MAP
from collections import defaultdict
import math


class SLTMetric:
    def __init__(self, processor):
        self.processor = processor

    def _parse_prediction(self, pred: EvalLoopOutput):
        tokenizer = self.processor.tokenizer

        preds_ids, pred_length, prompt_length = pred.predictions
        labels_ids, language_ids = pred.label_ids

        n_tokens = []
        full_prediction_texts = []
        predction_texts = []
        label_texts = []
        languages = []
        B = labels_ids.shape[0]
        for b in range(B):
            full_prediction = preds_ids[b][: pred_length[b]]
            prediction = full_prediction[prompt_length[b] :]
            label = labels_ids[b]
            # replace -100 in the labels as we can't decode them
            label = [l if l != -100 else tokenizer.pad_token_id for l in label]
            # decode
            full_pred_text = tokenizer.decode(
                full_prediction,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred_text = tokenizer.decode(
                prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            label_text = tokenizer.decode(
                label, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            full_prediction_texts.append(full_pred_text)
            predction_texts.append(pred_text)
            label_texts.append(label_text)
            languages.append(LANGUAGE_MAP.inverse[language_ids[b].item()])  #
            n_tokens.append((full_prediction != tokenizer.pad_token_id).sum().item())

        return full_prediction_texts, predction_texts, label_texts, languages, n_tokens

    def calculate_bleus(self, predictions, references, prefix=""):
        tokenizer = self.processor.tokenizer
        bleu = evaluate.load("bleu")
        results_bleu_1 = bleu.compute(
            predictions=predictions,
            references=[[l] for l in references],
            max_order=1,
        )
        results_bleu_4 = bleu.compute(
            predictions=predictions,
            references=[[l] for l in references],
            max_order=4,
        )

        # calculate sentence-level BLEU for analysis purpose
        sentence_bleu_1: list = []
        sentence_bleu_4: list = []
        for label, pred in zip(references, predictions):
            smoothie = SmoothingFunction().method3
            sentence_bleu_1.append(
                sentence_bleu(
                    [tokenizer.tokenize(label)],
                    tokenizer.tokenize(pred),
                    weights=(1, 0, 0, 0),
                    smoothing_function=smoothie,
                )
            )
            sentence_bleu_4.append(
                sentence_bleu(
                    [tokenizer.tokenize(label)],
                    tokenizer.tokenize(pred),
                    weights=(0, 0, 0, 1),
                    smoothing_function=smoothie,
                )
            )

        return {
            f"{prefix}bleu1": results_bleu_1["bleu"],
            f"{prefix}bleu4": results_bleu_4["bleu"],
            f"{prefix}sentence_bleu_1": np.mean(sentence_bleu_1),
            f"{prefix}sentence_bleu_4": np.mean(sentence_bleu_4),
        }

    def calcuate_rouge(self, predictions, references, prefix=""):
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=predictions, references=references)
        return {f"{prefix}{k}": v for k, v in results.items()}

    def calculate_metrics(self, predictions, references, prefix=""):
        metrics = {}
        metrics.update(self.calculate_bleus(predictions, references, prefix))
        metrics.update(self.calcuate_rouge(predictions, references, prefix))
        return metrics

    def get_language_buckets(self, pred: EvalLoopOutput):
        """获取按语言分组的桶，使用defaultdict简化代码"""
        _, prediction_texts, label_texts, languages, _ = self._parse_prediction(pred)

        buckets = defaultdict(lambda: {"predictions": [], "references": []})
        for pred_text, label_text, lang in zip(
            prediction_texts, label_texts, languages
        ):
            buckets[lang]["predictions"].append(pred_text)
            buckets[lang]["references"].append(label_text)

        return dict(buckets)  # 转换为普通dict以便于使用

    def __call__(self, pred: EvalLoopOutput) -> dict:
        full_prediction_texts, prediction_texts, label_texts, langauges, n_tokens = (
            self._parse_prediction(pred)
        )

        # 计算总体指标
        all_metrics = self.calculate_metrics(
            prediction_texts, label_texts, prefix="overall_"
        )

        # 使用语言桶计算每个语言的指标
        language_buckets = self.get_language_buckets(pred)

        for lang, bucket in language_buckets.items():
            if bucket["predictions"]:
                lang_metrics = self.calculate_bleus(
                    bucket["predictions"], bucket["references"], prefix=f"{lang}_"
                )
                all_metrics.update(lang_metrics)

        all_metrics["avg_n_tokens"] = np.mean(n_tokens)
        return all_metrics
