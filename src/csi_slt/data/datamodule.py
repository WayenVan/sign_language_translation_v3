from __future__ import annotations

import os
from functools import cached_property
from typing import Any, Literal, Mapping, Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from csi_slt.engine.prompt_resolver import PromptResolver

from .datamodule_strategies import DatasetMap, Split, Stage


class DataModule:
    """Build datasets, processors, and collators from two config domains.

    ``data_cfg`` describes how individual datasets and their processing pipeline
    are instantiated. ``datamodule_cfg`` describes how those datasets are
    arranged for an experiment.
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        datamodule_cfg: DictConfig,
        tokenizer,
        prompt_resolvers: Mapping[str, PromptResolver],
        accelerator=None,
    ) -> None:
        self.data_cfg = data_cfg
        self.datamodule_cfg = datamodule_cfg
        self.tokenizer = tokenizer
        self.prompt_resolvers = dict(prompt_resolvers)
        self.accelerator = accelerator
        unknown_resolvers = set(self.prompt_resolvers).difference(
            {"train", "val", "test"}
        )
        if unknown_resolvers:
            raise ValueError(
                "Unknown prompt resolver splits: " f"{sorted(unknown_resolvers)}"
            )

        self._register_processor_special_tokens()

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self._source_datasets: DatasetMap = {}

    def _register_processor_special_tokens(self) -> None:
        """Make configured video markers atomic without replacing existing ones."""
        processor_cfg = self.data_cfg.get("processor")
        if processor_cfg is None:
            return

        video_tokens = [
            processor_cfg.get(name)
            for name in ("video_soft_token", "video_start_token")
            if processor_cfg.get(name) is not None
        ]
        if not video_tokens:
            return

        # Gemma's <unusedN> entries exist in its vocabulary but are not matched
        # atomically in ordinary text until registered as extra special tokens.
        # Do not replace the tokenizer's existing special-token set (notably
        # Qwen's native vision tokens), and remove duplicates while preserving
        # the configuration order.
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(dict.fromkeys(video_tokens))},
            replace_extra_special_tokens=False,
        )

    @property
    def chat_template(self) -> str | None:
        """Read the chat template configured for the data processor."""
        path = self.data_cfg.get("chat_template_jinjia")
        if path is None:
            return None

        path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Chat template file not found: {path}")

        with open(path, "r", encoding="utf-8") as template_file:
            return template_file.read()

    def setup(self, stage: Stage = None) -> None:
        """Create datasets required by the requested execution stage.

        ``fit`` creates train and validation datasets. ``test`` and ``predict``
        create the test dataset. Passing ``None`` creates all datasets.
        """
        supported_stages = {None, "fit", "test", "predict"}
        if stage not in supported_stages:
            raise ValueError(
                f"Unsupported stage {stage!r}; expected one of {supported_stages}."
            )

        required_splits = self.strategy.required_splits(stage)
        source_datasets = {
            split: self._get_or_build_dataset(split) for split in required_splits
        }
        arranged_datasets = self.strategy.arrange(source_datasets, stage)
        self._assign_datasets(arranged_datasets)

    @cached_property
    def strategy(self):
        strategy_cfg = self.datamodule_cfg.get("strategy")
        if strategy_cfg is None:
            raise ValueError("Missing datamodule.strategy configuration.")
        return instantiate(strategy_cfg)

    def _get_or_build_dataset(self, split: Split) -> Dataset:
        if split not in self._source_datasets:
            self._source_datasets[split] = self._build_dataset(split)
        return self._source_datasets[split]

    def _assign_datasets(self, datasets: DatasetMap) -> None:
        if "train" in datasets:
            self.train_dataset = datasets["train"]
        if "val" in datasets:
            self.val_dataset = datasets["val"]
        if "test" in datasets:
            self.test_dataset = datasets["test"]

    def _build_dataset(self, split: Split) -> Dataset:
        split_cfg = self.data_cfg.get(split)
        if split_cfg is None or split_cfg.get("dataset") is None:
            raise ValueError(f"Missing data.{split}.dataset configuration.")

        dataset = instantiate(split_cfg.dataset)
        prepare = getattr(dataset, "prepare", None)
        if prepare is not None:
            prepare(self.tokenizer)
        return dataset

    @cached_property
    def ctc_tokenizer(self):
        """Load the optional word-level CTC tokenizer used for pseudo-gloss supervision."""
        path = self.data_cfg.get("ctc_tokenizer_dir")
        if path is None:
            return None

        path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"CTC tokenizer directory not found: {path}")

        return AutoTokenizer.from_pretrained(path)

    @cached_property
    def processor(self):
        processor_kwargs: dict[str, Any] = {}
        if self.ctc_tokenizer is not None:
            processor_kwargs["ctc_tokenizer"] = self.ctc_tokenizer

        return instantiate(
            self.data_cfg.processor,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            **processor_kwargs,
            _convert_="all",
        )

    @cached_property
    def train_collator(self):
        return self._build_collator("train")

    @cached_property
    def val_collator(self):
        return self._build_collator("val")

    @cached_property
    def test_collator(self):
        return self._build_collator("test")

    def _build_collator(self, split: Literal["train", "val", "test"]):
        split_cfg = self.data_cfg.get(split)
        if split_cfg is None or split_cfg.get("collator") is None:
            raise ValueError(f"Missing data.{split}.collator configuration.")
        if split not in self.prompt_resolvers:
            raise ValueError(f"Missing prompt resolver for split {split!r}.")

        return instantiate(
            split_cfg.collator,
            processor=self.processor,
            prompt_resolver=self.prompt_resolvers[split],
            training=split == "train",
        )

    def print_batch(
        self,
        batch_size: int,
        num_workers: int = 1,
        random: bool = False,
    ) -> None:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not set up; call setup('fit') first.")

        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.train_collator,
            num_workers=num_workers,
            shuffle=random,
        )

        for batch in dataloader:
            text_batch = self._text_fields_as_strings(batch)
            if not text_batch:
                print("No text-related fields found in batch.")
                break

            decoded_batch_size = len(next(iter(text_batch.values())))
            for batch_index in range(decoded_batch_size):
                print(f"SAMPLE {batch_index}")
                for field_name, values in text_batch.items():
                    print(f"#{field_name}:\n{values[batch_index]}")
                print("-" * 50)

            break

    def _text_fields_as_strings(
        self, batch: Mapping[str, Any]
    ) -> dict[str, list[str]]:
        """Convert every text-bearing batch field to per-sample strings.

        Token fields are discovered by convention, so newly added source paths
        are printed automatically. Input IDs use their corresponding attention
        mask, while labels use ``-100`` as the supervision mask. Existing
        string metadata such as sample names and languages is included too.
        """
        text_fields: dict[str, list[str]] = {}
        expected_batch_size: int | None = None

        for field_name, value in batch.items():
            if self._is_token_text_field(field_name, value):
                values = self._decode_token_field(field_name, value, batch)
            elif self._is_string_field(value):
                values = self._normalize_string_field(value)
            else:
                continue

            if expected_batch_size is None:
                expected_batch_size = len(values)
            elif len(values) != expected_batch_size:
                raise ValueError(
                    f"Text field {field_name!r} has batch size {len(values)}, "
                    f"expected {expected_batch_size}."
                )
            text_fields[field_name] = values

        return text_fields

    @staticmethod
    def _is_token_text_field(field_name: str, value: Any) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and (
                field_name == "input_ids"
                or field_name.endswith("_input_ids")
                or field_name == "labels"
                or field_name.endswith("_labels")
            )
        )

    @staticmethod
    def _is_string_field(value: Any) -> bool:
        if isinstance(value, str):
            return True
        return isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        )

    @staticmethod
    def _normalize_string_field(value: str | Sequence[str]) -> list[str]:
        return [value] if isinstance(value, str) else list(value)

    def _decode_token_field(
        self,
        field_name: str,
        token_ids: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> list[str]:
        if field_name == "labels" or field_name.endswith("_labels"):
            valid_mask = token_ids.ne(-100)
        else:
            mask_name = field_name.removesuffix("input_ids") + "attention_mask"
            attention_mask = batch.get(mask_name)
            if attention_mask is None:
                valid_mask = torch.ones_like(token_ids, dtype=torch.bool)
            else:
                if not isinstance(attention_mask, torch.Tensor):
                    raise TypeError(f"{mask_name} must be a torch.Tensor.")
                if attention_mask.shape != token_ids.shape:
                    raise ValueError(
                        f"{mask_name} shape {tuple(attention_mask.shape)} does not "
                        f"match {field_name} shape {tuple(token_ids.shape)}."
                    )
                valid_mask = attention_mask.bool()

        unpadded_ids = [
            row[mask].detach().cpu().tolist()
            for row, mask in zip(token_ids, valid_mask, strict=True)
        ]
        return list(
            self.tokenizer.batch_decode(
                unpadded_ids,
                skip_special_tokens=False,
            )
        )
