from __future__ import annotations

import os
from functools import cached_property
from typing import Literal

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset

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
        accelerator=None,
    ) -> None:
        self.data_cfg = data_cfg
        self.datamodule_cfg = datamodule_cfg
        self.tokenizer = tokenizer
        self.accelerator = accelerator

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self._source_datasets: DatasetMap = {}

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

    @property
    def _prompt_paths(self) -> dict[str, str]:
        """Resolve prompt-template paths relative to the current working directory."""
        prompt_templates = self.data_cfg.get("prompt_templates")
        if prompt_templates is None:
            return {}

        return {
            language: (
                path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
            )
            for language, path in prompt_templates.items()
        }

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
    def processor(self):
        return instantiate(
            self.data_cfg.processor,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            prompt_paths_per_language=self._prompt_paths,
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

        return instantiate(split_cfg.collator, processor=self.processor)

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
            input_text = self.tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=False
            )
            if "labels" in batch:
                labels = batch["labels"]
                labels = labels.masked_fill(
                    labels == -100,
                    self.tokenizer.pad_token_id,
                )
                label_text = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=False
                )
                for input_item, label_item in zip(input_text, label_text):
                    print(f"INPUT: \n {input_item}")
                    print(f"LABEL: \n {label_item}")
                    print("-" * 50)

            break
