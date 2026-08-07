from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset


Stage = Literal["fit", "test", "predict", None]
Split = Literal["train", "val", "test"]
DatasetMap = dict[Split, Dataset]


class StandardSplitStrategy:
    """Keep the configured train, validation, and test splits separate."""

    def __init__(self, train_with_val: bool = False) -> None:
        self.train_with_val = train_with_val

    def required_splits(self, stage: Stage) -> tuple[Split, ...]:
        if stage == "fit":
            return ("train", "val", "test")
        if stage in ("test", "predict"):
            return (
                "val",
                "test",
            )
        return ("train", "val", "test")

    def arrange(self, datasets: Mapping[Split, Dataset], stage: Stage) -> DatasetMap:
        arranged = dict(datasets)
        if "train" in arranged and self.train_with_val:
            if "val" not in arranged:
                raise ValueError("train_with_val requires the validation dataset.")
            arranged["train"] = ConcatDataset([arranged["train"], arranged["val"]])
        return arranged


class TrainSubsetStrategy:
    """Train on a deterministic percentage of train and keep val/test intact."""

    def __init__(self, percentage: float, seed: int = 42) -> None:
        if not 0 < percentage <= 100:
            raise ValueError("percentage must be greater than zero and at most 100.")

        self.percentage = percentage
        self.seed = seed

    def required_splits(self, stage: Stage) -> tuple[Split, ...]:
        if stage == "fit":
            return ("train", "val", "test")
        if stage in ("test", "predict"):
            return ("val", "test")
        return ("train", "val", "test")

    def arrange(self, datasets: Mapping[Split, Dataset], stage: Stage) -> DatasetMap:
        arranged = dict(datasets)
        if "train" not in arranged:
            return arranged

        train_dataset = arranged["train"]
        dataset_size = len(train_dataset)
        if dataset_size == 0:
            raise ValueError("Cannot sample from an empty training dataset.")

        num_samples = max(1, int(dataset_size * self.percentage / 100))
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=generator)[
            :num_samples
        ].tolist()
        arranged["train"] = Subset(train_dataset, indices)
        return arranged


class SharedSubsetStrategy:
    """Use one deterministic subset as train, validation, and test data."""

    def __init__(
        self,
        num_samples: int,
        source_split: Split = "train",
        seed: int = 42,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero.")

        self.num_samples = num_samples
        self.source_split = source_split
        self.seed = seed
        self._shared_dataset: Subset | None = None
        self._source_dataset: Dataset | None = None

    def required_splits(self, stage: Stage) -> tuple[Split, ...]:
        return (self.source_split,)

    def arrange(self, datasets: Mapping[Split, Dataset], stage: Stage) -> DatasetMap:
        source_dataset = datasets[self.source_split]
        shared_dataset = self._get_or_create_subset(source_dataset)
        return {
            "train": shared_dataset,
            "val": shared_dataset,
            "test": shared_dataset,
        }

    def _get_or_create_subset(self, source_dataset: Dataset) -> Subset:
        if self._shared_dataset is not None:
            if source_dataset is not self._source_dataset:
                raise RuntimeError(
                    "SharedSubsetStrategy cannot be reused with another source dataset."
                )
            return self._shared_dataset

        dataset_size = len(source_dataset)
        if self.num_samples > dataset_size:
            raise ValueError(
                f"num_samples ({self.num_samples}) exceeds the size of the "
                f"{self.source_split!r} dataset ({dataset_size})."
            )

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=generator)[
            : self.num_samples
        ].tolist()

        self._source_dataset = source_dataset
        self._shared_dataset = Subset(source_dataset, indices)
        return self._shared_dataset
