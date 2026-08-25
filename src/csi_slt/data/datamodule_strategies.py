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


class TrainWithValAndTestStrategy:
    """Append the validation and test splits to the training dataset."""

    def required_splits(self, stage: Stage) -> tuple[Split, ...]:
        if stage == "fit":
            return ("train", "val", "test")
        if stage in ("test", "predict"):
            return ("val", "test")
        return ("train", "val", "test")

    def arrange(self, datasets: Mapping[Split, Dataset], stage: Stage) -> DatasetMap:
        arranged = dict(datasets)
        if "train" in arranged:
            missing_splits = {"val", "test"}.difference(arranged)
            if missing_splits:
                missing = ", ".join(sorted(missing_splits))
                raise ValueError(
                    "TrainWithValAndTestStrategy requires the validation and test "
                    f"datasets; missing: {missing}."
                )
            arranged["train"] = ConcatDataset(
                [arranged["train"], arranged["val"], arranged["test"]]
            )
        return arranged


class SplitSubsetStrategy:
    """Select deterministic percentages of the train, val, and test splits."""

    def __init__(
        self,
        train_percentage: float = 100,
        val_percentage: float = 100,
        test_percentage: float = 100,
        train_seed: int = 42,
        val_seed: int = 42,
        test_seed: int = 42,
    ) -> None:
        self.percentages = {
            "train": train_percentage,
            "val": val_percentage,
            "test": test_percentage,
        }
        self.seeds = {
            "train": train_seed,
            "val": val_seed,
            "test": test_seed,
        }

        for split, percentage in self.percentages.items():
            if not 0 < percentage <= 100:
                raise ValueError(
                    f"{split}_percentage must be greater than zero and at most 100."
                )

    def required_splits(self, stage: Stage) -> tuple[Split, ...]:
        if stage == "fit":
            return ("train", "val", "test")
        if stage in ("test", "predict"):
            return ("val", "test")
        return ("train", "val", "test")

    def arrange(self, datasets: Mapping[Split, Dataset], stage: Stage) -> DatasetMap:
        arranged = dict(datasets)
        for split, dataset in datasets.items():
            arranged[split] = self._subset(
                dataset,
                split=split,
                percentage=self.percentages[split],
                seed=self.seeds[split],
            )
        return arranged

    @staticmethod
    def _subset(
        dataset: Dataset,
        split: Split,
        percentage: float,
        seed: int,
    ) -> Dataset:
        if percentage == 100:
            return dataset

        dataset_size = len(dataset)
        if dataset_size == 0:
            raise ValueError(f"Cannot sample from an empty {split} dataset.")

        num_samples = max(1, int(dataset_size * percentage / 100))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_size, generator=generator)[
            :num_samples
        ].tolist()
        return Subset(dataset, indices)


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
