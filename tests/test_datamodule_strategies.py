import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from csi_slt.data.datamodule_strategies import (
    SharedSubsetStrategy,
    SplitSubsetStrategy,
    StandardSplitStrategy,
)


def test_shared_subset_is_used_for_every_split():
    source = TensorDataset(torch.arange(20))
    strategy = SharedSubsetStrategy(num_samples=5, seed=7)

    datasets = strategy.arrange({"train": source}, stage=None)

    assert len(datasets["train"]) == 5
    assert datasets["train"] is datasets["val"]
    assert datasets["train"] is datasets["test"]


def test_shared_subset_indices_are_deterministic():
    source = TensorDataset(torch.arange(20))

    first = SharedSubsetStrategy(num_samples=5, seed=7).arrange(
        {"train": source}, stage=None
    )["train"]
    second = SharedSubsetStrategy(num_samples=5, seed=7).arrange(
        {"train": source}, stage=None
    )["train"]

    assert first.indices == second.indices


def test_shared_subset_rejects_too_many_samples():
    source = TensorDataset(torch.arange(3))
    strategy = SharedSubsetStrategy(num_samples=4)

    with pytest.raises(ValueError, match="exceeds the size"):
        strategy.arrange({"train": source}, stage=None)


def test_standard_strategy_can_append_validation_to_training():
    train = TensorDataset(torch.arange(3))
    val = TensorDataset(torch.arange(2))
    strategy = StandardSplitStrategy(train_with_val=True)

    datasets = strategy.arrange({"train": train, "val": val}, stage="fit")

    assert isinstance(datasets["train"], ConcatDataset)
    assert len(datasets["train"]) == 5
    assert datasets["val"] is val


def test_split_subset_samples_each_split_by_its_percentage():
    train = TensorDataset(torch.arange(20))
    val = TensorDataset(torch.arange(20))
    test = TensorDataset(torch.arange(20))

    datasets = SplitSubsetStrategy(
        train_percentage=25,
        val_percentage=50,
        test_percentage=75,
    ).arrange(
        {"train": train, "val": val, "test": test}, stage=None
    )

    assert len(datasets["train"]) == 5
    assert len(datasets["val"]) == 10
    assert len(datasets["test"]) == 15


def test_split_subset_seeds_control_splits_independently():
    datasets = {
        "train": TensorDataset(torch.arange(100)),
        "val": TensorDataset(torch.arange(100)),
        "test": TensorDataset(torch.arange(100)),
    }

    first = SplitSubsetStrategy(
        train_percentage=25,
        val_percentage=25,
        test_percentage=25,
        train_seed=1,
        val_seed=2,
        test_seed=3,
    ).arrange(datasets, stage=None)
    second = SplitSubsetStrategy(
        train_percentage=25,
        val_percentage=25,
        test_percentage=25,
        train_seed=1,
        val_seed=2,
        test_seed=3,
    ).arrange(datasets, stage=None)

    assert first["train"].indices == second["train"].indices
    assert first["val"].indices == second["val"].indices
    assert first["test"].indices == second["test"].indices
    assert first["train"].indices != first["val"].indices
    assert first["val"].indices != first["test"].indices


@pytest.mark.parametrize("percentage", [0, -1, 101])
@pytest.mark.parametrize(
    "argument", ["train_percentage", "val_percentage", "test_percentage"]
)
def test_split_subset_rejects_invalid_percentage(argument, percentage):
    with pytest.raises(ValueError, match=argument):
        SplitSubsetStrategy(**{argument: percentage})


def test_split_subset_keeps_at_least_one_sample():
    datasets = {
        "train": TensorDataset(torch.arange(3)),
        "val": TensorDataset(torch.arange(3)),
        "test": TensorDataset(torch.arange(3)),
    }

    subsets = SplitSubsetStrategy(
        train_percentage=1,
        val_percentage=1,
        test_percentage=1,
    ).arrange(datasets, stage=None)

    assert all(len(dataset) == 1 for dataset in subsets.values())


def test_split_subset_leaves_full_splits_unchanged():
    datasets = {
        "train": TensorDataset(torch.arange(3)),
        "val": TensorDataset(torch.arange(4)),
        "test": TensorDataset(torch.arange(5)),
    }

    arranged = SplitSubsetStrategy().arrange(datasets, stage=None)

    assert all(arranged[split] is dataset for split, dataset in datasets.items())
