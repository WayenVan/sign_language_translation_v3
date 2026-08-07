import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from csi_slt.data.datamodule_strategies import (
    SharedSubsetStrategy,
    StandardSplitStrategy,
    TrainSubsetStrategy,
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


def test_train_subset_samples_only_training_data():
    train = TensorDataset(torch.arange(20))
    val = TensorDataset(torch.arange(4))
    test = TensorDataset(torch.arange(5))

    datasets = TrainSubsetStrategy(percentage=25, seed=7).arrange(
        {"train": train, "val": val, "test": test}, stage=None
    )

    assert len(datasets["train"]) == 5
    assert datasets["val"] is val
    assert datasets["test"] is test


def test_train_subset_indices_are_deterministic():
    train = TensorDataset(torch.arange(20))

    first = TrainSubsetStrategy(percentage=25, seed=7).arrange(
        {"train": train}, stage="fit"
    )["train"]
    second = TrainSubsetStrategy(percentage=25, seed=7).arrange(
        {"train": train}, stage="fit"
    )["train"]

    assert first.indices == second.indices


@pytest.mark.parametrize("percentage", [0, -1, 101])
def test_train_subset_rejects_invalid_percentage(percentage):
    with pytest.raises(ValueError, match="percentage"):
        TrainSubsetStrategy(percentage=percentage)


def test_train_subset_keeps_at_least_one_sample():
    train = TensorDataset(torch.arange(3))

    dataset = TrainSubsetStrategy(percentage=1).arrange(
        {"train": train}, stage="fit"
    )["train"]

    assert len(dataset) == 1
