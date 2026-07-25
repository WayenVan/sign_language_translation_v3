from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import chain

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset
from transformers.trainer_pt_utils import get_length_grouped_indices


def get_dataset_lengths(dataset: Dataset) -> list[int]:
    """递归取得 Dataset、Subset 和 ConcatDataset 的样本长度。"""

    if isinstance(dataset, Subset):
        base_lengths = get_dataset_lengths(dataset.dataset)
        return [base_lengths[int(index)] for index in dataset.indices]

    if isinstance(dataset, ConcatDataset):
        return list(
            chain.from_iterable(
                get_dataset_lengths(sub_dataset) for sub_dataset in dataset.datasets
            )
        )

    lengths = getattr(dataset, "lengths", None)
    if lengths is None:
        raise TypeError(f"{type(dataset).__name__} must expose a `lengths` attribute.")

    lengths = [int(length) for length in lengths]

    if len(lengths) != len(dataset):
        raise ValueError(
            f"Length metadata mismatch: {len(lengths)=}, dataset_size={len(dataset)}."
        )

    return lengths


class GlobalLengthBucketSampler(Sampler[int]):
    """
    先针对整个 DDP micro-batch 分桶，再将它排列成连续的 per-rank batch。

    Accelerate 后续会把连续的 batch 分发给不同进程，因此这里不进行
    distributed rank slicing。
    """

    def __init__(
        self,
        lengths: Sequence[int],
        per_device_batch_size: int,
        num_processes: int,
        seed: int = 0,
        drop_last: bool = True,
        balance_batches: bool = True,
    ) -> None:
        if per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive.")

        if num_processes <= 0:
            raise ValueError("num_processes must be positive.")

        self.lengths = [int(length) for length in lengths]
        self.per_device_batch_size = per_device_batch_size
        self.num_processes = num_processes
        self.global_batch_size = per_device_batch_size * num_processes
        self.seed = seed
        self.drop_last = drop_last
        self.balance_batches = balance_batches
        self.epoch = 0

        if drop_last:
            self.num_samples = (
                len(self.lengths) // self.global_batch_size
            ) * self.global_batch_size
        else:
            self.num_samples = len(self.lengths)

        if self.num_samples == 0:
            raise ValueError(
                "Dataset is smaller than one global batch: "
                f"dataset_size={len(self.lengths)}, "
                f"global_batch_size={self.global_batch_size}."
            )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # 关键点：这里必须传 global_batch_size，而不是单卡 batch size。
        indices = get_length_grouped_indices(
            lengths=self.lengths,
            batch_size=self.global_batch_size,
            generator=generator,
        )

        if self.drop_last:
            indices = indices[: self.num_samples]

        if not self.balance_batches:
            return iter(indices)

        balanced_indices: list[int] = []

        for start in range(0, len(indices), self.global_batch_size):
            global_batch = indices[start : start + self.global_batch_size]

            # drop_last=False 时，最后可能是不完整的 global batch。
            if len(global_batch) != self.global_batch_size:
                balanced_indices.extend(global_batch)
                continue

            rank_batches = self._balance_global_batch(global_batch)

            # 避免物理 rank 0 每一步都拿到包含最长样本的 batch。
            rank_order = torch.randperm(
                self.num_processes,
                generator=generator,
            ).tolist()

            for rank_slot in rank_order:
                balanced_indices.extend(rank_batches[rank_slot])

        return iter(balanced_indices)

    def _balance_global_batch(
        self,
        global_batch: list[int],
    ) -> list[list[int]]:
        """
        使用贪心方式平衡各个 rank 的总帧数。

        例如长度：
            [300, 290, 280, 150, 140, 130]

        会近似分成：
            [300, 130]
            [290, 140]
            [280, 150]
        """

        sorted_indices = sorted(
            global_batch,
            key=lambda index: self.lengths[index],
            reverse=True,
        )

        rank_batches: list[list[int]] = [[] for _ in range(self.num_processes)]
        rank_loads = [0 for _ in range(self.num_processes)]

        for index in sorted_indices:
            available_ranks = [
                rank
                for rank in range(self.num_processes)
                if len(rank_batches[rank]) < self.per_device_batch_size
            ]

            target_rank = min(
                available_ranks,
                key=lambda rank: (
                    rank_loads[rank],
                    len(rank_batches[rank]),
                    rank,
                ),
            )

            rank_batches[target_rank].append(index)
            rank_loads[target_rank] += self.lengths[index]

        return rank_batches
