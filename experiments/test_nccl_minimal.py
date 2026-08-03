#!/usr/bin/env python3
"""Minimal two-GPU NCCL smoke test.

Run with, for example:

    CUDA_VISIBLE_DEVICES=1,2 \
    NCCL_DEBUG=INFO \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    torchrun --standalone --nproc-per-node=2 experiments/test_nccl_minimal.py
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def log(message: str) -> None:
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[rank={rank} local_rank={local_rank}] {message}", flush=True)


def main() -> None:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "This script must be launched with torchrun; missing environment "
            f"variables: {', '.join(missing)}"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    log(
        f"host={socket.gethostname()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"torch={torch.__version__} cuda_runtime={torch.version.cuda} "
        f"cuda_count={torch.cuda.device_count()}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process")
    if world_size != torch.cuda.device_count():
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, but this process sees "
            f"{torch.cuda.device_count()} CUDA devices"
        )

    # CUDA_VISIBLE_DEVICES remaps physical devices 1,2 to logical devices 0,1.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    log(f"bound to logical {device}: {properties.name}, uuid={properties.uuid}")

    # A short timeout avoids waiting for PyTorch's usual long NCCL timeout.
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        value = torch.tensor([float(rank + 1)], device=device)
        torch.cuda.synchronize(device)
        log(f"before all_reduce: value={value.item()}")

        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        expected = world_size * (world_size + 1) / 2
        actual = value.item()
        log(f"after all_reduce: value={actual}, expected={expected}")
        if actual != expected:
            raise RuntimeError(f"all_reduce returned {actual}, expected {expected}")

        dist.barrier()
        log("PASS: NCCL all_reduce and barrier completed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
