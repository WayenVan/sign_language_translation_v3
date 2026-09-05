#!/usr/bin/env python3
"""Run one randomly selected CUDA tensor operation per interval."""

from __future__ import annotations

import argparse
import random
import signal
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--size", type=int, default=2048, help="square tensor size")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between operations")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.size <= 0 or args.interval < 0:
        raise SystemExit("--size must be positive and --interval must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this script on a GPU node.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    rng = random.Random(args.seed)
    a = torch.randn((args.size, args.size), device=device, dtype=torch.float32)
    b = torch.randn_like(a)

    operations = {
        "matmul": lambda: a @ b,
        "relu": lambda: torch.relu(a + b),
        "sin": lambda: torch.sin(a),
        "softmax": lambda: torch.softmax(a, dim=-1),
        "multiply": lambda: a * b,
    }

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"GPU: {torch.cuda.get_device_name(device)} | tensor: {args.size}x{args.size}")
    step = 0
    while running:
        started = time.monotonic()
        name = rng.choice(tuple(operations))
        result = operations[name]()
        torch.cuda.synchronize(device)
        step += 1
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        elapsed = time.monotonic() - started
        print(
            f"[{step:06d}] operation={name:<8} elapsed={elapsed:.3f}s "
            f"allocated={allocated:.1f} MiB",
            flush=True,
        )
        del result
        time.sleep(max(0.0, args.interval - elapsed))

    print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
