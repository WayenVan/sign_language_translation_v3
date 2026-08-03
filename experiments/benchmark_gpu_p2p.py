#!/usr/bin/env python3
"""Compare NCCL communication with P2P enabled and disabled on physical GPUs 1,2.

The default invocation deliberately hides GPU 0:

    python experiments/benchmark_gpu_p2p.py

Use ``--help`` to change message sizes, iteration counts, or the selected GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import timedelta


def parse_sizes(value: str) -> list[int]:
    units = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    result: list[int] = []
    for item in value.split(","):
        item = item.strip().lower()
        multiplier = units.get(item[-1], 1)
        number = item[:-1] if item[-1] in units else item
        size = int(float(number) * multiplier)
        if size <= 0 or size % 4:
            raise argparse.ArgumentTypeError(
                f"size {item!r} must be positive and divisible by 4 bytes"
            )
        result.append(size)
    return result


def human_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def worker(args: argparse.Namespace) -> None:
    # Imported only in torchrun children, after NCCL_P2P_DISABLE has been set.
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"expected exactly two visible GPUs/ranks; world_size={world_size}, "
            f"visible_cuda_devices={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout))
    try:
        peer = 1 - local_rank
        props = torch.cuda.get_device_properties(device)
        can_peer = torch.cuda.can_device_access_peer(local_rank, peer)
        print(
            f"[rank {rank}] host={socket.gethostname()} logical_gpu={local_rank} "
            f"name={props.name!r} can_access_logical_gpu_{peer}={can_peer}",
            flush=True,
        )

        measurements = []
        for size in args.sizes:
            tensor = torch.full(
                (size // 4,), float(rank + 1), dtype=torch.float32, device=device
            )
            dist.all_reduce(tensor)
            torch.cuda.synchronize(device)
            expected = 3.0
            if tensor[0].item() != expected or tensor[-1].item() != expected:
                raise RuntimeError(f"all_reduce correctness check failed for {size} bytes")

            for _ in range(args.warmup):
                dist.all_reduce(tensor)
            torch.cuda.synchronize(device)
            dist.barrier()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                dist.all_reduce(tensor)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / args.iterations

            # Use the slowest rank. For all-reduce, bus bandwidth is
            # algorithm_bandwidth * 2*(N-1)/N (NCCL test convention).
            elapsed = torch.tensor(elapsed_ms, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            elapsed_ms = elapsed.item()
            alg_bw = size / (elapsed_ms / 1000) / 1e9
            bus_bw = alg_bw * 2 * (world_size - 1) / world_size
            if rank == 0:
                measurements.append(
                    {
                        "bytes": size,
                        "latency_ms": elapsed_ms,
                        "algorithm_GBps": alg_bw,
                        "bus_GBps": bus_bw,
                    }
                )

        if rank == 0:
            print(
                "RESULT_JSON="
                + json.dumps(
                    {
                        "p2p_disabled": os.environ.get("NCCL_P2P_DISABLE"),
                        "measurements": measurements,
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def show_topology(gpus: str) -> None:
    print("\n=== NVIDIA topology (read-only; no CUDA context is created) ===")
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found; skipping topology display")
        return
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"], text=True, capture_output=True, check=False
    )
    if result.returncode:
        print(f"nvidia-smi topo -m failed: {result.stderr.strip()}")
    else:
        print(result.stdout.rstrip())
        print(f"Benchmark selection: physical GPU(s) {gpus}")


def run_case(args: argparse.Namespace, disabled: bool) -> dict:
    label = "P2P disabled" if disabled else "P2P allowed"
    print(f"\n=== {label}: NCCL_P2P_DISABLE={int(disabled)} ===", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["NCCL_P2P_DISABLE"] = str(int(disabled))
    env["NCCL_DEBUG"] = args.nccl_debug
    env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,P2P,COLL")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        os.path.abspath(__file__),
        "--worker",
        "--sizes",
        args.sizes_text,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timeout",
        str(args.timeout),
    ]
    process = subprocess.run(command, env=env, text=True, capture_output=True)
    # NCCL commonly logs to stderr, so retain both streams in the terminal.
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    if process.returncode:
        raise RuntimeError(f"{label} run failed with exit code {process.returncode}")
    lines = [line for line in process.stdout.splitlines() if line.startswith("RESULT_JSON=")]
    if len(lines) != 1:
        raise RuntimeError(f"could not find unique benchmark result for {label}")
    return json.loads(lines[0].split("=", 1)[1])


def print_comparison(allowed: dict, disabled: dict) -> None:
    print("\n=== Comparison (two-rank NCCL all_reduce) ===")
    print(
        f"{'Size':>10}  {'P2P on ms':>10}  {'P2P off ms':>11}  "
        f"{'on GB/s':>9}  {'off GB/s':>10}  {'speedup':>8}"
    )
    for on, off in zip(allowed["measurements"], disabled["measurements"], strict=True):
        speedup = off["latency_ms"] / on["latency_ms"]
        print(
            f"{human_bytes(on['bytes']):>10}  {on['latency_ms']:10.3f}  "
            f"{off['latency_ms']:11.3f}  {on['bus_GBps']:9.2f}  "
            f"{off['bus_GBps']:10.2f}  {speedup:7.2f}x"
        )
    print("GB/s is decimal NCCL-style bus bandwidth; speedup = off latency / on latency.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2", help="physical GPU IDs (default: 1,2)")
    parser.add_argument(
        "--allow-gpu0", action="store_true", help="explicitly permit GPU 0 in --gpus"
    )
    parser.add_argument("--sizes", dest="sizes_text", default="4K,1M,64M,256M")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--nccl-debug", choices=("WARN", "INFO", "TRACE"), default="WARN",
        help="use INFO to inspect NCCL's selected transport",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.sizes = parse_sizes(args.sizes_text)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be >= 0 and --iterations must be > 0")
    gpu_ids = [item.strip() for item in args.gpus.split(",")]
    if len(gpu_ids) != 2 or any(not item.isdigit() for item in gpu_ids):
        parser.error("--gpus must contain exactly two numeric physical IDs, e.g. 1,2")
    if len(set(gpu_ids)) != 2:
        parser.error("--gpus must contain two different GPU IDs")
    if "0" in gpu_ids and not args.allow_gpu0:
        parser.error("GPU 0 is protected; pass --allow-gpu0 only if using it is intentional")

    if args.worker:
        worker(args)
        return
    show_topology(args.gpus)
    allowed = run_case(args, disabled=False)
    disabled = run_case(args, disabled=True)
    print_comparison(allowed, disabled)


if __name__ == "__main__":
    main()
