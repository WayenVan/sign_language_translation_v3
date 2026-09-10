#!/usr/bin/env python3
"""Compare two-GPU NCCL all-reduce speed with P2P disabled and enabled."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta


def worker(tensor_mib: int, iterations: int, timeout_seconds: float) -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))

    element_count = tensor_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    value = torch.ones(element_count, device=local_rank, dtype=torch.float32)

    # Initialize NCCL communicators before the timed section.
    dist.all_reduce(value)
    torch.cuda.synchronize()
    dist.barrier()

    started = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(value)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if local_rank == 0:
        payload_bytes = value.numel() * value.element_size()
        algorithm_gbps = payload_bytes * iterations / elapsed / 1e9
        # For ring all-reduce, bus bandwidth = algorithm bandwidth * 2(N-1)/N.
        world_size = dist.get_world_size()
        bus_gbps = algorithm_gbps * 2 * (world_size - 1) / world_size
        print(
            "NCCL_BENCH_RESULT="
            + json.dumps(
                {
                    "elapsed_seconds": elapsed,
                    "iterations": iterations,
                    "tensor_mib": tensor_mib,
                    "average_ms": elapsed * 1000 / iterations,
                    "algorithm_GBps": algorithm_gbps,
                    "bus_GBps": bus_gbps,
                }
            ),
            flush=True,
        )

    dist.destroy_process_group()


def run_mode(p2p_disabled: bool, args: argparse.Namespace) -> dict[str, object]:
    mode = "disabled" if p2p_disabled else "enabled"
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1" if p2p_disabled else "0"
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        os.path.abspath(__file__),
        "--worker",
        "--tensor-mib",
        str(args.tensor_mib),
        "--iterations",
        str(args.iterations),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]

    print(f"\nTesting NCCL P2P {mode} (hard timeout: {args.timeout_seconds:g}s)...", flush=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        # Kill the whole torchrun process group so hung worker processes cannot
        # survive the controller and continue occupying the GPUs.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        if output:
            print(output.rstrip())
        print(f"Result: TIMEOUT after {args.timeout_seconds:g}s (processes terminated)")
        return {"mode": mode, "status": "timeout"}

    if output:
        print(output.rstrip())
    if process.returncode != 0:
        print(f"Result: FAILED (exit code {process.returncode})")
        return {"mode": mode, "status": "failed"}

    marker = "NCCL_BENCH_RESULT="
    result_line = next(
        (line for line in output.splitlines() if line.startswith(marker)), None
    )
    if result_line is None:
        print("Result: FAILED (worker produced no benchmark result)")
        return {"mode": mode, "status": "failed"}

    metrics = json.loads(result_line.removeprefix(marker))
    print(
        f"Result: OK, {metrics['average_ms']:.2f} ms/all-reduce, "
        f"{metrics['bus_GBps']:.2f} GB/s bus bandwidth "
        f"(total wall time {time.perf_counter() - started:.2f}s)"
    )
    return {"mode": mode, "status": "ok", **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NCCL all-reduce with NCCL_P2P_DISABLE=1 and 0."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tensor-mib", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.tensor_mib < 1 or args.iterations < 1 or args.timeout_seconds <= 0:
        parser.error("tensor-mib, iterations, and timeout-seconds must be positive")
    return args


def print_environment(torch: object) -> None:
    """Print the software versions and GPU topology relevant to NCCL P2P."""
    print("Environment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  PyTorch CUDA runtime: {torch.version.cuda}")
    try:
        nccl_version = torch.cuda.nccl.version()
    except Exception as error:  # NCCL may be absent from unusual PyTorch builds.
        nccl_version = f"unavailable ({error})"
    print(f"  NCCL: {nccl_version}")

    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=False,
        )
        driver_versions = sorted(set(driver.stdout.split()))
        print(f"  NVIDIA driver: {', '.join(driver_versions) or 'unknown'}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(f"  NVIDIA driver: unavailable ({error})")

    gpu_count = torch.cuda.device_count()
    print(f"  CUDA GPU count: {gpu_count}")
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"  GPU {index}: {properties.name}, "
            f"{properties.total_memory / 1024**3:.1f} GiB"
        )

    print("  CUDA peer access:")
    for source in range(gpu_count):
        peers = []
        for target in range(gpu_count):
            if source != target:
                peers.append(
                    f"{source}->{target}="
                    f"{torch.cuda.can_device_access_peer(source, target)}"
                )
        if peers:
            print(f"    {', '.join(peers)}")

    try:
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=False,
        )
        print("  nvidia-smi topo -m:")
        for line in topology.stdout.rstrip().splitlines():
            print(f"    {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(f"  nvidia-smi topology: unavailable ({error})")


def main() -> int:
    args = parse_args()
    if args.worker:
        worker(args.tensor_mib, args.iterations, args.timeout_seconds)
        return 0

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in the active Python environment.", file=sys.stderr)
        return 2

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print(
            f"This benchmark requires at least 2 CUDA GPUs; found {torch.cuda.device_count()}.",
            file=sys.stderr,
        )
        return 2

    print_environment(torch)
    print(
        f"\nBenchmark GPUs: {torch.cuda.get_device_name(0)} / "
        f"{torch.cuda.get_device_name(1)}\n"
        f"Workload: {args.iterations} x {args.tensor_mib} MiB all-reduce"
    )
    results = [run_mode(True, args), run_mode(False, args)]

    successful = [result for result in results if result["status"] == "ok"]
    if len(successful) == 2:
        disabled, enabled = successful
        speedup = disabled["average_ms"] / enabled["average_ms"]
        print(f"\nP2P enabled/disabled speed ratio: {speedup:.2f}x")
    elif successful:
        print("\nOnly one mode completed; no speed ratio can be calculated.")
    else:
        print("\nNeither mode completed successfully.")

    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
