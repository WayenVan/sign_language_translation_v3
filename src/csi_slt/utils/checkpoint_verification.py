"""Utilities for checking an in-memory model against a saved checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from csi_slt.modeling_slt.slt import SltModel


_LOADING_PROBLEM_KEYS = (
    "missing_keys",
    "unexpected_keys",
    "mismatched_keys",
    "error_msgs",
)


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor storage bytes, including NaN payloads and signed zero."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return torch.equal(
        left.contiguous().view(torch.uint8),
        right.contiguous().view(torch.uint8),
    )


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Return a bounded-memory maximum absolute difference for numeric tensors."""
    if not (left.is_floating_point() or left.is_complex()):
        return None

    maximum = 0.0
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    chunk_size = 1_000_000
    for start in range(0, left_flat.numel(), chunk_size):
        stop = start + chunk_size
        difference = (left_flat[start:stop] - right_flat[start:stop]).abs().max()
        maximum = max(maximum, float(difference.item()))
    return maximum


@torch.no_grad()
def verify_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare every persistent in-memory tensor with a freshly loaded model.

    The comparison is bitwise exact, including dtype and shape. The returned
    report is JSON serializable and is written when a tensor comparison fails,
    which makes failed batch jobs diagnosable from their output folder.
    """
    checkpoint_dir = Path(checkpoint_dir)
    reloaded, loading_info = SltModel.from_pretrained(
        checkpoint_dir,
        output_loading_info=True,
    )
    loading_problems = {
        key: loading_info.get(key)
        for key in _LOADING_PROBLEM_KEYS
        if loading_info.get(key)
    }

    memory_state = model.state_dict()
    checkpoint_state = reloaded.state_dict()
    memory_keys = set(memory_state)
    checkpoint_keys = set(checkpoint_state)
    missing_after_reload = sorted(memory_keys - checkpoint_keys)
    unexpected_after_reload = sorted(checkpoint_keys - memory_keys)
    differences: list[dict[str, Any]] = []

    for name in sorted(memory_keys & checkpoint_keys):
        memory_tensor = memory_state[name].detach().cpu()
        checkpoint_tensor = checkpoint_state[name].detach().cpu()
        if _bitwise_equal(memory_tensor, checkpoint_tensor):
            continue

        difference: dict[str, Any] = {
            "name": name,
            "memory_dtype": str(memory_tensor.dtype),
            "checkpoint_dtype": str(checkpoint_tensor.dtype),
            "memory_shape": list(memory_tensor.shape),
            "checkpoint_shape": list(checkpoint_tensor.shape),
        }
        if (
            memory_tensor.dtype == checkpoint_tensor.dtype
            and memory_tensor.shape == checkpoint_tensor.shape
        ):
            difference["mismatched_elements"] = int(
                torch.count_nonzero(memory_tensor != checkpoint_tensor).item()
            )
            difference["max_abs_diff"] = _max_abs_diff(
                memory_tensor, checkpoint_tensor
            )
        differences.append(difference)

    report = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "equal": not (
            loading_problems
            or missing_after_reload
            or unexpected_after_reload
            or differences
        ),
        "tensors_checked": len(memory_keys & checkpoint_keys),
        "loading_problems": loading_problems,
        "missing_after_reload": missing_after_reload,
        "unexpected_after_reload": unexpected_after_reload,
        "differences": differences,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    del reloaded
    if not report["equal"]:
        names = [item["name"] for item in differences[:20]]
        raise AssertionError(
            "Checkpoint reload differs from the in-memory model: "
            f"loading_problems={loading_problems}, "
            f"missing={missing_after_reload[:20]}, "
            f"unexpected={unexpected_after_reload[:20]}, "
            f"different_tensors={names}. Full report: {report_path}"
        )
    return report
