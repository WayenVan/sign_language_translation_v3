"""Rewrite the 0-dim gate parameters of old checkpoints as shape-(1,) tensors.

FSDP2 shards every parameter along dim 0 and refuses 0-dim ones outright
("fully_shard doesn't support scalar parameters"), so the V28 adapter gates and
``SltModel.visual_scale`` are now created as shape-(1,) tensors. Checkpoints
written before that change store them with shape ``[]``, which makes
``from_pretrained`` report a size mismatch and re-initialize them.

This rewrites the affected ``*.safetensors`` files in place. The payload is
untouched -- a 0-dim and a 1-element tensor hold the same four bytes -- so the
conversion is lossless and ``--revert`` restores the original shapes exactly.

    python -m csi_slt.commands.migrate_scalar_gates outputs/<run>/checkpoint-84000
    python -m csi_slt.commands.migrate_scalar_gates --revert outputs/<run>/checkpoint-*

Note: ``optimizer.pt`` is not migrated. Its Adam moments for these parameters
keep the old shape, so a migrated checkpoint cannot be resumed from, only
loaded for evaluation or as an initialization.
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Parameters that changed shape. Adapter gates are matched by suffix because
# they appear under several sub-module prefixes; ``visual_scale`` is a
# top-level parameter of SltModel.
GATE_SUFFIXES = (".motion_gate", ".temporal_gate", ".residual_gate")
EXACT_KEYS = ("visual_scale",)


def is_migrated_key(key: str) -> bool:
    return key in EXACT_KEYS or key.endswith(GATE_SUFFIXES)


def migrate_file(path: Path, revert: bool, force: bool) -> int:
    """Rewrite one safetensors file in place. Returns the number of keys changed."""
    want_ndim = 1 if revert else 0
    new_shape = () if revert else (1,)

    tensors: dict[str, torch.Tensor] = {}
    changed: list[str] = []
    unexpected: list[str] = []

    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if is_migrated_key(key) and tensor.ndim == want_ndim:
                tensor = tensor.reshape(new_shape)
                changed.append(key)
            elif tensor.ndim == 0:
                unexpected.append(key)
            tensors[key] = tensor

    if unexpected and not force:
        raise SystemExit(
            f"{path}: found 0-dim tensors that this migration does not know "
            f"about: {unexpected}. FSDP2 will reject them too. Re-run with "
            f"--force to rewrite only the known keys and leave these alone."
        )

    if not changed:
        print(f"  {path.name}: nothing to do")
        return 0

    # Write beside the original and swap atomically, so an interrupted run
    # cannot leave a half-written checkpoint behind.
    temporary = path.with_suffix(path.suffix + ".migrating")
    mode = path.stat().st_mode
    save_file(tensors, temporary, metadata=metadata)
    # `save_file` creates the file with the process umask; keep the original
    # permissions so a migrated checkpoint stays as readable as it was.
    os.chmod(temporary, mode)
    os.replace(temporary, path)

    for key in changed:
        print(f"  {path.name}: {key} -> {tuple(new_shape)}")
    return len(changed)


def main(args) -> None:
    total = 0
    for raw in args.checkpoint_dir:
        directory = Path(raw)
        if not directory.is_dir():
            raise SystemExit(f"not a directory: {directory}")

        files = sorted(directory.glob("*.safetensors"))
        if not files:
            raise SystemExit(f"no *.safetensors in {directory}")

        print(f"{directory}:")
        for path in files:
            total += migrate_file(path, revert=args.revert, force=args.force)

    verb = "reverted" if args.revert else "migrated"
    print(f"✅ {verb} {total} parameter(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite 0-dim gate parameters in a checkpoint as shape-(1,) "
            "tensors so the checkpoint can be loaded under FSDP2."
        )
    )
    parser.add_argument(
        "checkpoint_dir",
        nargs="+",
        help="Checkpoint directories to rewrite in place.",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Restore the original 0-dim shapes instead.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if unrecognized 0-dim tensors are present.",
    )
    main(parser.parse_args())
