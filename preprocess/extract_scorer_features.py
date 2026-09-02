"""Cache visual-backbone patch features for the hand-patch scorer's fitting set.

Usage::

    # default: C-RADIOv4-SO400M last layer over dataset/ph14_train_scorer_dataset
    python preprocess/extract_scorer_features.py

    # another backbone or layer; --backbone is a key of VISUAL_BACKBONES and
    # --backbone-config is passed straight to its from_pretrained_backbone
    python preprocess/extract_scorer_features.py \
        --backbone c_radio_v4 \
        --backbone-config '{"id": "nvidia/C-RADIOv4-SO400M", "output_layer": -8}' \
        --out dataset/ph14_scorer_features_L8

    # smoke test: 128 frames per split, marked truncated in meta.json
    python preprocess/extract_scorer_features.py --limit-frames 128 --out /tmp/check

The scorer is one linear map over a single patch feature, so fitting it is a
matter of seconds -- but every fit needs the same 2.7 million patch vectors, and
recomputing them through a 400M-parameter backbone each time is the only
expensive part of the loop.  This script runs that pass once.

Every patch of every frame is stored, unsampled and unlabelled.  Labels depend
on where the ignore ring is drawn and sampling depends on the labels, so baking
either one in here would mean re-running the backbone to revisit a decision that
is otherwise a second of numpy.  The 3.4 GB that sampling would have saved is
not worth that.

Frames are read in the frame dataset's parquet order and written in the same
order, so ``features[i]`` is the frame in row ``i`` of ``<split>.parquet``.
Nothing binds a frame id to its row physically; instead ``meta.json`` records a
digest of the id column, which turns "the parquet was regenerated and every
feature now carries the wrong label" from a silent corruption into a failed
assertion.  Consumers should check it::

    meta = json.load(open(f"{features}/meta.json"))
    ids = pd.read_parquet(f"{dataset}/train.parquet")["id"].tolist()
    digest = hashlib.sha1("\\n".join(ids).encode()).hexdigest()
    assert digest == meta["splits"]["train"]["id_sha1"]

Output::

    dataset/ph14_scorer_features/
        meta.json          backbone class and kwargs, feature shape, alignment digests
        train.npy          [N, P, C] float16
        test.npy
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from csi_slt.modeling_slt.registry import VISUAL_BACKBONES  # noqa: E402

DEFAULT_BACKBONE_CONFIG = '{"id": "nvidia/C-RADIOv4-SO400M", "output_layer": -1}'


class FrameDataset(Dataset):
    """Frames of one split, in parquet order, as RGB floats in [0, 1].

    C-RADIO applies its own input conditioner and rejects externally normalized
    tensors, so no mean/std is applied here -- only the 255 scaling.
    """

    def __init__(self, dataset_root: Path, paths: list[str]) -> None:
        self.dataset_root = dataset_root
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.dataset_root / self.paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"failed to read {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


def build_backbone(name: str, config: dict, dtype: torch.dtype, device: str):
    if name not in VISUAL_BACKBONES:
        raise ValueError(
            f"unknown backbone {name!r}; registered: {sorted(VISUAL_BACKBONES)}"
        )
    backbone_class = VISUAL_BACKBONES[name]
    backbone = backbone_class.from_pretrained_backbone(config=config, dtype=dtype)
    return backbone.to(device).eval()


def git_revision() -> str | None:
    """Best effort: the checkout may not be a git repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@torch.no_grad()
def extract_split(backbone, args, split: str) -> dict:
    """Run one split through the backbone into a float16 ``.npy``."""
    frames = pd.read_parquet(args.dataset / f"{split}.parquet")
    if args.limit_frames:
        frames = frames.head(args.limit_frames)
    ids = frames["id"].tolist()
    loader = DataLoader(
        FrameDataset(args.dataset, frames["path"].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    destination = args.out / f"{split}.npy"
    memmap = None
    written = 0
    for batch in tqdm(loader, desc=f"{split}"):
        batch = batch.to(args.device, dtype=args.torch_dtype, non_blocking=True)
        # t_lengths is optional and the backbone is per-frame, so the batch does
        # not need to be presented as a packed video.
        features = backbone(batch).visual_features
        features = features.float().cpu().numpy().astype(np.float16)
        if memmap is None:
            # Allocated from the first batch rather than from the config: this
            # records the patch count and width that actually came out.
            memmap = np.lib.format.open_memmap(
                destination,
                mode="w+",
                dtype=np.float16,
                shape=(len(frames), features.shape[1], features.shape[2]),
            )
        memmap[written : written + len(features)] = features
        written += len(features)
    if memmap is None:
        raise RuntimeError(f"{split} split is empty")
    memmap.flush()
    patch_count, feature_dim = memmap.shape[1], memmap.shape[2]
    del memmap

    side = math.isqrt(patch_count)
    print(
        f"wrote {destination}: {written} frames x {patch_count} patches x "
        f"{feature_dim} dims ({destination.stat().st_size / 1e9:.2f} GB)"
    )
    return {
        "file": destination.name,
        "num_rows": written,
        "patches_per_frame": patch_count,
        "feature_dim": feature_dim,
        "patch_grid": [side, side] if side * side == patch_count else None,
        "source_parquet": f"{split}.parquet",
        "id_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest(),
        "first_id": ids[0],
        "last_id": ids[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/ph14_train_scorer_dataset")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("dataset/ph14_scorer_features")
    )
    parser.add_argument(
        "--backbone",
        default="c_radio_v4",
        help=f"registry key; one of {sorted(VISUAL_BACKBONES)}",
    )
    parser.add_argument(
        "--backbone-config",
        default=DEFAULT_BACKBONE_CONFIG,
        help="JSON dict passed to the backbone's from_pretrained_backbone",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="truncate each split, for smoke tests; marks the output truncated",
    )
    args = parser.parse_args()
    args.torch_dtype = getattr(torch, args.dtype)
    backbone_config = json.loads(args.backbone_config)

    dataset_info_path = args.dataset / "dataset_info.json"
    if not dataset_info_path.exists():
        raise FileNotFoundError(
            f"{dataset_info_path} not found; build the frame dataset first with "
            "preprocess/build_scorer_dataset.py"
        )
    dataset_info = json.loads(dataset_info_path.read_text())

    backbone = build_backbone(
        args.backbone, backbone_config, args.torch_dtype, args.device
    )
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {split: extract_split(backbone, args, split) for split in args.splits}

    backbone_class = type(backbone)
    meta = {
        "backbone": {
            # Taken from the instance so it cannot drift from a hand-written
            # string when the module is moved.
            "class": f"{backbone_class.__module__}.{backbone_class.__qualname__}",
            "registry_key": args.backbone,
            "constructor": "from_pretrained_backbone",
            "config": backbone_config,
            "dtype": str(args.torch_dtype),
        },
        "features": {
            "stored_dtype": "float16",
            "row_alignment": (
                "features[i] is row i of the split's parquet, read in file order; "
                "verify with id_sha1 before trusting any label join"
            ),
        },
        "input": {
            "frames": dataset_info["source"]["transform"],
            "do_normalize": False,
            "note": "C-RADIO applies its own input conditioner; pixels are [0, 1] RGB",
        },
        "source_dataset": {
            "path": str(args.dataset),
            "dataset_info_created": dataset_info["created"],
        },
        "splits": splits,
        "truncated": bool(args.limit_frames),
        "versions": {
            "torch": torch.__version__,
            "csi_slt_git_sha": git_revision(),
        },
    }
    path = args.out / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
