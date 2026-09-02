"""Fit the frozen hand-patch scorer on cached backbone patch features.

Usage::

    # default: fit on dataset/ph14_scorer_features, save to checkpoints/hand_patch_scorer
    python preprocess/train_scorer.py

    # a different feature cache, and a wider ignore ring
    python preprocess/train_scorer.py \\
        --features dataset/ph14_scorer_features_L8 --ring-radius 2 \\
        --out checkpoints/hand_patch_scorer_L8

The scorer answers one question -- is this patch a hand? -- with one linear map,
and it is used by hard per-frame top-k selection.  Two consequences run through
this script:

* **No class rebalancing.**  Positives are outnumbered roughly 1:20, but the
  class prior only moves the intercept, and an intercept is a constant added to
  every patch of a frame, so it cannot change a within-frame ranking.  Whether
  the imbalance hurt anything is answered directly by recall@k on held-out
  videos rather than pre-empted by resampling.  ``--negative-ratio`` is there to
  reproduce the 3:1 sampling of ``experiments/diagnose/weight_scorer.py`` for
  comparison, not because the fit needs it.
* **AUC is reported but is not the metric.**  A selector does not need hands
  ranked above faces in general, only kept inside the top k of their own frame,
  so recall@k and what the kept budget was spent on are what decide k.

Labels come from the frame dataset's MediaPipe landmarks:

    positive  the patch a hand joint falls in
    ring      the neighbourhood of a positive -- excluded from fitting and from
              the recall denominator; half a hand is wrong under either label,
              but at selection time such patches land between the two classes,
              which is exactly where a ranking wants them
    negative  everything else: face, torso, arms, clothing, background

Frames where MediaPipe found no hand are dropped entirely.  They mean "not
detected", not "no hand", so their patches cannot be trusted as negatives.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from csi_slt.configuration_slt.configuration_scorer import (  # noqa: E402
    HandPatchScorerConfig,
)
from csi_slt.modeling_slt.scorer import HandPatchScorer  # noqa: E402

POSITIVE, NEGATIVE, RING = 1, 0, -1


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def patch_labels(frames: pd.DataFrame, side: int, ring_radius: int) -> np.ndarray:
    """Per-patch labels for every frame: ``[N, side * side]`` int8.

    Landmark coordinates are normalized to the stored crop, so the patch a joint
    falls in is just its coordinate times the grid side.
    """
    count = len(frames)
    labels = np.full((count, side * side), NEGATIVE, dtype=np.int8)
    rows = np.arange(side * side) // side
    columns = np.arange(side * side) % side

    for index, (xs, ys) in enumerate(zip(frames["hand_x"], frames["hand_y"])):
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        column = np.clip((x[valid] * side).astype(int), 0, side - 1)
        row = np.clip((y[valid] * side).astype(int), 0, side - 1)
        positive = np.unique(row * side + column)

        if ring_radius > 0:
            # Chebyshev neighbourhood on the patch grid, positives removed after.
            distance = np.maximum(
                np.abs(rows[None, :] - rows[positive][:, None]),
                np.abs(columns[None, :] - columns[positive][:, None]),
            )
            ring = np.flatnonzero((distance <= ring_radius).any(axis=0))
            labels[index, ring] = RING
        labels[index, positive] = POSITIVE
    return labels


def load_split(args, split: str) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    """Features, per-patch labels and the frame index, aligned and verified."""
    meta = json.loads((args.features / "meta.json").read_text())
    info = meta["splits"][split]
    frames = pd.read_parquet(args.dataset / f"{split}.parquet")
    digest = hashlib.sha1("\n".join(frames["id"].tolist()).encode()).hexdigest()
    if digest != info["id_sha1"]:
        raise RuntimeError(
            f"{split}.parquet does not match the features in {args.features}: the "
            "frame dataset was regenerated, so every feature would carry the "
            "wrong label. Re-run preprocess/extract_scorer_features.py."
        )

    features = np.load(args.features / info["file"], mmap_mode="r")
    if len(features) != len(frames):
        raise RuntimeError(f"{split}: {len(features)} feature rows, {len(frames)} rows")

    side = math.isqrt(info["patches_per_frame"])
    if side * side != info["patches_per_frame"]:
        raise RuntimeError("patch count is not a square grid; labels need a grid")
    labels = patch_labels(frames, side, args.ring_radius)
    # "Not detected" is not "no hand", so these frames cannot supply negatives.
    detected = frames["num_hands_detected"].to_numpy() > 0
    return features, labels, detected


def flatten(features, labels, detected, negative_ratio: float, seed: int):
    """Flatten to patch rows, dropping ring patches and undetected frames."""
    keep_frames = np.flatnonzero(detected)
    labels = labels[keep_frames]
    usable = labels != RING
    frame_of_patch = np.repeat(keep_frames, usable.sum(axis=1))

    if negative_ratio > 0:
        rng = np.random.default_rng(seed)
        positive_count = int((labels == POSITIVE).sum())
        negative_flat = np.flatnonzero((labels == NEGATIVE).ravel())
        drop = negative_flat[
            rng.permutation(len(negative_flat))[int(positive_count * negative_ratio) :]
        ]
        usable.ravel()[drop] = False
        frame_of_patch = np.repeat(keep_frames, usable.sum(axis=1))

    # Read the memmap frame by frame: a fancy-index over 5 GB would materialize
    # the whole array first.
    chunks = [
        features[frame][usable[position]]
        for position, frame in enumerate(tqdm(keep_frames, desc="gather", leave=False))
    ]
    x = torch.from_numpy(np.concatenate(chunks))
    y = torch.from_numpy(labels[usable].astype(np.float32))
    return x, y, frame_of_patch


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, features, labels, detected, k_values, device, batch=256):
    """Per-frame recall@k and how the kept budget was spent, on whole frames."""
    keep = np.flatnonzero(detected)
    stats = {
        k: {"recall": [], "hand": [], "ring": [], "negative": []} for k in k_values
    }
    scores_all, targets_all = [], []
    for start in range(0, len(keep), batch):
        rows = keep[start : start + batch]
        x = torch.from_numpy(np.asarray(features[rows])).to(device)
        y = torch.from_numpy(labels[rows]).to(device)
        scores = model(x.float())
        scores_all.append(scores[y != RING].flatten().cpu())
        targets_all.append((y[y != RING] == POSITIVE).flatten().cpu())

        hand = y == POSITIVE
        for k in k_values:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(
                -1, scores.topk(min(k, scores.shape[-1]), dim=-1).indices, True
            )
            total = hand.sum(dim=1).clamp(min=1)
            # Macro: every frame counts once, because one ROI token is built per
            # frame and a frame that loses its hand yields a useless token.
            stats[k]["recall"].append(((mask & hand).sum(1) / total).cpu())
            kept = mask.sum(1).clamp(min=1)
            stats[k]["hand"].append(((mask & hand).sum(1) / kept).cpu())
            stats[k]["ring"].append(((mask & (y == RING)).sum(1) / kept).cpu())
            stats[k]["negative"].append(((mask & (y == NEGATIVE)).sum(1) / kept).cpu())

    scores = torch.cat(scores_all).double()
    targets = torch.cat(targets_all)
    # AUC as the rank statistic, without pulling in sklearn for one number.
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    positives, negatives = int(targets.sum()), int((~targets).sum())
    auc = (ranks[targets].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )

    report = {"auc": float(auc)}
    for k in k_values:
        recall = torch.cat(stats[k]["recall"])
        report[f"k={k}"] = {
            "recall_macro": float(recall.mean()),
            "frames_below_half": float((recall < 0.5).float().mean()),
            "hand_share": float(torch.cat(stats[k]["hand"]).mean()),
            "ring_share": float(torch.cat(stats[k]["ring"]).mean()),
            "negative_share": float(torch.cat(stats[k]["negative"]).mean()),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/ph14_train_scorer_dataset")
    )
    parser.add_argument(
        "--features", type=Path, default=Path("dataset/ph14_scorer_features")
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/hand_patch_scorer"))
    parser.add_argument("--ring-radius", type=int, default=1)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.0,
        help="negatives per positive; 0 keeps them all",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k-values", type=int, nargs="+", default=[16, 24, 32, 48])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    accelerator = Accelerator()
    torch.manual_seed(args.seed)

    features, labels, detected = load_split(args, "train")
    x, y, _ = flatten(features, labels, detected, args.negative_ratio, args.seed)
    accelerator.print(
        f"train: {len(x)} patches, {int(y.sum())} positive "
        f"({y.mean():.1%}), from {int(detected.sum())}/{len(detected)} frames"
    )

    meta = json.loads((args.features / "meta.json").read_text())
    backbone_meta = meta["backbone"]
    config = HandPatchScorerConfig(
        input_dim=x.shape[-1],
        patch_grid_size=tuple(meta["splits"]["train"]["patch_grid"]),
        # Enough to rebuild the feature extractor these coefficients only make
        # sense against: the class, plus the literal kwargs of the constructor
        # every registry backbone is built through.
        visual_backbone_class=backbone_meta["class"],
        visual_backbone_init_kwargs={
            "config": backbone_meta["config"],
            "dtype": backbone_meta["dtype"].removeprefix("torch."),
        },
    )
    model = HandPatchScorer(config)
    # Statistics of the fitting set, installed before the linear layer trains
    # against them so it is not chasing a moving target.
    model.set_feature_statistics(x.float().mean(0), x.float().std(0).clamp_min(1e-6))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    for epoch in range(args.epochs):
        total, seen = 0.0, 0
        for batch_x, batch_y in tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        ):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch_x.float()), batch_y
            )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += loss.item() * len(batch_y)
            seen += len(batch_y)
        accelerator.print(f"epoch {epoch + 1}: loss {total / seen:.4f}")

    model = accelerator.unwrap_model(model).eval()
    test_features, test_labels, test_detected = load_split(args, "test")
    report = evaluate(
        model,
        test_features,
        test_labels,
        test_detected,
        args.k_values,
        accelerator.device,
    )
    accelerator.print(
        f"\nheld-out AUC (hand vs everything, ring excluded): {report['auc']:.4f}"
    )
    accelerator.print(
        f"{'k':>4}{'recall':>9}{'frames<0.5':>12}{'hand%':>8}{'ring%':>8}{'other%':>8}"
    )
    for k in args.k_values:
        row = report[f"k={k}"]
        accelerator.print(
            f"{k:>4}{row['recall_macro']:>9.3f}{row['frames_below_half']:>12.1%}"
            f"{row['hand_share']:>8.1%}{row['ring_share']:>8.1%}{row['negative_share']:>8.1%}"
        )

    if accelerator.is_main_process:
        args.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out)
        (args.out / "fit_report.json").write_text(
            json.dumps(
                {"args": {k: str(v) for k, v in vars(args).items()}, "eval": report},
                indent=2,
            )
            + "\n"
        )
        accelerator.print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
