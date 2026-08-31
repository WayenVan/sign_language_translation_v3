"""How many hand patches survive a top-k cut, and at what k do they start to go?

AUC asks whether hand patches outrank face patches; it says no.  But a selector
never needs that ordering to be right -- it only needs the hands to fall inside
the kept set.  Those are different questions, and recall@k is the one that
matches how the score is actually used.

The attention post-processing here is copied from
``experiments/visualize_cradio_attention.py`` (edge-corrected 3x3 mean, corners
zeroed before and after) so these curves describe the pipeline that was actually
eyeballed, not the raw attention.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PATCH_SIZE = 16


def smooth_grid(attention, kernel_size):
    """Edge-corrected spatial mean: pad with zeros but divide by the valid count."""
    if kernel_size <= 1:
        return attention
    values = attention[:, None]
    padding = kernel_size // 2
    total = F.avg_pool2d(
        values, kernel_size, stride=1, padding=padding, divisor_override=1
    )
    counts = F.avg_pool2d(
        torch.ones_like(values), kernel_size, stride=1, padding=padding,
        divisor_override=1,
    )
    return (total / counts).squeeze(1)


def zero_corners(attention):
    attention = attention.clone()
    attention[:, 0, 0] = 0
    attention[:, 0, -1] = 0
    attention[:, -1, 0] = 0
    attention[:, -1, -1] = 0
    return attention


def postprocess(attention, grid_h, grid_w, kernel_size, mask_corners):
    grid = torch.as_tensor(attention).float().reshape(-1, grid_h, grid_w)
    if mask_corners:
        grid = zero_corners(grid)
    grid = smooth_grid(grid, kernel_size)
    if mask_corners:
        # Smoothing can repopulate a masked corner from its neighbours.
        grid = zero_corners(grid)
    return grid.reshape(len(grid), -1)


def recall_at_k(scores, positive, k_values):
    """Fraction of positive patches inside the per-frame top-k, over all frames."""
    order = np.argsort(-scores, axis=1)
    recalls = []
    for k in k_values:
        kept = np.zeros_like(scores, dtype=bool)
        np.put_along_axis(kept, order[:, :k], True, axis=1)
        hits = (kept & positive).sum(axis=1)
        total = positive.sum(axis=1)
        usable = total > 0
        recalls.append(float((hits[usable] / total[usable]).mean()))
    return np.array(recalls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/diagnose/features.npz")
    parser.add_argument("--labels", default="outputs/diagnose/labels_hand_vs_face.npz")
    parser.add_argument("--out-dir", default="outputs/diagnose/recall_at_k")
    parser.add_argument("--layers", type=int, nargs="*", default=[-8, -13, -16, -25])
    parser.add_argument("--smooth", type=int, default=3)
    args = parser.parse_args()

    data = np.load(args.features)
    attention_all = data["cls_attention"]  # [N, L, P]
    motion_all = data["motion"]
    layers = data["layers"].tolist()
    frames = data["frames"]
    grid_h = frames.shape[1] // PATCH_SIZE
    grid_w = frames.shape[2] // PATCH_SIZE

    part = np.load(args.labels)["labels"]  # 1 = hand, 0 = face, -1 = other
    hand = part == 1
    face = part == 0
    usable = hand.any(axis=1) & face.any(axis=1)
    print(
        f"{usable.sum()}/{len(frames)} frames have both hand and face patches; "
        f"mean {hand[usable].sum(axis=1).mean():.1f} hand / "
        f"{face[usable].sum(axis=1).mean():.1f} face patches per frame"
    )

    k_values = np.array([8, 12, 16, 24, 32, 40, 48, 64, 96])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    print(f"\n{'':<26}" + "".join(f"{k:>7}" for k in k_values))
    for layer in args.layers:
        index = layers.index(layer)
        for mask_corners, style in ((False, ":"), (True, "-")):
            scores = postprocess(
                attention_all[usable, index], grid_h, grid_w,
                args.smooth if mask_corners else 1, mask_corners,
            ).numpy()
            hand_recall = recall_at_k(scores, hand[usable], k_values)
            face_recall = recall_at_k(scores, face[usable], k_values)
            label = f"CLS {layer}" + (" +smooth+corner" if mask_corners else " raw")
            axes[0].plot(k_values, hand_recall, style, marker="o", label=label)
            axes[1].plot(k_values, face_recall, style, marker="o", label=label)
            if mask_corners:
                print(f"  hand recall {label:<24}"[:26]
                      + "".join(f"{v:>7.2f}" for v in hand_recall))
                print(f"  face recall {label:<24}"[:26]
                      + "".join(f"{v:>7.2f}" for v in face_recall))

    # Same post-processing as the attention curves, so the comparison is fair.
    motion_scores = postprocess(
        motion_all[usable, layers.index(-21)], grid_h, grid_w, args.smooth, True
    ).numpy()
    axes[0].plot(k_values, recall_at_k(motion_scores, hand[usable], k_values),
                 "--", marker="^", color="#38a169", label="motion -21")
    axes[1].plot(k_values, recall_at_k(motion_scores, face[usable], k_values),
                 "--", marker="^", color="#38a169", label="motion -21")

    for axis, name in zip(axes, ("hand", "face")):
        axis.axhline(1.0, ls=":", color="#718096")
        axis.set_xlabel("top-k")
        axis.set_ylabel(f"{name} patch recall")
        axis.set_title(f"{name} patches kept by a top-k cut")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(Path(args.out_dir) / "recall_at_k.png", dpi=150)
    print(f"\nwrote {args.out_dir}/recall_at_k.png")


if __name__ == "__main__":
    main()
