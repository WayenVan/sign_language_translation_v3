"""Can a frozen linear scorer replace the motion signal, and does it still drop hands?

``recall_at_k.py`` measured the two parameter-free scorers (CLS attention and
frame-to-frame motion) and found both of them lopsided: CLS keeps the face and
throws away three quarters of the hand patches, motion does the reverse.  The
remaining candidate was never measured that way -- a ``Linear(1152 -> 1)`` fitted
offline on MediaPipe labels and then frozen, which scores AUC 1.000 on hand vs
face but was only ever reported as an AUC.

AUC is not the statistic the selector uses, so this script re-runs the same
recall@k and kept-set composition on the fitted scorer, and then asks whether the
two proposed repairs -- smoothing the score map, and dilating the selected mask
so a kept patch pulls in its neighbours -- recover anything that is still lost.

Everything is measured on held-out videos only, so the fitted scorers are judged
on signers/frames they were not fitted on.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from recall_at_k import postprocess

PATCH_SIZE = 16


def split_by_video(video_ids, holdout_fraction, seed):
    unique_ids = np.unique(video_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    holdout_count = max(1, int(round(len(shuffled) * holdout_fraction)))
    holdout = set(shuffled[:holdout_count].tolist())
    is_test = np.isin(video_ids, list(holdout))
    return ~is_test, is_test


def fit_scorer(features, positive, negative, is_train, negative_ratio, seed):
    """Fit Linear(1152->1) on train videos; return scores for every patch."""
    rng = np.random.default_rng(seed)
    train_positive = np.flatnonzero((positive & is_train[:, None]).ravel())
    train_negative = np.flatnonzero((negative & is_train[:, None]).ravel())
    keep = min(len(train_negative), int(len(train_positive) * negative_ratio))
    train_negative = rng.choice(train_negative, size=keep, replace=False)
    index = np.concatenate([train_positive, train_negative])

    flat = features.reshape(-1, features.shape[-1])
    x = flat[index].astype(np.float32)
    y = np.concatenate([np.ones(len(train_positive)), np.zeros(len(train_negative))])
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
    model = LogisticRegression(C=0.01, max_iter=2000)
    model.fit((x - mean) / scale, y)

    scores = model.decision_function((flat.astype(np.float32) - mean) / scale)
    return scores.reshape(features.shape[:2])


def rank_normalize(scores):
    """Per-frame rank in [0, 1]; makes differently-scaled scorers combinable."""
    order = np.argsort(np.argsort(scores, axis=1), axis=1)
    return order / (scores.shape[1] - 1)


def topk_mask(scores, k):
    order = np.argsort(-scores, axis=1)
    kept = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(kept, order[:, :k], True, axis=1)
    return kept


def dilate(mask, grid_h, grid_w):
    """3x3 max over the patch grid: a kept patch pulls in its 8 neighbours."""
    values = torch.as_tensor(mask).float().reshape(-1, 1, grid_h, grid_w)
    grown = F.max_pool2d(values, 3, stride=1, padding=1)
    return grown.reshape(len(values), -1).numpy() > 0.5


def report(name, kept, hand, face, ring, background):
    """Recall of each part plus what the kept budget was actually spent on."""
    size = kept.sum(axis=1)
    def recall(target):
        total = target.sum(axis=1)
        usable = total > 0
        return float((kept & target).sum(axis=1)[usable].mean() / total[usable].mean())
    def share(target):
        return float(((kept & target).sum(axis=1) / np.maximum(size, 1)).mean())
    kept_background = (kept & background).sum(axis=1).sum()
    return {
        "scorer": name,
        "kept": float(size.mean()),
        "hand_recall": recall(hand),
        "face_recall": recall(face),
        "hand_share": share(hand),
        "face_share": share(face),
        "ring_share": share(ring),
        "background_share": share(background),
        "background_suppression": 1.0 - float(kept_background / background.sum()),
    }


def print_table(rows, title):
    print(f"\n{title}")
    header = (
        f"{'scorer':<34}{'kept':>6}{'hand R':>8}{'face R':>8}"
        f"{'hand%':>7}{'face%':>7}{'ring%':>7}{'bg%':>6}{'bg supp':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scorer']:<34}{row['kept']:>6.1f}{row['hand_recall']:>8.2f}"
            f"{row['face_recall']:>8.2f}{row['hand_share']:>7.0%}{row['face_share']:>7.0%}"
            f"{row['ring_share']:>7.0%}{row['background_share']:>6.0%}"
            f"{row['background_suppression']:>9.1%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/diagnose/features.npz")
    parser.add_argument("--labels", default="outputs/diagnose/labels_hand_vs_face.npz")
    parser.add_argument("--out", default="outputs/diagnose/weight_scorer.json")
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument("--cls-layer", type=int, default=-13)
    parser.add_argument("--motion-layer", type=int, default=-21)
    parser.add_argument("--k-values", type=int, nargs="*", default=[16, 24, 32, 48])
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.features)
    layers = data["layers"].tolist()
    video_ids = data["video_ids"]
    frames = data["frames"]
    grid_h, grid_w = frames.shape[1] // PATCH_SIZE, frames.shape[2] // PATCH_SIZE

    region = data["labels"]  # 1 = hand/face, 0 = clean background, -1 = ring
    part = np.load(args.labels)["labels"]  # 1 = hand, 0 = face, -1 = other
    hand, face = part == 1, part == 0
    ring, background = region == -1, region == 0

    is_train, is_test = split_by_video(video_ids, args.holdout_fraction, args.seed)
    has_both = hand.any(axis=1) & face.any(axis=1)
    evaluate = is_test & has_both
    print(
        f"{is_train.sum()} train / {is_test.sum()} test frames over "
        f"{len(np.unique(video_ids))} videos; {evaluate.sum()} test frames carry "
        f"both hand and face"
    )
    print(
        f"per frame: {hand[evaluate].sum(axis=1).mean():.1f} hand / "
        f"{face[evaluate].sum(axis=1).mean():.1f} face / "
        f"{ring[evaluate].sum(axis=1).mean():.1f} ring / "
        f"{background[evaluate].sum(axis=1).mean():.1f} clean background"
    )

    features = data["features"][:, layers.index(args.feature_layer)]
    signal_scores = fit_scorer(
        features, region == 1, region == 0, is_train, 3.0, args.seed
    )
    hand_scores = fit_scorer(
        features, hand, face | background, is_train, 3.0, args.seed
    )
    del features

    for name, scores, target in (
        ("signal probe", signal_scores, region == 1),
        ("hand probe", hand_scores, hand),
    ):
        mask = evaluate[:, None] & (target | background)
        print(
            f"held-out AUC, {name}: "
            f"{roc_auc_score(target[mask], scores[mask]):.4f}"
        )

    cls_raw = data["cls_attention"][:, layers.index(args.cls_layer)]
    motion_raw = data["motion"][:, layers.index(args.motion_layer)]
    smooth = lambda values: postprocess(values, grid_h, grid_w, 3, True).numpy()

    candidates = {
        f"CLS {args.cls_layer} smoothed": smooth(cls_raw[evaluate]),
        f"motion {args.motion_layer} smoothed": smooth(motion_raw[evaluate]),
        "50% motion + 50% CLS": (
            rank_normalize(smooth(motion_raw[evaluate]))
            + rank_normalize(smooth(cls_raw[evaluate]))
        ),
        "signal probe raw": signal_scores[evaluate],
        "signal probe smoothed": smooth(signal_scores[evaluate]),
        "hand probe raw": hand_scores[evaluate],
        "hand probe smoothed": smooth(hand_scores[evaluate]),
    }

    labelled = (hand[evaluate], face[evaluate], ring[evaluate], background[evaluate])
    results = {}
    for k in args.k_values:
        rows = [
            report(name, topk_mask(scores, k), *labelled)
            for name, scores in candidates.items()
        ]
        rows.append(report("oracle (hand+face)", (hand | face)[evaluate], *labelled))
        print_table(rows, f"=== top-k = {k} (held-out videos) ===")
        results[f"k={k}"] = rows

    # Do the two repairs recover the patches a top-k cut still loses?
    print("\n=== repair: seed then dilate, at a matched final budget ===")
    rows = []
    for name, scores in (
        (f"CLS {args.cls_layer}", smooth(cls_raw[evaluate])),
        ("hand probe", smooth(hand_scores[evaluate])),
    ):
        for seeds in (8, 12, 16):
            grown = dilate(topk_mask(scores, seeds), grid_h, grid_w)
            rows.append(report(f"{name}, {seeds} seeds + 3x3", grown, *labelled))
        rows.append(report(f"{name}, plain top-32", topk_mask(scores, 32), *labelled))
    print_table(rows, "")
    results["dilation"] = rows

    # How large a budget does each scorer need to keep most of the hands?
    print("\n=== budget needed for a given hand recall ===")
    for name, scores in candidates.items():
        needed = {}
        for target in (0.80, 0.90, 0.95):
            for k in range(1, 197):
                row = report(name, topk_mask(scores, k), *labelled)
                if row["hand_recall"] >= target:
                    needed[target] = k
                    break
            else:
                needed[target] = None
        print(
            f"{name:<34}"
            + "".join(
                f"  R>={target:.2f}: k={value if value else '>196':<5}"
                for target, value in needed.items()
            )
        )
        results.setdefault("budget", {})[name] = needed

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
