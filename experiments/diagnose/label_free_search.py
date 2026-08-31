"""Best label-free selector, found by search rather than by guessing a pair.

``weight_scorer.py`` showed a fitted scorer wins, but fitting it needs MediaPipe.
This asks what is reachable with no annotation at all.  Four families are
label-free, each available at every block:

  cls      CLS-to-patch attention                      -- finds the signer's face
  motion   ||f(t) - f(t-1)||                           -- finds whatever moves
  dev      ||f(p) - mean_p f||, deviation from frame   -- generic saliency
  bgsub    ||f(p,t) - median_t f(p)||, per-patch        -- background subtraction;
           temporal median over the video                unlike motion it survives
                                                         a static hold

Candidates are rank-normalised per frame so they can be mixed, then pairs and
triples are searched over a weight grid.  **Every configuration is chosen on the
training videos and reported on the held-out ones**, because a combination picked
directly on the test frames would report its own selection noise.  MediaPipe
labels appear only in the evaluation, never in a scorer.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from recall_at_k import postprocess
from weight_scorer import print_table, report, split_by_video, topk_mask

PATCH_SIZE = 16


def rank_normalize(scores):
    order = np.argsort(np.argsort(scores, axis=1), axis=1)
    return (order / (scores.shape[1] - 1)).astype(np.float32)


def recall(scores, target, k):
    kept = topk_mask(scores, k)
    total = target.sum(axis=1)
    usable = total > 0
    return float(((kept & target).sum(axis=1)[usable] / total[usable]).mean())


def weight_grid(count, step):
    """Points on the simplex, so the weights always sum to one."""
    if count == 1:
        yield (1.0,)
        return
    steps = int(round(1 / step))
    for head in range(steps + 1):
        for tail in weight_grid(count - 1, step):
            if count == 2:
                yield (head * step, 1 - head * step)
                break
            remaining = 1 - head * step
            if remaining < -1e-9:
                continue
            yield (head * step,) + tuple(value * remaining for value in tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/diagnose/features.npz")
    parser.add_argument("--labels", default="outputs/diagnose/labels_hand_vs_face.npz")
    parser.add_argument("--out", default="outputs/diagnose/label_free_search.json")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--report-k", type=int, nargs="*", default=[24, 32, 48])
    parser.add_argument("--keep-per-family", type=int, default=4)
    args = parser.parse_args()

    data = np.load(args.features)
    layers = data["layers"].tolist()
    video_ids = data["video_ids"]
    frames = data["frames"]
    grid_h, grid_w = frames.shape[1] // PATCH_SIZE, frames.shape[2] // PATCH_SIZE

    region = data["labels"]
    part = np.load(args.labels)["labels"]
    hand, face, ring, background = part == 1, part == 0, region == -1, region == 0

    is_train, is_test = split_by_video(video_ids, 0.3, 0)
    has_both = hand.any(axis=1) & face.any(axis=1)
    fit_on, test_on = is_train & has_both, is_test & has_both
    print(f"{fit_on.sum()} train / {test_on.sum()} held-out frames with hand and face")

    print("building label-free candidates ...")
    candidates = {}
    cls_all, motion_all = data["cls_attention"], data["motion"]
    for index, layer in enumerate(layers):
        candidates[f"cls {layer}"] = cls_all[:, index]
        candidates[f"motion {layer}"] = motion_all[:, index]
    for index, layer in enumerate(layers):
        features = data["features"][:, index].astype(np.float32)
        candidates[f"dev {layer}"] = np.linalg.norm(
            features - features.mean(axis=1, keepdims=True), axis=-1
        )
        reference = np.stack(
            [
                np.median(features[video_ids == video], axis=0)
                for video in video_ids
            ]
        )
        candidates[f"bgsub {layer}"] = np.linalg.norm(features - reference, axis=-1)
        del features

    # Same post-processing the eyeballed pipeline uses; these scorers are noisy.
    normalized = {
        name: rank_normalize(postprocess(values, grid_h, grid_w, 3, True).numpy())
        for name, values in candidates.items()
    }

    singles = []
    for name, scores in normalized.items():
        singles.append(
            {
                "name": name,
                "family": name.split()[0],
                "train_hand": recall(scores[fit_on], hand[fit_on], args.k),
                "train_face": recall(scores[fit_on], face[fit_on], args.k),
            }
        )
    print(f"\n=== best single scorer per family (train, k={args.k}) ===")
    shortlist = []
    for family in ("cls", "motion", "dev", "bgsub"):
        members = sorted(
            [row for row in singles if row["family"] == family],
            key=lambda row: -row["train_hand"],
        )
        for row in members[: args.keep_per_family]:
            shortlist.append(row["name"])
        best_face = max(
            (row for row in singles if row["family"] == family),
            key=lambda row: row["train_face"],
        )
        shortlist.append(best_face["name"])
        print(
            f"  {family:<7} best hand: {members[0]['name']:<12}"
            f" hand {members[0]['train_hand']:.2f} face {members[0]['train_face']:.2f}"
            f"   |  best face: {best_face['name']:<12}"
            f" hand {best_face['train_hand']:.2f} face {best_face['train_face']:.2f}"
        )
    shortlist = sorted(set(shortlist))
    print(f"\nshortlist ({len(shortlist)}): {', '.join(shortlist)}")

    objectives = {
        "max hand recall": lambda h, f: h,
        "balanced (min of hand, face)": lambda h, f: min(h, f),
    }
    best = {name: {"score": -1.0} for name in objectives}

    def consider(names, weights):
        stacked = sum(
            weight * normalized[name][fit_on] for name, weight in zip(names, weights)
        )
        hand_recall = recall(stacked, hand[fit_on], args.k)
        face_recall = recall(stacked, face[fit_on], args.k)
        for objective, function in objectives.items():
            value = function(hand_recall, face_recall)
            if value > best[objective]["score"]:
                best[objective] = {
                    "score": value, "names": names, "weights": weights,
                    "train_hand": hand_recall, "train_face": face_recall,
                }

    print("\nsearching pairs and triples on the training videos ...")
    for name in shortlist:
        consider((name,), (1.0,))
    for pair in combinations(shortlist, 2):
        for weights in weight_grid(2, 0.1):
            consider(pair, weights)
    for triple in combinations(shortlist, 3):
        for weights in weight_grid(3, 0.25):
            consider(triple, weights)

    results = {}
    for objective, chosen in best.items():
        recipe = " + ".join(
            f"{weight:.0%} {name}"
            for name, weight in zip(chosen["names"], chosen["weights"])
            if weight > 0
        )
        print(f"\n=== {objective} ===\n  chosen on train: {recipe}")
        stacked_test = sum(
            weight * normalized[name][test_on]
            for name, weight in zip(chosen["names"], chosen["weights"])
        )
        rows = []
        for k in args.report_k:
            rows.append(
                report(
                    f"{recipe}, k={k}",
                    topk_mask(stacked_test, k),
                    hand[test_on], face[test_on], ring[test_on], background[test_on],
                )
            )
        print_table(rows, "  held-out:")
        results[objective] = {"recipe": recipe, "train": chosen, "held_out": rows}

    print("\n=== reference points on the same held-out frames ===")
    reference_rows = []
    for name in ("cls -13", "cls -25", "motion -21", "bgsub -21", "dev -23"):
        if name not in normalized:
            continue
        for k in (32, 48):
            reference_rows.append(
                report(
                    f"{name}, k={k}",
                    topk_mask(normalized[name][test_on], k),
                    hand[test_on], face[test_on], ring[test_on], background[test_on],
                )
            )
    print_table(reference_rows, "")
    results["reference"] = reference_rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
