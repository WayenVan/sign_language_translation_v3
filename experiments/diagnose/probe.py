"""Answer "are hand/face patches linearly separable from background, and where?".

For every cached layer this fits a logistic probe on patch features and reports
ROC AUC, then renders four panels:

  1. AUC per layer against a position-only floor.  A layer only carries usable
     appearance information where it clears that floor: in signing footage the
     hands sit in a predictable part of the frame, so patch coordinates alone
     already score well above 0.5.
  2. Mean pairwise patch cosine similarity per layer (token uniformity) beside a
     Fisher ratio, to separate "features are uninformative" from "features have
     collapsed onto a shared component".
  3. Probe-score histograms per class.  This is the direct read on separability:
     two clean lobes means separable, one overlapping lump means not.
  4. Probe scores painted back onto frames, so a mediocre AUC can be traced to
     which regions the probe actually gets wrong.
  5. Candidate selector comparison.  CLS attention is only one way to rank
     patches; a frozen linear probe, frame-to-frame motion and feature-norm
     deviation are equally free.  All four are scored against the same held-out
     labels, so "should the selector use CLS at all" becomes a measurement.

Splits group on video id.  Splitting on patches or frames leaks neighbouring
patches of the same frame across the split and pins every layer near AUC 1.0.
"""

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

PATCH_SIZE = 16


def split_by_video(video_ids, holdout_fraction, seed):
    unique_ids = np.unique(video_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    holdout_count = max(1, int(round(len(shuffled) * holdout_fraction)))
    holdout = set(shuffled[:holdout_count].tolist())
    is_test = np.isin(video_ids, list(holdout))
    return ~is_test, is_test


def fit_probe(train_x, train_y, test_x, test_y, negative_ratio, seed):
    """Fit on class-balanced train data, score on the untouched test balance."""
    rng = np.random.default_rng(seed)
    positive_index = np.flatnonzero(train_y == 1)
    negative_index = np.flatnonzero(train_y == 0)
    keep = min(len(negative_index), int(len(positive_index) * negative_ratio))
    negative_index = rng.choice(negative_index, size=keep, replace=False)
    index = np.concatenate([positive_index, negative_index])

    mean = train_x[index].mean(axis=0)
    scale = train_x[index].std(axis=0) + 1e-6
    model = LogisticRegression(C=0.01, max_iter=2000)
    model.fit((train_x[index] - mean) / scale, train_y[index])

    scores = model.decision_function((test_x - mean) / scale)
    return scores, roc_auc_score(test_y, scores), average_precision_score(test_y, scores)


def token_uniformity(features):
    """Mean off-diagonal cosine similarity between patches, averaged over frames."""
    normalized = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-6)
    patch_count = normalized.shape[1]
    gram = normalized @ np.transpose(normalized, (0, 2, 1))
    off_diagonal = gram.sum(axis=(1, 2)) - np.trace(gram, axis1=1, axis2=2)
    return float((off_diagonal / (patch_count * (patch_count - 1))).mean())


def fisher_ratio(features, labels):
    """Between-class centre distance over pooled within-class spread."""
    positive = features[labels == 1]
    negative = features[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    gap = np.linalg.norm(positive.mean(axis=0) - negative.mean(axis=0))
    spread = np.sqrt(positive.var(axis=0).sum() + negative.var(axis=0).sum())
    return float(gap / (spread + 1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/diagnose/features.npz")
    parser.add_argument(
        "--labels",
        default=None,
        help="npz with a replacement `labels` array, e.g. a hand-vs-face task",
    )
    parser.add_argument("--positive-name", default="hand/face")
    parser.add_argument("--negative-name", default="background")
    parser.add_argument("--out-dir", default="outputs/diagnose")
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--show-layers", type=int, nargs="*", default=None)
    parser.add_argument("--num-overlay-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.features)
    features_all = data["features"]  # [N, L, P, D]
    layers = data["layers"]
    labels_all = data["labels"]  # [N, P]
    if args.labels:
        labels_all = np.load(args.labels)["labels"]
    frames = data["frames"]
    video_ids = data["video_ids"]
    cls_attention = data["cls_attention"] if "cls_attention" in data else None
    motion = data["motion"] if "motion" in data else None

    frame_count, layer_count, patch_count, _ = features_all.shape
    grid_h = frames.shape[1] // PATCH_SIZE
    grid_w = frames.shape[2] // PATCH_SIZE

    is_train_frame, is_test_frame = split_by_video(
        video_ids, args.holdout_fraction, args.seed
    )
    print(
        f"{frame_count} frames, {layer_count} layers, {patch_count} patches; "
        f"train {is_train_frame.sum()} / test {is_test_frame.sum()} frames "
        f"split over {len(np.unique(video_ids))} videos"
    )

    keep = labels_all >= 0  # drop the ignore ring
    train_mask = keep & is_train_frame[:, None]
    test_mask = keep & is_test_frame[:, None]
    train_y = labels_all[train_mask].astype(np.int64)
    test_y = labels_all[test_mask].astype(np.int64)

    # Position-only floor: the probe may only claim credit above this.
    rows, cols = np.divmod(np.arange(patch_count), grid_w)
    coordinates = np.stack(
        [rows / grid_h, cols / grid_w, (rows / grid_h) ** 2, (cols / grid_w) ** 2],
        axis=-1,
    )
    coordinates = np.broadcast_to(
        coordinates, (frame_count, patch_count, 4)
    ).astype(np.float32)
    _, position_auc, _ = fit_probe(
        coordinates[train_mask], train_y,
        coordinates[test_mask], test_y,
        args.negative_ratio, args.seed,
    )

    metrics = {"layers": layers.tolist(), "position_auc": position_auc, "per_layer": []}
    test_scores_by_layer = {}
    for layer_index, layer in enumerate(layers):
        features = features_all[:, layer_index].astype(np.float32)
        scores, auc, ap = fit_probe(
            features[train_mask], train_y,
            features[test_mask], test_y,
            args.negative_ratio, args.seed,
        )
        test_scores_by_layer[int(layer)] = scores
        entry = {
            "layer": int(layer),
            "auc": auc,
            "average_precision": ap,
            "token_uniformity": token_uniformity(features[is_test_frame]),
            "fisher_ratio": fisher_ratio(features[test_mask], test_y),
        }
        metrics["per_layer"].append(entry)
        print(
            f"layer {layer:>4}  AUC {auc:.3f}  AP {ap:.3f}  "
            f"cos {entry['token_uniformity']:.3f}  fisher {entry['fisher_ratio']:.3f}",
            flush=True,
        )

    auc_values = np.array([entry["auc"] for entry in metrics["per_layer"]])
    best_layer = int(layers[int(auc_values.argmax())])
    metrics["best_layer"] = best_layer
    show_layers = args.show_layers or sorted(
        {best_layer, int(layers[-1]), int(layers[len(layers) // 2]), int(layers[0])}
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "separability.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    # ---- Panel 1 + 2: layer-wise curves -------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].plot(layers, auc_values, "o-", color="#2b6cb0", label="patch features")
    axes[0].axhline(position_auc, ls="--", color="#dd6b20",
                    label=f"position only ({position_auc:.3f})")
    axes[0].axhline(0.5, ls=":", color="#718096", label="chance")
    axes[0].axvline(best_layer, ls="-", lw=1, color="#c53030", alpha=0.5)
    axes[0].set_xlabel("block index (negative = from the end)")
    axes[0].set_ylabel("held-out ROC AUC")
    axes[0].set_title(
        f"{args.positive_name} vs {args.negative_name} separability "
        f"(best: {best_layer})"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    uniformity = [entry["token_uniformity"] for entry in metrics["per_layer"]]
    fisher = [entry["fisher_ratio"] for entry in metrics["per_layer"]]
    axes[1].plot(layers, uniformity, "s-", color="#805ad5", label="mean patch cosine")
    axes[1].set_xlabel("block index")
    axes[1].set_ylabel("mean pairwise cosine", color="#805ad5")
    axes[1].set_title("token uniformity vs class separation")
    axes[1].grid(alpha=0.3)
    twin = axes[1].twinx()
    twin.plot(layers, fisher, "^-", color="#38a169", label="Fisher ratio")
    twin.set_ylabel("Fisher ratio", color="#38a169")
    figure.tight_layout()
    figure.savefig(os.path.join(args.out_dir, "1_layer_curves.png"), dpi=150)

    # ---- Panel 3: score histograms ------------------------------------------
    figure, axes = plt.subplots(
        1, len(show_layers), figsize=(3.6 * len(show_layers), 3.2), squeeze=False
    )
    for axis, layer in zip(axes[0], show_layers):
        scores = test_scores_by_layer[layer]
        auc = metrics["per_layer"][layers.tolist().index(layer)]["auc"]
        bins = np.linspace(scores.min(), scores.max(), 60)
        axis.hist(scores[test_y == 0], bins=bins, alpha=0.6,
                  color="#718096", label=args.negative_name, density=True)
        axis.hist(scores[test_y == 1], bins=bins, alpha=0.6,
                  color="#c53030", label=args.positive_name, density=True)
        axis.set_title(f"block {layer}   AUC {auc:.3f}")
        axis.set_xlabel("probe score")
        axis.legend(fontsize=8)
    figure.suptitle("two lobes = separable; one lump = not", fontsize=10)
    figure.tight_layout()
    figure.savefig(os.path.join(args.out_dir, "2_score_histograms.png"), dpi=150)

    # ---- Panel 4: probe score painted back onto frames ----------------------
    test_frame_index = np.flatnonzero(is_test_frame)[: args.num_overlay_frames]
    figure, axes = plt.subplots(
        len(test_frame_index), len(show_layers) + 1,
        figsize=(2.6 * (len(show_layers) + 1), 2.6 * len(test_frame_index)),
        squeeze=False,
    )
    scores_dense = {}
    for layer in show_layers:
        dense = np.full((frame_count, patch_count), np.nan, dtype=np.float32)
        dense[test_mask] = test_scores_by_layer[layer]
        scores_dense[layer] = dense
    for row, frame_index in enumerate(test_frame_index):
        axes[row][0].imshow(frames[frame_index])
        axes[row][0].contour(
            np.kron(
                (labels_all[frame_index] == 1).reshape(grid_h, grid_w),
                np.ones((PATCH_SIZE, PATCH_SIZE)),
            ),
            levels=[0.5], colors="#f6e05e", linewidths=1.5,
        )
        axes[row][0].set_ylabel(f"frame {frame_index}", fontsize=8)
        if row == 0:
            axes[row][0].set_title("frame + label", fontsize=9)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        for column, layer in enumerate(show_layers, start=1):
            heat = scores_dense[layer][frame_index].reshape(grid_h, grid_w)
            axes[row][column].imshow(frames[frame_index])
            axes[row][column].imshow(
                np.kron(heat, np.ones((PATCH_SIZE, PATCH_SIZE))),
                cmap="RdBu_r", alpha=0.6,
            )
            if row == 0:
                axes[row][column].set_title(f"block {layer}", fontsize=9)
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
    figure.suptitle("probe score overlay: red = predicted hand/face", fontsize=10)
    figure.tight_layout()
    figure.savefig(os.path.join(args.out_dir, "3_score_overlay.png"), dpi=150)

    # ---- Panel 5: which signal should drive the selector? -------------------
    # These three need no training, so they are scored directly.  The probe
    # curve is the supervised ceiling for a linear ranking of the same features.
    candidates = {"linear probe (trained)": auc_values}
    if cls_attention is not None:
        candidates["CLS attention"] = np.array(
            [roc_auc_score(test_y, cls_attention[:, i][test_mask])
             for i in range(layer_count)]
        )
    if motion is not None:
        candidates["frame-to-frame motion"] = np.array(
            [roc_auc_score(test_y, motion[:, i][test_mask])
             for i in range(layer_count)]
        )
    norm_auc = []
    for layer_index in range(layer_count):
        features = features_all[:, layer_index].astype(np.float32)
        deviation = np.linalg.norm(
            features - features.mean(axis=1, keepdims=True), axis=-1
        )
        norm_auc.append(roc_auc_score(test_y, deviation[test_mask]))
    candidates["feature-norm deviation"] = np.array(norm_auc)

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    styles = {"linear probe (trained)": ("o-", "#2b6cb0"),
              "CLS attention": ("s-", "#c53030"),
              "frame-to-frame motion": ("^-", "#38a169"),
              "feature-norm deviation": ("d-", "#805ad5")}
    for name, values in candidates.items():
        marker, color = styles[name]
        axis.plot(layers, values, marker, color=color, label=name, alpha=0.85)
    axis.axhline(position_auc, ls="--", color="#dd6b20",
                 label=f"position only ({position_auc:.3f})")
    axis.axhline(0.5, ls=":", color="#718096")
    axis.set_xlabel("block index")
    axis.set_ylabel("held-out ROC AUC")
    axis.set_title(f"which signal ranks {args.positive_name} patches best?")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(args.out_dir, "4_scorer_comparison.png"), dpi=150)

    metrics["scorers"] = {name: values.tolist() for name, values in candidates.items()}
    with open(os.path.join(args.out_dir, "separability.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nbest layer {best_layer} (AUC {auc_values.max():.3f}) "
          f"vs position floor {position_auc:.3f}")
    for name, values in candidates.items():
        best = int(layers[int(values.argmax())])
        print(f"  {name:<24} best AUC {values.max():.3f} at block {best}")
    print(f"wrote figures and separability.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
