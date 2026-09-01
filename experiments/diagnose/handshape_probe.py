"""Probe how much hand information survives the adapter's pooling.

The separability diagnosis left one question open: patch features clearly encode
*that* a patch is a hand, but do they encode *which* handshape -- and does any of
it reach the LLM after the adapter pools a frame into one token?

Targets come from MediaPipe's 21 hand joints, normalized three ways:

    position      the hand's image-space centroid.  A control: if this is not
                  near-perfect the pipeline is broken, not the features.
    shape+orient  joints translated to the wrist and divided by the wrist->MCP
                  distance.  Scale and location removed, orientation kept.
    shape only    the same, then rotated so wrist->MCP points a fixed way.
                  Pure finger configuration.

Every probe is ridge over a 48-component PCA, grouped 4-fold by video with an
inner grouped fold picking alpha.  Grouping matters: six frames of one video
share a signer, a background and often a sign, so a random split leaks.

Usage:
    python experiments/diagnose/handshape_probe.py                    # 224, 14x14
    python experiments/diagnose/handshape_probe.py \
        --features outputs/diagnose/features512.npz --grid 32         # 512, 32x32
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "experiments/diagnose"))


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def hand_landmarks(frames, model_dir, cache, detect_size=512):
    """MediaPipe's 21 joints per frame, cached: the detector is the slow part."""
    if os.path.exists(cache):
        return np.load(cache)["coords"]
    import cv2
    import mediapipe as mp

    from extract import build_landmarkers

    hand_landmarker, _ = build_landmarkers(model_dir)
    coords = np.full((len(frames), 2, 21, 3), np.nan, np.float32)
    for index, frame in enumerate(frames):
        enlarged = cv2.resize(
            frame, (detect_size, detect_size), interpolation=cv2.INTER_CUBIC
        )
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(enlarged)
        )
        for slot, landmarks in enumerate(
            hand_landmarker.detect(image).hand_landmarks[:2]
        ):
            coords[index, slot] = [[p.x, p.y, p.z] for p in landmarks]
        if (index + 1) % 60 == 0:
            print(f"  landmarked {index + 1}/{len(frames)}", flush=True)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez_compressed(cache, coords=coords)
    return coords


def build_targets(coords):
    """Return per-frame targets plus the mask of frames holding a hand."""
    count = len(coords)
    # The larger hand has more pixels behind it, so its joints are the less
    # noisy of the two. Signers use both, but only one is probed per frame.
    extent = np.nan_to_num(
        np.nanmax(coords[..., :2], 2) - np.nanmin(coords[..., :2], 2)
    ).prod(-1)
    landmarks = coords[np.arange(count), extent.argmax(1)][..., :2]
    usable = ~np.isnan(landmarks[:, 0, 0])

    wrist, mcp = landmarks[:, 0], landmarks[:, 9]
    scale = np.linalg.norm(mcp - wrist, axis=-1, keepdims=True)[:, None]
    centred = (landmarks - wrist[:, None]) / np.clip(scale, 1e-6, None)
    angle = np.arctan2((mcp - wrist)[:, 1], (mcp - wrist)[:, 0])
    cos, sin = np.cos(-angle), np.sin(-angle)
    rotation = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)
    targets = {
        "shape+orient": centred[:, 1:].reshape(count, -1),
        "shape only": np.einsum("nij,nkj->nki", rotation, centred[:, 1:]).reshape(
            count, -1
        ),
        "position": landmarks.mean(1),
    }
    return landmarks, usable, targets


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def probe(features, target, groups, n_components=48):
    """Grouped-CV R^2 of a ridge readout, alpha chosen on an inner grouped fold."""
    reduced = PCA(
        n_components=min(n_components, features.shape[0] - 1, features.shape[1])
    ).fit_transform(features)
    predictions = np.zeros_like(target)
    for train, test in GroupKFold(n_splits=4).split(reduced, target, groups):
        best_alpha, best_score = 1e2, -np.inf
        for alpha in (1e0, 1e1, 1e2, 1e3, 1e4):
            scores = [
                r2_score(
                    target[train][inner_test],
                    Ridge(alpha=alpha)
                    .fit(reduced[train][inner_train], target[train][inner_train])
                    .predict(reduced[train][inner_test]),
                    multioutput="variance_weighted",
                )
                for inner_train, inner_test in GroupKFold(n_splits=3).split(
                    reduced[train], target[train], groups[train]
                )
            ]
            if np.mean(scores) > best_score:
                best_alpha, best_score = alpha, np.mean(scores)
        predictions[test] = (
            Ridge(alpha=best_alpha)
            .fit(reduced[train], target[train])
            .predict(reduced[test])
        )
    return r2_score(target, predictions, multioutput="variance_weighted")


def hand_box_masks(landmarks, usable, side):
    """Patches overlapping each frame's hand bounding box."""
    patches = side * side
    rows, columns = np.arange(patches) // side, np.arange(patches) % side
    masks = np.ones((len(landmarks), patches), bool)
    for index in np.where(usable)[0]:
        low, high = np.nanmin(landmarks[index], 0), np.nanmax(landmarks[index], 0)
        mask = (
            ((columns + 1) / side > low[0])
            & (columns / side < high[0])
            & ((rows + 1) / side > low[1])
            & (rows / side < high[1])
        )
        if mask.any():
            masks[index] = mask
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/diagnose/features.npz")
    parser.add_argument("--grid", type=int, default=14, help="patch grid side")
    parser.add_argument("--landmarks", default="outputs/diagnose/hand_landmarks.npz")
    parser.add_argument(
        "--mediapipe-dir", default=str(PROJECT_ROOT / ".cache" / "mediapipe")
    )
    parser.add_argument("--motion-layer", type=int, default=-21)
    args = parser.parse_args()

    cached = np.load(args.features)
    features, video_ids = cached["features"], cached["video_ids"]
    layers = list(cached["layers"])
    coords = hand_landmarks(cached["frames"], args.mediapipe_dir, args.landmarks)
    landmarks, usable, targets = build_targets(coords)
    groups = video_ids[usable]
    print(
        f"{args.features}: {features.shape}, {usable.sum()} usable frames over "
        f"{len(np.unique(groups))} videos"
    )

    # Floor: handshape is partly predictable from *where* the hand is, because
    # signers hold particular shapes in particular places. Only what a probe
    # scores above this floor is actually read off the features.
    print("\nposition-only floor:")
    for name, target in targets.items():
        print(
            f"  {name:>14}: R2 = "
            f"{probe(targets['position'][usable], target[usable], groups, 2):.3f}"
        )

    masks = hand_box_masks(landmarks, usable, args.grid)
    print(
        f"\nlayer sweep, input = mean of the hand box "
        f"({masks[usable].sum(1).mean():.1f} of {args.grid**2} patches)"
    )
    print(f"{'layer':>6} {'shape+orient':>13} {'shape only':>11} {'position':>9}")
    for position, layer in enumerate(layers):
        patch_features = torch.from_numpy(features[:, position]).float()
        pooled = np.stack(
            [patch_features[i, masks[i]].mean(0).numpy() for i in range(len(masks))]
        )[usable]
        scores = [probe(pooled, target[usable], groups) for target in targets.values()]
        print(f"{layer:>6} {scores[0]:>13.3f} {scores[1]:>11.3f} {scores[2]:>9.3f}")

    report_pooling(cached, features, landmarks, usable, targets, groups, args)


def report_pooling(cached, features, landmarks, usable, targets, groups, args):
    """What survives each candidate adapter pooling at the last probed layer."""
    side = args.grid
    patches = side * side
    rows, columns = np.arange(patches) // side, np.arange(patches) % side
    x = torch.from_numpy(features[:, -1]).float()
    global_mean = x.mean(1).numpy()
    region = (rows * 3 // side) * 2 + (columns * 2 // side)

    candidates = {
        "1  hand box (hard crop)": np.stack(
            [
                x[i, m].mean(0).numpy()
                for i, m in enumerate(hand_box_masks(landmarks, usable, side))
            ]
        ),
        "1  global mean (current)": global_mean,
        "6  spatial grid 3x2": torch.stack(
            [x[:, torch.from_numpy(region == i)].mean(1) for i in range(6)], 1
        )
        .flatten(1)
        .numpy(),
    }
    # Soft, aimed pooling needs the extras only the 224 cache carries.
    if "motion" in cached and cached["motion"].shape[-1] == patches:
        motion = torch.from_numpy(
            cached["motion"][:, list(cached["layers"]).index(args.motion_layer)]
        ).float()
        weights = torch.softmax(motion / motion.std(1, keepdim=True) / 0.5, 1)
        candidates["2  global + motion-weighted"] = np.concatenate(
            [global_mean, torch.einsum("np,npd->nd", weights, x).numpy()], 1
        )

    print(f"\npooling at layer {list(cached['layers'])[-1]}:")
    print(
        f"{'tokens/frame':>28} {'shape+orient':>13} {'shape only':>11} {'position':>9}"
    )
    for name, pooled in candidates.items():
        scores = [
            probe(pooled[usable], target[usable], groups) for target in targets.values()
        ]
        print(f"{name:>28} {scores[0]:>13.3f} {scores[1]:>11.3f} {scores[2]:>9.3f}")


if __name__ == "__main__":
    main()
