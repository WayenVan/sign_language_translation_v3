"""Re-label cached frames for a harder probe task.

hand-vs-background saturates: in PHOENIX14T the hands and face are the only skin
against dark clothing and a grey backdrop, so skin colour alone separates them
and every block scores the same.  hand-vs-face keeps only skin patches, removing
that shortcut, and asks whether a layer encodes *which* body part a patch is.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from extract import PATCH_SIZE, build_landmarkers


def landmark_grid(landmark_sets, grid_h, grid_w):
    grid = np.zeros((grid_h, grid_w), dtype=bool)
    for landmark_set in landmark_sets:
        for landmark in landmark_set:
            row = int(np.clip(landmark.y, 0, 0.999) * grid_h)
            col = int(np.clip(landmark.x, 0, 0.999) * grid_w)
            grid[row, col] = True
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/diagnose/features.npz")
    parser.add_argument("--out", default="outputs/diagnose/labels_hand_vs_face.npz")
    parser.add_argument(
        "--mediapipe-dir",
        default=str(Path(__file__).resolve().parents[2] / ".cache" / "mediapipe"),
    )
    parser.add_argument("--detect-size", type=int, default=512)
    args = parser.parse_args()

    frames = np.load(args.features)["frames"]
    grid_h = frames.shape[1] // PATCH_SIZE
    grid_w = frames.shape[2] // PATCH_SIZE
    hand_landmarker, face_landmarker = build_landmarkers(args.mediapipe_dir)

    labels = np.full((len(frames), grid_h, grid_w), -1, dtype=np.int8)
    usable = 0
    for index, frame in enumerate(frames):
        enlarged = cv2.resize(
            frame, (args.detect_size, args.detect_size), interpolation=cv2.INTER_CUBIC
        )
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(enlarged)
        )
        hand = landmark_grid(
            hand_landmarker.detect(image).hand_landmarks, grid_h, grid_w
        )
        face = landmark_grid(
            face_landmarker.detect(image).face_landmarks, grid_h, grid_w
        )
        # A patch claimed by both is genuinely ambiguous (hand in front of the
        # face), so it is excluded rather than assigned.
        frame_labels = np.full((grid_h, grid_w), -1, dtype=np.int8)
        frame_labels[face & ~hand] = 0
        frame_labels[hand & ~face] = 1
        labels[index] = frame_labels
        if frame_labels.max() == 1:
            usable += 1
        if (index + 1) % 50 == 0:
            print(f"  relabelled {index + 1}/{len(frames)}", flush=True)

    np.savez_compressed(args.out, labels=labels.reshape(len(frames), -1))
    print(
        f"wrote {args.out}: {int((labels == 1).sum())} hand / "
        f"{int((labels == 0).sum())} face patches over {usable} usable frames"
    )


if __name__ == "__main__":
    main()
