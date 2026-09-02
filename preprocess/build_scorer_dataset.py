"""Carve a labelled frame dataset for the hand-patch scorer out of PHOENIX14T *train*.

Usage::

    # print the per-signer sampling plan without writing anything
    python preprocess/build_scorer_dataset.py --dry-run

    # build it: ~13.7k frames, a few minutes across --workers processes
    python preprocess/build_scorer_dataset.py --out dataset/ph14_train_scorer_dataset

    # a bigger fitting set: raise the per-signer frame budget
    python preprocess/build_scorer_dataset.py --frames-per-signer 2500

Requires ``.cache/mediapipe/hand_landmarker.task``; the script prints the exact
curl command if it is missing.

The scorer is a ``Linear(1152 -> 1)`` fitted offline on hand/background patch
labels and then frozen into the visual adapter.  Because it ships inside the
model, its own fitting and evaluation data must come from the train split only:
touching dev or test would leak into the translation numbers those splits exist
to report.  So this script never reads them, and holds its evaluation videos out
of ``train`` instead.

Three sampling decisions, all of them deliberate:

* **Stratified by signer, on a frame budget rather than a video budget.**
  The train split is badly skewed -- Signer01 has 1862 videos, Signer06 has 35 --
  so a uniform video sample would leave the rare signers with almost no patches.
  Each signer gets the same *frame* budget instead, and the rare ones make it up
  by contributing more frames per video.
* **Frames per video is capped.** Adjacent frames of one video are near
  duplicates, so past roughly 25 frames a video adds samples but almost no
  information.  A signer that cannot fill its budget under that cap simply comes
  in short; padding it with redundant frames would only look like coverage.
* **Evenly spaced frames, no RNG.** Frame choice is a deterministic linspace over
  the video, with the first and last frames dropped -- the video processor pads
  the time axis by repeating the final frame, and boundary frames are the ones
  most often half-way into a transition.

Splits group on video: every frame of a video lands wholly in train or wholly in
test, so no evaluation frame shares a video (or a background, or a sign) with a
fitting frame.

**Frames are stored as the 224x224 centre crop, not the 256x256 source.** That
crop is exactly what the model's own predict transform feeds the backbone
(``CenterCrop(224)`` with ``do_resize=false``; note that the config's
``do_crop`` flag is not consulted on that path, so the crop happens whatever it
says).  Landmarks are detected on the same crop for the same reason: coordinates
taken on the uncropped frame would put every patch-grid mapping off by the crop
offset, silently.

Landmark coordinates are MediaPipe's normalized [0, 1], so the upscale used to
help the detector see a small image does not enter the coordinate space.

Output::

    ph14_train_scorer_dataset/
        dataset_info.json                 landmark names, conventions, provenance
        train.parquet                     one row per extracted frame
        test.parquet
        train/<Signer>/<video_id>/imagesNNNN.png     224x224
        test/<Signer>/<video_id>/imagesNNNN.png
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SPLIT = "train"
SOURCE_RESOLUTION = "fullFrame-210x260px"
TARGET_RESOLUTION = "fullFrame-256x256px"
HAND_SLOTS = 2

# MediaPipe's joint order is a fixed public spec, but this version of the
# package no longer exposes the enum (``mediapipe.python`` is gone), so the
# names are written out once here. The index is the MediaPipe landmark id.
HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# One landmarker per worker process: the object is not picklable, so it is built
# in the pool initializer and reached through this module global.
_LANDMARKER = None


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def read_corpus(data_root: Path, split: str) -> list[dict]:
    """Read one split's annotation csv.

    The csv is pipe separated with columns
    ``name|video|start|end|speaker|orth|translation``.  Its ``video`` column
    still carries the PHOENIX-2014 ``<name>/1/*.png`` layout, which this release
    does not use on disk, so frame paths are listed from the directory instead.
    """
    path = (
        data_root
        / "PHOENIX-2014-T/annotations/manual"
        / f"PHOENIX-2014-T.{split}.corpus.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"annotation csv not found: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    if not rows:
        raise RuntimeError(f"annotation csv is empty: {path}")
    return rows


def source_frames(data_root: Path, split: str, video_id: str) -> list[Path]:
    """Every original-resolution frame of one video, in temporal order."""
    directory = (
        data_root / "PHOENIX-2014-T/features" / SOURCE_RESOLUTION / split / video_id
    )
    if not directory.is_dir():
        raise FileNotFoundError(f"frame directory not found: {directory}")
    frames = sorted(directory.glob("*.png"))
    if not frames:
        raise RuntimeError(f"no frames in {directory}")
    return frames


def to_target_resolution(path: Path) -> Path:
    """Rewrite a frame path onto the pre-resized 256x256 variant.

    A missing file means the variant was never generated for this video and is
    raised rather than skipped: a silently short dataset is far more expensive
    to debug later.
    """
    parts = [
        TARGET_RESOLUTION if part == SOURCE_RESOLUTION else part for part in path.parts
    ]
    target = Path(*parts)
    if target == path:
        raise ValueError(f"path does not contain {SOURCE_RESOLUTION}: {path}")
    if not target.exists():
        raise FileNotFoundError(
            f"{TARGET_RESOLUTION} frame missing for {path}; expected {target}"
        )
    return target


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def frames_per_video(budget: int, video_count: int, minimum: int, maximum: int) -> int:
    """Frames to take from each video so a signer roughly meets its budget."""
    return int(np.clip(round(budget / max(video_count, 1)), minimum, maximum))


def pick_frame_indices(total: int, count: int, margin: int) -> list[int]:
    """Evenly spaced frame indices, dropping ``margin`` frames at each end."""
    usable = list(range(margin, total - margin)) or list(range(total))
    if len(usable) <= count:
        return usable
    # linspace rather than a fixed stride so the picks always span the whole
    # usable range regardless of how the two divide.
    positions = np.linspace(0, len(usable) - 1, count).round().astype(int)
    return [usable[position] for position in sorted(dict.fromkeys(positions))]


def plan_signer(
    videos: list[str], args, rng: np.random.Generator
) -> tuple[list[str], list[str], int]:
    """Choose one signer's videos and split them into train and test."""
    quota = min(len(videos), args.max_videos_per_signer)
    # Sorted first so the permutation depends only on the seed, never on csv or
    # filesystem ordering.
    chosen = rng.permutation(sorted(videos))[:quota].tolist()
    test_count = min(max(1, round(quota * args.test_fraction)), max(quota - 1, 0))
    per_video = frames_per_video(
        args.frames_per_signer,
        quota,
        args.min_frames_per_video,
        args.max_frames_per_video,
    )
    return sorted(chosen[test_count:]), sorted(chosen[:test_count]), per_video


# --------------------------------------------------------------------------- #
# Frames and landmarks
# --------------------------------------------------------------------------- #
def centre_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Match torchvision ``v2.CenterCrop``: round the offset, do not resize."""
    height, width = image.shape[:2]
    if height < size or width < size:
        raise ValueError(f"cannot crop {size}x{size} out of a {width}x{height} frame")
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return image[top : top + size, left : left + size]


def build_hand_landmarker(model_dir: Path):
    """Create the MediaPipe hand landmarker from a local asset.

    MediaPipe Tasks never downloads: ``model_asset_path`` is a local file, so a
    missing asset is reported here with what to fetch rather than as a C++
    initialization failure.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    asset = Path(model_dir) / "hand_landmarker.task"
    if not asset.exists():
        raise FileNotFoundError(
            f"MediaPipe asset not found: {asset}\n"
            f"Download it once with:\n"
            f"  mkdir -p {model_dir} && curl -L -o {asset} {HAND_LANDMARKER_URL}"
        )
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(asset)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=HAND_SLOTS,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
    )


def detect_hands(landmarker, image_rgb: np.ndarray, detect_size: int) -> dict:
    """Detect up to ``HAND_SLOTS`` hands, largest first, in normalized coords.

    The crop is upscaled before detection because MediaPipe finds far fewer
    hands in a 224px image, but the returned coordinates are normalized, so the
    upscale never enters the coordinate space.
    """
    import mediapipe as mp

    enlarged = cv2.resize(
        image_rgb, (detect_size, detect_size), interpolation=cv2.INTER_CUBIC
    )
    result = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(enlarged))
    )

    hands = []
    for index, landmarks in enumerate(result.hand_landmarks[:HAND_SLOTS]):
        x = np.array([point.x for point in landmarks], dtype=np.float32)
        y = np.array([point.y for point in landmarks], dtype=np.float32)
        handedness = None
        if index < len(result.handedness) and result.handedness[index]:
            handedness = result.handedness[index][0].category_name
        # Bounding-box area: the larger hand covers more pixels, so its joints
        # are the less noisy of the two.
        area = float((x.max() - x.min()) * (y.max() - y.min()))
        hands.append((area, x, y, handedness))
    hands.sort(key=lambda hand: hand[0], reverse=True)

    slots_x = np.full((HAND_SLOTS, len(HAND_LANDMARK_NAMES)), np.nan, dtype=np.float32)
    slots_y = np.full_like(slots_x, np.nan)
    slots_handedness: list[str | None] = [None] * HAND_SLOTS
    for slot, (_, x, y, handedness) in enumerate(hands):
        slots_x[slot], slots_y[slot] = x, y
        slots_handedness[slot] = handedness
    return {
        "num_hands_detected": len(hands),
        "hand_x": slots_x.reshape(-1).tolist(),
        "hand_y": slots_y.reshape(-1).tolist(),
        "handedness": slots_handedness,
    }


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _init_worker(mediapipe_dir: str) -> None:
    global _LANDMARKER
    _LANDMARKER = build_hand_landmarker(Path(mediapipe_dir))


def extract_video(task: tuple) -> list[dict]:
    """Crop, save and label one video's selected frames."""
    row, split_name, per_video, args = task
    video_id = row["name"]
    frames = source_frames(args.data_root, SPLIT, video_id)
    indices = pick_frame_indices(len(frames), per_video, args.margin)
    destination = args.out / split_name / row["speaker"] / video_id
    destination.mkdir(parents=True, exist_ok=True)

    records = []
    for frame_index in indices:
        source = to_target_resolution(frames[frame_index])
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"failed to read {source}")
        cropped = centre_crop(image, args.crop_size)
        target = destination / source.name
        if not cv2.imwrite(str(target), cropped):
            raise RuntimeError(f"failed to write {target}")

        record = {
            "id": f"{video_id}#{frame_index:05d}",
            "split": split_name,
            "signer": row["speaker"],
            "video_id": video_id,
            "frame_index": frame_index,
            "video_num_frames": len(frames),
            # Where in the utterance this frame sits; signing at the very start
            # and end of a clip is often a rest pose.
            "relative_position": frame_index / max(len(frames) - 1, 1),
            "path": str(target.relative_to(args.out)),
            "source_path": str(source.relative_to(args.data_root)),
            "orth": row["orth"],
            "translation": row["translation"],
        }
        record.update(
            detect_hands(
                _LANDMARKER, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), args.detect_size
            )
        )
        records.append(record)
    return records


def write_dataset_info(args, plan: dict, counts: dict) -> None:
    """One place recording what the columns mean and how the set was drawn."""
    import mediapipe

    info = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hand_landmark_names": list(HAND_LANDMARK_NAMES),
        "coordinate_space": (
            f"MediaPipe normalized [0, 1] relative to the "
            f"{args.crop_size}x{args.crop_size} frame stored at `path`; multiply "
            f"by {args.crop_size} for pixels"
        ),
        "hand_slots": (
            f"hand_x / hand_y hold {HAND_SLOTS} slots x "
            f"{len(HAND_LANDMARK_NAMES)} joints, flattened slot-major; slots are "
            "ordered by bounding-box area descending and an absent hand is NaN"
        ),
        "handedness": (
            "as reported by MediaPipe, which labels from the image's own "
            "perspective; not verified against the signer's dominant hand"
        ),
        "detector": {
            "model": "hand_landmarker.task",
            "detect_size": args.detect_size,
            "min_hand_detection_confidence": 0.3,
            "mediapipe_version": mediapipe.__version__,
        },
        "source": {
            "data_root": str(args.data_root),
            "split": SPLIT,
            "resolution_variant": TARGET_RESOLUTION,
            "transform": (
                f"centre crop to {args.crop_size}x{args.crop_size}, no resize -- "
                "matches SignVideoProcessor.build_predict_transform"
            ),
        },
        "sampling": {
            "seed": args.seed,
            "frames_per_signer": args.frames_per_signer,
            "max_videos_per_signer": args.max_videos_per_signer,
            "min_frames_per_video": args.min_frames_per_video,
            "max_frames_per_video": args.max_frames_per_video,
            "margin": args.margin,
            "test_fraction": args.test_fraction,
        },
        "per_signer": {
            signer: {
                "train_videos": len(train_videos),
                "test_videos": len(test_videos),
                "frames_per_video": per_video,
            }
            for signer, (train_videos, test_videos, per_video) in plan.items()
        },
        "counts": counts,
    }
    path = args.out / "dataset_info.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("dataset/PHOENIX-2014-T-release-v3")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("dataset/ph14_train_scorer_dataset")
    )
    parser.add_argument(
        "--frames-per-signer",
        type=int,
        default=1700,
        help="frame budget each signer gets, before the per-video cap",
    )
    parser.add_argument("--max-videos-per-signer", type=int, default=200)
    parser.add_argument("--min-frames-per-video", type=int, default=8)
    parser.add_argument("--max-frames-per-video", type=int, default=25)
    parser.add_argument(
        "--margin", type=int, default=2, help="frames dropped at each end of a video"
    )
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="centre crop the backbone actually sees",
    )
    parser.add_argument(
        "--detect-size",
        type=int,
        default=512,
        help="upscale used for detection only; coordinates stay normalized",
    )
    parser.add_argument("--mediapipe-dir", type=Path, default=Path(".cache/mediapipe"))
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() // 2, 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sampling plan without writing anything",
    )
    args = parser.parse_args()

    rows = read_corpus(args.data_root, SPLIT)
    by_signer = defaultdict(list)
    row_by_video = {}
    for row in rows:
        by_signer[row["speaker"]].append(row["name"])
        row_by_video[row["name"]] = row

    rng = np.random.default_rng(args.seed)
    plan = {
        signer: plan_signer(videos, args, rng)
        for signer, videos in sorted(by_signer.items())
    }

    print(f"{'signer':<10}{'avail':>7}{'train':>7}{'test':>6}{'f/vid':>7}{'frames':>8}")
    print("-" * 45)
    for signer, (train_videos, test_videos, per_video) in plan.items():
        total = (len(train_videos) + len(test_videos)) * per_video
        print(
            f"{signer:<10}{len(by_signer[signer]):>7}{len(train_videos):>7}"
            f"{len(test_videos):>6}{per_video:>7}{total:>8}"
        )
    grand_total = sum(
        (len(train) + len(test)) * per_video for train, test, per_video in plan.values()
    )
    print("-" * 45)
    print(f"{'total':<10}{len(rows):>7}{'':>7}{'':>6}{'':>7}{grand_total:>8}")
    if args.dry_run:
        print("\ndry run: nothing written")
        return

    # Fail before any work if the asset is missing, rather than in every worker.
    build_hand_landmarker(args.mediapipe_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [
        (row_by_video[video_id], split_name, per_video, args)
        for train_videos, test_videos, per_video in plan.values()
        for split_name, videos in (("train", train_videos), ("test", test_videos))
        for video_id in videos
    ]

    records = defaultdict(list)
    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(args.mediapipe_dir),),
    ) as pool:
        for video_records in tqdm(
            pool.imap_unordered(extract_video, tasks), total=len(tasks), desc="frames"
        ):
            records[video_records[0]["split"]].extend(video_records)

    counts = {}
    for split_name in ("train", "test"):
        frame = pd.DataFrame(records[split_name]).sort_values(
            ["video_id", "frame_index"]
        )
        path = args.out / f"{split_name}.parquet"
        frame.to_parquet(path, index=False)
        detected = float((frame["num_hands_detected"] > 0).mean())
        counts[split_name] = {
            "frames": len(frame),
            "videos": int(frame["video_id"].nunique()),
            "signers": int(frame["signer"].nunique()),
            "frames_with_a_hand": float(round(detected, 4)),
            "frames_with_two_hands": float(
                round((frame["num_hands_detected"] == 2).mean(), 4)
            ),
        }
        print(
            f"wrote {path}: {len(frame)} frames, {counts[split_name]['videos']} videos, "
            f"{detected:.1%} with a hand"
        )
    write_dataset_info(args, plan, counts)


if __name__ == "__main__":
    main()
