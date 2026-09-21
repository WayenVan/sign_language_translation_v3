"""Carve a labelled frame dataset for the hand-patch scorer out of CSL-Daily *train*.

Usage::

    # print the per-signer sampling plan without writing anything
    python preprocess/csl_daily/build_scorer_dataset.py --dry-run

    # build it: ~16k frames, a few minutes across --workers processes
    python preprocess/csl_daily/build_scorer_dataset.py

    # a bigger fitting set: raise the per-signer frame budget
    python preprocess/csl_daily/build_scorer_dataset.py --frames-per-signer 2500

Reads the frames written by ``preprocess/csl_daily/extract_frames.py`` (by
default ``preprocessed/full-trim-256x256px``, already cut to the signing span)
and requires ``.cache/mediapipe/hand_landmarker.task``; the script prints the
exact curl command if it is missing.

The scorer is a ``Linear(1152 -> 1)`` fitted offline on hand/background patch
labels and then frozen into the visual adapter.  Because it ships inside the
model, its own fitting and evaluation data must come from the train split only:
touching dev or test would leak into the translation numbers those splits exist
to report.  So this script never reads them, and holds its evaluation videos out
of ``train`` instead.

Sampling follows ``preprocess/ph14t/build_scorer_dataset.py``:

* **Stratified by signer, on a frame budget rather than a video budget.**
  Train is skewed -- P0000 has 6593 clips, P0001 has 664 -- so each signer gets
  the same *frame* budget instead of the same share of videos.
* **Frames per video is capped.** Adjacent frames of one video are near
  duplicates, so past roughly 25 frames a video adds samples but almost no
  information.
* **Evenly spaced frames, no RNG.** Frame choice is a deterministic linspace over
  the clip with ``--margin`` frames dropped at each end.  The clips are already
  trimmed to the signing span, so the margin only guards against the first and
  last frames being half-way into a transition.

Splits group on video: every frame of a video lands wholly in train or wholly in
test, so no evaluation frame shares a video (or a sign) with a fitting frame.

**Frames are stored as the crop the backbone sees at evaluation, not the
256x256 source.**  CSL-Daily is evaluated with ``eval_crop_bottom_aligned``,
so by default the crop is ``BottomCenterCrop(224)`` -- the processor's own
transform, imported rather than re-implemented, so the offset cannot drift.
``--crop-anchor center`` stores ``v2.CenterCrop(224)`` instead.  Landmarks are
detected on the same crop: coordinates taken on the uncropped frame would put
every patch-grid mapping off by the crop offset, silently.  Crops are stored as
PNG so the JPEG source is not compressed a second time.

Landmark coordinates are MediaPipe's normalized [0, 1], so the upscale used to
help the detector see a small image does not enter the coordinate space.

Output::

    preprocessed/scorer_dataset/
        dataset_info.json                 landmark names, conventions, provenance
        train.parquet                     one row per extracted frame
        test.parquet
        train/<signer>/<clip_id>/imagesNNNN.png     224x224
        test/<signer>/<clip_id>/imagesNNNN.png

``imagesNNNN`` keeps the source frame number, and each row records both
``frame_index`` (position within the trimmed clip) and ``source_frame_index``
(frame number in the original video).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from csi_slt.data.processors.video_transforms import BottomCenterCrop  # noqa: E402

SPLIT = "train"
HAND_SLOTS = 2
FRAME_NAME = re.compile(r"images(\d+)\.jpg$")

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
def read_source(frames_root: Path) -> tuple[pd.DataFrame, dict]:
    """The train index of a finished extract_frames.py run, and its meta.json.

    An unfinished run is refused rather than sampled: a clip missing from the
    index would silently shift which videos each signer contributes.
    """
    if not (frames_root / ".complete").exists():
        raise FileNotFoundError(
            f"{frames_root} has no .complete marker; finish "
            "preprocess/csl_daily/extract_frames.py first"
        )
    meta = json.loads((frames_root / "meta.json").read_text())
    index = pd.read_parquet(frames_root / f"{SPLIT}.parquet")
    if index.empty:
        raise RuntimeError(f"{frames_root / SPLIT}.parquet is empty")
    return index, meta


def source_frame_index(path: str) -> int:
    """images0020.jpg -> source frame 19."""
    match = FRAME_NAME.search(path)
    if match is None:
        raise ValueError(f"unexpected frame file name: {path}")
    return int(match.group(1)) - 1


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
    # Sorted first so the permutation depends only on the seed, never on
    # parquet ordering.
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
def build_crop(anchor: str, size: int):
    """The processor's eval crop, so stored frames match what the backbone sees."""
    if anchor == "bottom":
        return BottomCenterCrop((size, size))
    return v2.CenterCrop((size, size))


def apply_crop(crop, image: np.ndarray, size: int) -> np.ndarray:
    """HWC in, HWC out; refuses to crop more than the frame holds.

    v2.CenterCrop would pad a small frame instead, which here would only mean
    the source was not the resolution this dataset assumes.
    """
    height, width = image.shape[:2]
    if height < size or width < size:
        raise ValueError(f"cannot crop {size}x{size} out of a {width}x{height} frame")
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return np.ascontiguousarray(crop(tensor).permute(1, 2, 0).numpy())


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
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _LANDMARKER = build_hand_landmarker(Path(mediapipe_dir))


def extract_video(task: tuple) -> list[dict]:
    """Crop, save and label one video's selected frames."""
    row, split_name, per_video, args = task
    clip_id = row["clip_id"]
    frames = list(row["frames"])
    indices = pick_frame_indices(len(frames), per_video, args.margin)
    destination = args.out / split_name / row["signer"] / clip_id
    destination.mkdir(parents=True, exist_ok=True)
    crop = build_crop(args.crop_anchor, args.crop_size)

    records = []
    for frame_index in indices:
        source = args.frames_root / frames[frame_index]
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"failed to read {source}")
        cropped = apply_crop(crop, image, args.crop_size)
        source_index = source_frame_index(frames[frame_index])
        target = destination / f"images{source_index + 1:04d}.png"
        if not cv2.imwrite(str(target), cropped):
            raise RuntimeError(f"failed to write {target}")

        record = {
            "id": f"{clip_id}#{source_index:05d}",
            "split": split_name,
            "signer": row["signer"],
            "video_id": clip_id,
            "frame_index": frame_index,
            "source_frame_index": source_index,
            "video_num_frames": len(frames),
            # Where in the (trimmed) utterance this frame sits.
            "relative_position": frame_index / max(len(frames) - 1, 1),
            "path": str(target.relative_to(args.out)),
            "source_path": frames[frame_index],
            "gloss": row["gloss"],
            "translation": row["translation"],
        }
        record.update(
            detect_hands(
                _LANDMARKER, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), args.detect_size
            )
        )
        records.append(record)
    return records


def write_dataset_info(args, source_meta: dict, plan: dict, counts: dict) -> None:
    """One place recording what the columns mean and how the set was drawn."""
    import mediapipe

    crop_name = "BottomCenterCrop" if args.crop_anchor == "bottom" else "v2.CenterCrop"
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
            "frames_root": str(args.frames_root),
            "split": SPLIT,
            "extract_frames_params": source_meta.get("params"),
            "crop_anchor": args.crop_anchor,
            "transform": (
                f"{crop_name}({args.crop_size}) of the stored frame, no resize -- "
                "matches SignVideoProcessor.build_predict_transform with "
                f"eval_crop_bottom_aligned={args.crop_anchor == 'bottom'}"
            ),
            "stored_format": "png",
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("dataset/CSL-Daily-HG/preprocessed/full-trim-256x256px"),
        help="output directory of preprocess/csl_daily/extract_frames.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/CSL-Daily-HG/preprocessed/scorer_dataset"),
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
        help="eval crop the backbone actually sees",
    )
    parser.add_argument(
        "--crop-anchor",
        choices=("bottom", "center"),
        default="bottom",
        help="bottom matches eval_crop_bottom_aligned=true (default: bottom)",
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

    index, source_meta = read_source(args.frames_root)
    by_signer = defaultdict(list)
    row_by_video = {}
    for row in index[["clip_id", "signer", "gloss", "translation", "frames"]].to_dict("records"):
        by_signer[row["signer"]].append(row["clip_id"])
        row_by_video[row["clip_id"]] = row

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
    print(f"{'total':<10}{len(index):>7}{'':>7}{'':>6}{'':>7}{grand_total:>8}")
    print("(frames is the upper bound; clips shorter than f/vid + 2*margin give fewer)")
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
    write_dataset_info(args, source_meta, plan, counts)


if __name__ == "__main__":
    main()
