"""Decode How2Sign front-view clips into square, resized JPEG frame sequences.

Usage::

    # smoke test: the first 20 clips of every split
    python preprocess/how2sign/extract_frames.py --limit 20

    # full run; re-running resumes and skips clips that are already done
    python preprocess/how2sign/extract_frames.py --workers 32

Output, under ``<dataset-root>/preprocessed/sq720-x+50-24fps-256x256px/`` (the
name follows ``--shift``, ``--fps`` and ``--size``)::

    meta.json                        parameters, git revision, per-split stats
    train.parquet validation.parquet test.parquet
    .complete                        written only after a full, error-free run
    <split>/<clip_id>/images0001.jpg ...

Each parquet row carries ``clip_id, signer, video_id, video_name, translation,
source_fps, speed_fix, width, height, video_num_frames, num_frames,
source_frames, frames``; ``frames`` is the ordered list of that clip's frame paths, relative
to the output directory, so the whole directory can be copied elsewhere and
read through a new data root unchanged.  ``clip_id`` is the dataset's
``SENTENCE_NAME`` and ``translation`` its ``SENTENCE``.

Frames
    The MP4s are already cut to sentence level and carry no signing span, so
    the whole clip is kept.  The source frame rate is not uniform (mostly
    24 fps, some 23.98, 30 and 60), so every clip is resampled to ``--fps``:
    output frame k is the source frame nearest to time k / fps, i.e. source
    index ``floor(k * source_fps / fps + 0.5)``.  Clips within
    ``FPS_TOLERANCE`` of the target (24 and 23.98) keep every frame, and a
    clip is never upsampled.  Output frames are numbered consecutively
    (``images0001.jpg`` is output frame 0); ``source_frames`` maps each one
    back to its source frame index.  ``video_num_frames`` is the container's
    frame count, which can disagree with the decoded count by a frame or two.

Playback speed
    Every clip of signer 5 runs 4/3 as long as its START/END span in the
    metadata (median ratio 1.333, 5th-95th percentile 1.32-1.34, at every
    source frame rate; 1.000 for the other signers), and signs 2.40 words per
    second by its own clock against 3.1-3.5 for everyone else and 3.19 by the
    metadata clock.  The videos were slowed to 0.75x, so ``SPEED_FIX`` scales
    those clips' source rate by 4/3 before resampling (24 fps is read as 32,
    keeping 3 of every 4 frames).  The factor is stored per clip in
    ``speed_fix``.

Framing
    Every clip is 1280x720 with the signer seated on a green screen, a little
    right of centre.  A full-height 720x720 square is cut, shifted ``--shift``
    pixels right of the centred position (x 280-1000 -> 330-1050 by default),
    then resized to ``--size``.  From MediaPipe pose and hand landmarks on 4320
    frames (9 signers x 60 clips x 8 frames) a +50 shift balances the
    landmarks cut on either side, about 0.2% of frames each against 0.09% /
    0.72% for a centred crop; the heads never reach the top edge.

Signers
    The metadata has no signer column.  The number in ``VIDEO_NAME``
    (``<VIDEO_ID>-<n>-rgb_front``) is the signer: nine values, one person each.
"""

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


SPLITS = ("train", "validation", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SIZE = (1280, 720)
# Source rates this close to the target are treated as the target (23.98 -> 24).
FPS_TOLERANCE = 0.5
# Signer -> factor by which that signer's videos are slowed down; see
# "Playback speed" above.
SPEED_FIX = {5: 4 / 3}


def git_revision() -> str | None:
    """Best effort: the checkout may not be a git repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def frame_name(index: int) -> str:
    """Output frame index -> 1-based, PHOENIX-2014-T style name."""
    return f"images{index + 1:04d}.jpg"


def output_name(shift: int, fps: int, size: int) -> str:
    return f"sq720-x{shift:+d}-{fps}fps-{size}x{size}px"


def frame_step(source_fps: float, fps: int) -> float:
    """Source frames per output frame; 1 keeps every frame, never below 1."""
    if not source_fps > 0 or abs(source_fps - fps) <= FPS_TOLERANCE:
        return 1.0
    return max(source_fps / fps, 1.0)


def source_index(k: int, step: float) -> int:
    """Source frame index for output frame k: the one nearest to time k / fps."""
    return int(np.floor(k * step + 0.5))


def resampled_indices(num_source_frames: int, source_fps: float, fps: int) -> list[int]:
    """Source frame indices kept from a clip of num_source_frames frames."""
    step = frame_step(source_fps, fps)
    indices = []
    while (i := source_index(len(indices), step)) < num_source_frames:
        indices.append(i)
    return indices


def crop_frame(frame, shift, size):
    """Full-height square, shifted `shift` px right of centre, resized to size."""
    height, width = frame.shape[:2]
    side = min(height, width)
    left = min(max((width - side) // 2 + shift, 0), width - side)
    crop = frame[height - side : height, left : left + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def open_video(video_path: Path) -> cv2.VideoCapture:
    """Single-threaded decoder: cv2.setNumThreads does not reach the FFmpeg
    backend, which otherwise starts ~16 threads per capture and oversubscribes
    the CPUs once every worker has one open."""
    return cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])


def extract_clip(task):
    """Decode one video at `fps` into <out>/<split>/<clip_id>/."""
    split, clip_id, video_path, out_root, shift, speed, fps, size, quality = task
    cv2.setNumThreads(1)
    final_dir = out_root / split / clip_id
    result = dict(split=split, clip_id=clip_id, error=None)

    tmp_dir = final_dir.with_name(clip_id + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    capture = open_video(video_path)
    if not capture.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return dict(result, error=f"cannot open {video_path}")

    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    step = frame_step(capture.get(cv2.CAP_PROP_FPS) * speed, fps)
    count = 0
    try:
        # H.264 seeking is not frame-accurate, so walk the stream; grab()
        # skips the colour conversion for frames that are dropped.
        index = 0
        while capture.grab():
            if index == source_index(count, step):
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"failed to decode source frame {index}")
                image = crop_frame(frame, shift, size)
                if not cv2.imwrite(str(tmp_dir / frame_name(count)), image, params):
                    raise RuntimeError(f"failed to write frame {count}")
                count += 1
            index += 1
    except Exception as exc:  # noqa: BLE001 -- reported per clip, run continues
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return dict(result, error=f"{type(exc).__name__}: {exc}")
    finally:
        capture.release()

    if count == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return dict(result, error="decoded 0 frames")

    os.rename(tmp_dir, final_dir)
    return result


def probe(video_path: Path) -> dict:
    """Container properties; cheap, no decoding."""
    capture = open_video(video_path)
    try:
        return dict(
            source_fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            video_num_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def build_index(split, metadata, dataset_root, out_root, fps):
    """One row per clip that has a finished frame directory."""
    rows = []
    for record in metadata.itertuples(index=False):
        clip_dir = out_root / split / record.clip_id
        if not clip_dir.is_dir():
            continue
        # Zero-padded names sort in frame order.
        names = sorted(p.name for p in clip_dir.glob("images*.jpg"))
        source = probe(dataset_root / split / record.file_name)
        speed = SPEED_FIX.get(record.signer, 1.0)
        step = frame_step(source["source_fps"] * speed, fps)
        rows.append(
            dict(
                clip_id=record.clip_id,
                signer=record.signer,
                video_id=record.VIDEO_ID,
                video_name=record.VIDEO_NAME,
                translation=record.SENTENCE,
                **source,
                speed_fix=speed,
                num_frames=len(names),
                source_frames=[source_index(k, step) for k in range(len(names))],
                frames=[f"{split}/{record.clip_id}/{name}" for name in names],
            )
        )
    return pd.DataFrame(rows)


def load_metadata(dataset_root: Path, split: str) -> pd.DataFrame:
    metadata = pd.read_parquet(dataset_root / split / "metadata.parquet")
    metadata["clip_id"] = metadata.SENTENCE_NAME
    if not metadata.clip_id.is_unique:
        raise ValueError(f"{split}: SENTENCE_NAME is not unique")
    metadata["signer"] = metadata.VIDEO_NAME.str.extract(r"-(\d+)-rgb_front$", expand=False)
    if metadata.signer.isna().any():
        bad = metadata.VIDEO_NAME[metadata.signer.isna()].tolist()[:5]
        raise ValueError(f"cannot parse signer from video names such as {bad}")
    metadata["signer"] = metadata.signer.astype(int)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/how2sign-front-clips"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <dataset-root>/preprocessed/sq720-x<shift>-<fps>fps-<size>x<size>px",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=50,
        help="horizontal offset of the 720x720 crop from centre, px, positive = right (default: 50)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="output frame rate; faster sources are resampled, slower ones kept as is (default: 24)",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N clips of each split; the run is then never marked complete",
    )
    args = parser.parse_args()

    max_shift = (SOURCE_SIZE[0] - SOURCE_SIZE[1]) // 2
    if abs(args.shift) > max_shift:
        parser.error(f"--shift must be within +-{max_shift} for {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} video")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in [1, 100]")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    out_root = args.out or (args.dataset_root / "preprocessed" / output_name(args.shift, args.fps, args.size))
    params = dict(
        crop="full-height square",
        shift=args.shift,
        fps=args.fps,
        fps_tolerance=FPS_TOLERANCE,
        speed_fix={str(k): v for k, v in SPEED_FIX.items()},
        resample="nearest source frame",
        size=args.size,
        interpolation="INTER_AREA",
        format="jpeg",
        quality=args.quality,
    )

    # Resuming reuses finished clips, so they must have been made the same way.
    meta_path = out_root / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())["params"]
        if previous != params:
            parser.error(
                f"{out_root} was built with different parameters {previous}; "
                "choose another --out or delete it"
            )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ".complete").unlink(missing_ok=True)
    if not meta_path.exists():
        # Record the parameters before any clip lands, so a run that dies
        # midway still guards its partial output against a mismatched resume.
        meta_path.write_text(json.dumps(dict(params=params, complete=False), indent=2) + "\n")

    metadata = {}
    tasks = []
    for split in args.splits:
        frame = load_metadata(args.dataset_root, split)
        if args.limit is not None:
            frame = frame.head(args.limit)
        metadata[split] = frame
        (out_root / split).mkdir(exist_ok=True)
        for record in frame.itertuples(index=False):
            if (out_root / split / record.clip_id).is_dir():
                continue
            video = args.dataset_root / split / record.file_name
            speed = SPEED_FIX.get(record.signer, 1.0)
            tasks.append(
                (split, record.clip_id, video, out_root, args.shift, speed,
                 args.fps, args.size, args.quality)
            )

    total = sum(len(frame) for frame in metadata.values())
    print(f"{total} clips in {args.splits}; {total - len(tasks)} already done, {len(tasks)} to extract")
    print(f"output: {out_root}")

    results = []
    with Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(extract_clip, tasks), total=len(tasks)):
            results.append(result)

    errors = [r for r in results if r["error"]]
    for r in errors:
        print(f"ERROR {r['split']}/{r['clip_id']}: {r['error']}")

    stats = {}
    for split, frame in metadata.items():
        index = build_index(split, frame, args.dataset_root, out_root, args.fps)
        index.to_parquet(out_root / f"{split}.parquet", index=False)
        if len(index):
            # Container frame counts, resampled the same way, as a sanity check.
            expected = [
                len(resampled_indices(n, f * k, args.fps))
                for n, f, k in zip(index.video_num_frames, index.source_fps, index.speed_fix)
            ]
            diff = (index.num_frames - pd.Series(expected, index=index.index)).abs()
            off_size = index[(index.width != SOURCE_SIZE[0]) | (index.height != SOURCE_SIZE[1])]
            fps_counts = index.source_fps.round(2).value_counts().sort_index()
        else:
            diff, off_size, fps_counts = pd.Series(dtype=int), index, pd.Series(dtype=int)
        stats[split] = dict(
            clips=int(len(index)),
            clips_expected=int(len(frame)),
            frames=int(index.num_frames.sum()) if len(index) else 0,
            frame_count_mismatches=int((diff > 0).sum()),
            max_frame_count_diff=int(diff.max()) if len(diff) else 0,
            source_fps=[{"fps": float(k), "clips": int(v)} for k, v in fps_counts.items()],
            not_1280x720=off_size.clip_id.tolist()[:20],
        )
        print(f"{split}: {stats[split]}")

    complete = (
        args.limit is None
        and not errors
        and all(s["clips"] == s["clips_expected"] for s in stats.values())
        and set(args.splits) == set(SPLITS)
    )
    meta = dict(
        params=params,
        source=str(args.dataset_root),
        splits=stats,
        limit=args.limit,
        this_run=dict(extracted=len(results), errors=len(errors)),
        complete=complete,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        csi_slt_git_sha=git_revision(),
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    if complete:
        (out_root / ".complete").touch()
        print("complete")
    else:
        print("not marked complete (limited run, errors, or missing clips); re-run to resume")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
