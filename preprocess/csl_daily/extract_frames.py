"""Decode CSL-Daily videos into resized JPEG frame sequences.

Usage::

    # smoke test: the first 20 clips of every split
    python preprocess/csl_daily/extract_frames.py --limit 20

    # full run; re-running resumes and skips clips that are already done
    python preprocess/csl_daily/extract_frames.py --workers 32

    # keep the idle frames before and after signing as well
    python preprocess/csl_daily/extract_frames.py --no-trim

Output, under ``<dataset-root>/preprocessed/full-trim-256x256px/`` (the name
follows ``--ratio``, ``--trim`` and ``--size``)::

    meta.json                        parameters, git revision, per-split stats
    train.parquet dev.parquet test.parquet
    .complete                        written only after a full, error-free run
    <split>/<clip_id>/images0020.jpg ...

Each parquet row carries ``clip_id, signer, translation, gloss, start_frame,
end_frame_exclusive, num_frames, frames``; ``frames`` is the ordered list of
that clip's frame paths, relative to the output directory, so the whole
directory can be copied elsewhere and read through a new data root unchanged.

Trimming
    Every clip opens and closes on the signer standing still for a dozen or so
    frames.  ``--trim`` (the default) keeps only the signing span
    ``[start_frame, end_frame_exclusive)`` from the split's metadata.  Frame
    files keep their source frame number, so ``images0020.jpg`` is source frame
    19 whether or not the clip was trimmed.  ``start_frame`` and
    ``end_frame_exclusive`` are written either way, so an untrimmed extraction
    can still be cut at load time.

Framing
    Frames are resized whole by default (``--ratio 1.0``).  Hands leave through
    the bottom edge and never the top, so the evaluation crop should hug the
    bottom edge rather than the centre: set ``eval_crop_bottom_aligned: true``
    on ``SignVideoProcessor``.  From a 256 frame that crop costs 0.2 points of
    hand-bearing frames against the uncropped frame, against 7.6 for a centred
    crop.  ``--ratio`` below 1 instead bakes a bottom-anchored square of that
    fraction of the frame side into the stored frames.

    Five of the ten signers were recorded at 1280x1280, which is 1280x1080
    padded upward with a flat 200-row band.  Any crop that keeps part of it
    would mark those signers' camera setup, so the band is overwritten with the
    first real row before cropping.
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


SPLITS = ("train", "dev", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source height -> number of padded rows at the top of the frame.
PADDED_ROWS = {1280: 200}
# A padded band is a flat fill; above this per-channel std it is real content.
PAD_MAX_STD = 4.0


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
    """Source frame index -> 1-based, PHOENIX-2014-T style name."""
    return f"images{index + 1:04d}.jpg"


def output_name(ratio: float, trim: bool, size: int) -> str:
    framing = "full" if ratio == 1 else f"crop{ratio:g}b"
    return f"{framing}{'-trim' if trim else ''}-{size}x{size}px"


def has_padded_band(frame: np.ndarray, rows: int) -> bool:
    band = frame[:rows].reshape(-1, frame.shape[2])
    return bool(band.std(axis=0).max() < PAD_MAX_STD)


def crop_frame(frame, ratio, size, padded_rows):
    """Bottom-anchored, horizontally centred square crop, resized to size."""
    height, width = frame.shape[:2]
    side = int(round(ratio * min(height, width)))
    top = height - side
    left = (width - side) // 2
    if padded_rows and top < padded_rows:
        frame[:padded_rows] = frame[padded_rows]
    crop = frame[top:height, left : left + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def extract_clip(task):
    """Decode frames [start, end) of one video into <out>/<split>/<clip_id>/."""
    split, clip_id, video_path, start, end, out_root, ratio, size, quality = task
    cv2.setNumThreads(1)
    final_dir = out_root / split / clip_id
    result = dict(split=split, clip_id=clip_id, band="none", error=None)

    tmp_dir = final_dir.with_name(clip_id + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return dict(result, error=f"cannot open {video_path}")

    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    count = 0
    padded_rows = None
    try:
        # H.264 seeking is not frame-accurate, so walk the stream; grab()
        # decodes without the colour conversion read() would pay for.
        for _ in range(start):
            if not capture.grab():
                raise RuntimeError(f"stream ended before start frame {start}")
        for index in range(start, end):
            ok, frame = capture.read()
            if not ok:
                break
            if padded_rows is None:
                padded_rows = PADDED_ROWS.get(frame.shape[0], 0)
                if padded_rows:
                    if has_padded_band(frame, padded_rows):
                        result["band"] = "filled"
                    else:
                        padded_rows = 0
                        result["band"] = "expected_but_absent"
            image = crop_frame(frame, ratio, size, padded_rows)
            if not cv2.imwrite(str(tmp_dir / frame_name(index)), image, params):
                raise RuntimeError(f"failed to write frame {index}")
            count += 1
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


def build_index(split, metadata, out_root):
    """One row per clip that has a finished frame directory."""
    rows = []
    for record in metadata.itertuples(index=False):
        clip_dir = out_root / split / record.clip_id
        if not clip_dir.is_dir():
            continue
        # Zero-padded names sort in frame order.
        names = sorted(p.name for p in clip_dir.glob("images*.jpg"))
        rows.append(
            dict(
                clip_id=record.clip_id,
                signer=record.signer,
                translation=record.translation,
                gloss=record.gloss,
                start_frame=int(record.start_frame),
                end_frame_exclusive=int(record.end_frame_exclusive),
                num_frames=len(names),
                frames=[f"{split}/{record.clip_id}/{name}" for name in names],
            )
        )
    return pd.DataFrame(rows)


def load_metadata(dataset_root: Path, split: str) -> pd.DataFrame:
    metadata = pd.read_parquet(dataset_root / split / "metadata.parquet")
    metadata["signer"] = metadata.clip_id.str.extract(r"_(P\d+)_", expand=False)
    if metadata.signer.isna().any():
        bad = metadata.clip_id[metadata.signer.isna()].tolist()[:5]
        raise ValueError(f"cannot parse signer from clip ids such as {bad}")
    start, end = metadata.start_frame, metadata.end_frame_exclusive
    invalid = ~((start >= 0) & (start < end) & (end <= metadata.video_num_frames))
    if invalid.any():
        bad = metadata.clip_id[invalid].tolist()[:5]
        raise ValueError(f"invalid [start_frame, end_frame_exclusive) for clips such as {bad}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/CSL-Daily-HG"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <dataset-root>/preprocessed/<full|crop<ratio>b>[-trim]-<size>x<size>px",
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only frames [start_frame, end_frame_exclusive) of each clip (default: on)",
    )
    parser.add_argument("--ratio", type=float, default=1.0)
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

    if not 0 < args.ratio <= 1:
        parser.error("--ratio must be in (0, 1]")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in [1, 100]")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    out_root = args.out or (
        args.dataset_root / "preprocessed" / output_name(args.ratio, args.trim, args.size)
    )
    params = dict(
        trim=args.trim,
        ratio=args.ratio,
        anchor="bottom",
        size=args.size,
        interpolation="INTER_AREA",
        format="jpeg",
        quality=args.quality,
        padded_rows=PADDED_ROWS,
    )

    # Resuming reuses finished clips, so they must have been made the same way.
    meta_path = out_root / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())["params"]
        # JSON turns the int keys of PADDED_ROWS into strings; compare as JSON.
        if previous != json.loads(json.dumps(params)):
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
            if args.trim:
                start, end = int(record.start_frame), int(record.end_frame_exclusive)
            else:
                start, end = 0, int(record.video_num_frames)
            video = args.dataset_root / split / record.file_name
            tasks.append(
                (split, record.clip_id, video, start, end, out_root,
                 args.ratio, args.size, args.quality)
            )

    total = sum(len(frame) for frame in metadata.values())
    print(f"{total} clips in {args.splits}; {total - len(tasks)} already done, {len(tasks)} to extract")
    print(f"output: {out_root}")

    results = []
    with Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(extract_clip, tasks), total=len(tasks)):
            results.append(result)

    errors = [r for r in results if r["error"]]
    absent = [r["clip_id"] for r in results if r["band"] == "expected_but_absent"]
    for r in errors:
        print(f"ERROR {r['split']}/{r['clip_id']}: {r['error']}")
    if absent:
        print(
            f"WARNING {len(absent)} clips at a padded resolution had no flat band "
            f"and were cropped without filling, e.g. {absent[:5]}"
        )

    stats = {}
    for split, frame in metadata.items():
        index = build_index(split, frame, out_root)
        index.to_parquet(out_root / f"{split}.parquet", index=False)
        column = "trimmed_num_frames" if args.trim else "video_num_frames"
        expected = frame.set_index("clip_id")[column]
        got = index.set_index("clip_id").num_frames if len(index) else pd.Series(dtype=int)
        diff = (got - expected.reindex(got.index)).abs()
        stats[split] = dict(
            clips=int(len(index)),
            clips_expected=int(len(frame)),
            frames=int(got.sum()),
            frame_count_mismatches=int((diff > 0).sum()),
            max_frame_count_diff=int(diff.max()) if len(diff) else 0,
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
        this_run=dict(
            extracted=len(results),
            errors=len(errors),
            band_filled=sum(r["band"] == "filled" for r in results),
            band_expected_but_absent=absent,
        ),
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
