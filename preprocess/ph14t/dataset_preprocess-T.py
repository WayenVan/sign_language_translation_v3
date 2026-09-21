import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
from tqdm import tqdm


SPLITS = ("dev", "test", "train")


def parse_resolution(value):
    """Convert a value such as ``256x256px`` to an OpenCV (width, height) tuple."""
    normalized = value.lower().removesuffix("px")
    try:
        width, height = (int(size) for size in normalized.split("x"))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "resolution must use the WIDTHxHEIGHTpx format, for example 256x256px"
        ) from None

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def resize_sequence(sequence_dir, source_root, destination_root, dsize):
    """Resize every PNG in one sequence while preserving its relative path."""
    sequence_dir = Path(sequence_dir)
    source_root = Path(source_root)
    destination_root = Path(destination_root)

    output_dir = destination_root / sequence_dir.relative_to(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(sequence_dir.glob("*.png")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")

        resized = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)
        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), resized):
            raise RuntimeError(f"failed to write image: {output_path}")


def find_sequences(source_root):
    """Return sequence directories from the train/dev/test dataset splits."""
    source_root = Path(source_root)
    return [
        sequence_dir
        for split in SPLITS
        for sequence_dir in sorted((source_root / split).glob("*"))
        if sequence_dir.is_dir()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Resize PHOENIX-2014-T frame sequences."
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T",
        help="path to the PHOENIX-2014-T dataset",
    )
    parser.add_argument(
        "--input-res",
        default="210x260px",
        help="source frame directory suffix (default: 210x260px)",
    )
    parser.add_argument(
        "--output-res",
        default="256x256px",
        help="output resolution in WIDTHxHEIGHTpx format (default: 256x256px)",
    )
    parser.add_argument(
        "--multiprocessing",
        "-m",
        action="store_true",
        help="resize sequences using multiple worker processes",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        "-w",
        type=int,
        default=os.cpu_count() or 1,
        help="number of multiprocessing workers (default: available CPU count)",
    )
    args = parser.parse_args()

    dsize = parse_resolution(args.output_res)
    source_root = (
        Path(args.dataset_root) / "features" / f"fullFrame-{args.input_res}"
    )
    destination_root = (
        Path(args.dataset_root) / "features" / f"fullFrame-{args.output_res}"
    )

    if not source_root.is_dir():
        parser.error(f"source frame directory does not exist: {source_root}")
    if source_root.resolve() == destination_root.resolve():
        parser.error("input and output resolutions must be different")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")

    sequences = find_sequences(source_root)
    if not sequences:
        parser.error(f"no frame sequences found under: {source_root}")

    resize_func = partial(
        resize_sequence,
        source_root=source_root,
        destination_root=destination_root,
        dsize=dsize,
    )
    print(
        f"Resize {len(sequences)} sequences from {source_root} "
        f"to {destination_root}"
    )

    if args.multiprocessing:
        with Pool(args.num_workers) as pool:
            list(tqdm(pool.imap(resize_func, sequences), total=len(sequences)))
    else:
        for sequence_dir in tqdm(sequences):
            resize_func(sequence_dir)


if __name__ == "__main__":
    main()
