"""Train the CSL-Daily word-level CTC tokenizer from pseudo glosses.

The vocabulary is built from ``pseudo_gloss_relaxed`` in all three local
splits (train, dev and test).  The relaxed pseudo-gloss vocabulary is a
superset of the default and strict variants, so the saved tokenizer can be
used with any of them without retraining.

Usage::

    python preprocess/csl_daily/train_ctc_tokenizer_csl_daily.py

    python preprocess/csl_daily/train_ctc_tokenizer_csl_daily.py \
        --data-root /path/to/full-trim-256x256px \
        --output-dir outputs/ctc_tokenizer_csl_daily_relaxed
"""

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "dataset/CSL-Daily-HG/preprocessed/full-trim-256x256px"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/ctc_tokenizer_csl_daily_relaxed"
SPLIT_FILES = {
    "train": "train.parquet",
    "validation": "dev.parquet",
    "test": "test.parquet",
}
TEXT_COLUMN = "pseudo_gloss_relaxed"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<blank>")


def batch_iterator(dataset, batch_size: int):
    """Yield non-null pseudo-gloss strings without changing their spaces."""

    for start in range(0, len(dataset), batch_size):
        values = dataset[start : start + batch_size][TEXT_COLUMN]
        yield [value if value is not None else "" for value in values]


def train_tokenizer(data_root: Path, output_dir: Path, batch_size: int) -> None:
    data_files = {
        split: str(data_root / filename)
        for split, filename in SPLIT_FILES.items()
    }
    missing = [path for path in data_files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing CSL-Daily parquet files: {missing}")

    splits = load_dataset(
        "parquet",
        data_files=data_files,
        columns=[TEXT_COLUMN],
    )
    full_set = concatenate_datasets([splits[name] for name in SPLIT_FILES])

    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    # Pseudo-gloss words are already separated by ASCII spaces.  Do not apply
    # Chinese character segmentation or punctuation splitting here.
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        special_tokens=list(SPECIAL_TOKENS),
        min_frequency=1,
    )
    tokenizer.train_from_iterator(
        batch_iterator(full_set, batch_size),
        trainer=trainer,
        length=len(full_set),
    )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        # Transformers has no standard blank_token constructor argument.
        extra_special_tokens={"blank_token": "<blank>"},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)

    # Reload from disk so a partially or incorrectly serialized tokenizer is
    # caught before its IDs are copied into a model configuration.
    reloaded = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    blank_id = reloaded.convert_tokens_to_ids("<blank>")
    if blank_id == reloaded.unk_token_id:
        raise RuntimeError("<blank> was not preserved in the saved tokenizer")

    print(f"rows: {len(full_set)}")
    print(f"text column: {TEXT_COLUMN}")
    print(f"vocab size: {len(reloaded)}")
    print(f"pad: {reloaded.pad_token_id}")
    print(f"unk: {reloaded.unk_token_id}")
    print(f"blank: {blank_id}")
    print(f"saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CSL-Daily word-level CTC tokenizer."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "directory containing train/dev/test parquet files "
            f"(default: {DEFAULT_DATA_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"tokenizer output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="number of rows yielded to the tokenizer trainer at once",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    train_tokenizer(args.data_root, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
