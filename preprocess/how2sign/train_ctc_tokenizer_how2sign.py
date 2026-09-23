"""Train a word-level How2Sign CTC tokenizer from relaxed pseudo-glosses.

The vocabulary is built from ``pseudo_gloss_relaxed`` across train,
validation and test.  Relaxed is a superset of the default and strict POS
variants, so one tokenizer supports all three.

Usage::

    python preprocess/how2sign/train_ctc_tokenizer_how2sign.py

    python preprocess/how2sign/train_ctc_tokenizer_how2sign.py \
        --data-root /path/to/how2sign/preprocessed \
        --output-dir outputs/ctc_tokenizer_how2sign_relaxed
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
    PROJECT_ROOT
    / "dataset/how2sign-front-clips/preprocessed/sq720-x+50-24fps-256x256px"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/ctc_tokenizer_how2sign_relaxed"
SPLIT_FILES = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "test": "test.parquet",
}
TEXT_COLUMN = "pseudo_gloss_relaxed"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<blank>")


def batch_iterator(dataset, batch_size: int):
    """Yield pseudo-gloss strings in bounded batches."""
    for start in range(0, len(dataset), batch_size):
        yield dataset[start : start + batch_size][TEXT_COLUMN]


def train_tokenizer(data_root: Path, output_dir: Path, batch_size: int) -> None:
    data_files = {
        split: str(data_root / filename) for split, filename in SPLIT_FILES.items()
    }
    missing = [path for path in data_files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing How2Sign parquet files: {missing}")

    splits = load_dataset("parquet", data_files=data_files, columns=[TEXT_COLUMN])
    full_set = concatenate_datasets([splits[name] for name in SPLIT_FILES])
    empty_rows = sum(
        value is None or not value.strip() for value in full_set[TEXT_COLUMN]
    )
    if empty_rows:
        raise RuntimeError(
            f"{empty_rows} rows have empty {TEXT_COLUMN}; rebuild pseudo-glosses "
            "before training a CTC tokenizer"
        )

    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
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
        extra_special_tokens={"blank_token": "<blank>"},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)

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
        description="Train the How2Sign word-level CTC tokenizer."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    train_tokenizer(args.data_root, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
