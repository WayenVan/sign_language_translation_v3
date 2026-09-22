"""CSL-Daily as a PyTorch dataset over the frames of extract_frames.py."""

import os
from pathlib import Path

import cv2
import numpy
from datasets import config as datasets_config
from datasets import load_dataset, load_from_disk
from filelock import FileLock
from torch.utils.data import Dataset


# The data configs name the held-out split "validation", after PHOENIX14T's
# Hugging Face splits; CSL-Daily calls it "dev".
SPLIT_FILES = {
    "train": "train.parquet",
    "validation": "dev.parquet",
    "dev": "dev.parquet",
    "test": "test.parquet",
}


def _estimate_label_lengths(batch, tokenizer):
    """Tokenize labels in batches; ``tokenizer`` participates in map hashing."""

    eos_token = tokenizer.eos_token or ""
    labels = [text + eos_token for text in batch["translation"]]
    tokenized = tokenizer(
        labels,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return {
        "label_ids_length": [len(input_ids) for input_ids in tokenized["input_ids"]]
    }


def sentence_id(clip_id: str) -> str:
    """S000005_P0004_T00 -> S000005, the sentence every signer's take shares."""

    return clip_id.split("_", 1)[0]


class CSLDailyDataset(Dataset):
    """CSL-Daily clips as THWC uint8 frame stacks with Chinese targets.

    ``data_root`` is an output directory of
    ``preprocess/csl_daily/extract_frames.py`` (for example
    ``dataset/CSL-Daily-HG/preprocessed/full-trim-256x256px``).  Its
    ``<split>.parquet`` lists every clip's frame paths in order, relative to
    ``data_root``, so the directory can be copied elsewhere and read through a
    new root.  Whether the idle frames around the signing were trimmed was
    decided at extraction time; this class reads whatever the index lists.

    A pseudo-gloss column can be selected for auxiliary objectives such as
    CTC.  Leaving ``pseudo_gloss_column`` unset preserves the text-only setup
    and makes each sample's ``pseudo_gloss`` value None.
    """

    LANGUAGE = "zh"

    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        pseudo_gloss_column: str | None = None,
        pipline=None,
    ):
        if mode not in SPLIT_FILES:
            raise ValueError(f"mode must be one of {sorted(SPLIT_FILES)}, got {mode!r}")
        # A half-written extraction would train on whichever clips happened to
        # finish; refuse it rather than quietly shrinking the split.
        if not (Path(data_root) / ".complete").exists():
            raise FileNotFoundError(
                f"{data_root} has no .complete marker; finish "
                "preprocess/csl_daily/extract_frames.py first"
            )

        self.data_root = data_root
        self.mode = mode
        self.pseudo_gloss_column = pseudo_gloss_column
        self.pipline = pipline

        self.hg_dataset = load_dataset(
            "parquet",
            data_files={mode: os.path.join(data_root, SPLIT_FILES[mode])},
            split=mode,
        )
        if (
            self.pseudo_gloss_column is not None
            and self.pseudo_gloss_column not in self.hg_dataset.column_names
        ):
            raise ValueError(
                f"pseudo-gloss column {self.pseudo_gloss_column!r} is missing "
                f"from {SPLIT_FILES[mode]}"
            )

    @property
    def cache_namespace(self) -> str:
        """Directory namespace for this dataset variant's prepared artifacts."""

        return "csl_daily"

    def __len__(self):
        return len(self.hg_dataset)

    def _text_item_from_info(self, data_info):
        """Return metadata needed by text-only objectives such as D-SID."""
        return dict(
            id=data_info["clip_id"],
            # Every signer's take of one sentence shares its sentence id, so
            # those clips are semantically equivalent positives for the global
            # contrastive objective.
            semantic_ids=sentence_id(data_info["clip_id"]),
            text=data_info["translation"],
            lang=self.LANGUAGE,
            pseudo_gloss=(
                data_info[self.pseudo_gloss_column]
                if self.pseudo_gloss_column is not None
                else None
            ),
        )

    def get_text_item(self, idx):
        """Read one sample without opening its video frames."""
        ret = self._text_item_from_info(self.hg_dataset[idx])
        if self.pipline:
            ret = self.pipline(ret)
        return ret

    def _read_frame(self, frame_file):
        path = os.path.join(self.data_root, frame_file)
        image = cv2.imdecode(numpy.fromfile(path, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        data_info = self.hg_dataset[idx]

        ret = {
            **self._text_item_from_info(data_info),
            # THWC uint8 in [0, 255]; SignVideoProcessor converts it to float32.
            "video": numpy.stack([self._read_frame(f) for f in data_info["frames"]]),
        }

        if self.pipline:
            ret = self.pipline(ret)

        return ret

    def prepare(self, tokenizer, cache_dir: str | os.PathLike | None = None):
        """Use Datasets' native fingerprint/cache system for token lengths."""

        cache_root = Path(cache_dir or datasets_config.HF_DATASETS_CACHE)
        cache_root = cache_root / "csi_slt" / self.cache_namespace
        cache_root.mkdir(parents=True, exist_ok=True)
        assembled_path = cache_root / (
            f"assembled-{self.mode}-{self.hg_dataset._fingerprint}"
        )
        lock = FileLock(f"{assembled_path}.lock")

        # Persist once so map() has a disk-backed dataset to place reusable
        # cache files next to; its cache fingerprint then covers the function,
        # its arguments (including the tokenizer) and this dataset.
        with lock:
            if not assembled_path.exists():
                self.hg_dataset.save_to_disk(assembled_path)

            self.hg_dataset = load_from_disk(assembled_path)
            self.hg_dataset = self.hg_dataset.map(
                _estimate_label_lengths,
                batched=True,
                batch_size=1000,
                fn_kwargs={"tokenizer": tokenizer},
                load_from_cache_file=True,
                desc=f"Tokenizing {self.mode} labels for length bucketing",
            )

        self.label_ids_lengths = [
            int(length) for length in self.hg_dataset["label_ids_length"]
        ]
        # The index already records each clip's frame count; no image is opened.
        self.video_lengths = [int(n) for n in self.hg_dataset["num_frames"]]

    @classmethod
    def create_prepared_dataset(
        cls, tokenizer, *args, cache_dir: str | os.PathLike | None = None, **kwargs
    ):
        dataset = cls(*args, **kwargs)
        dataset.prepare(tokenizer, cache_dir=cache_dir)
        return dataset
