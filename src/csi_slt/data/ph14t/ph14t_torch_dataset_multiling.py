from torch.utils.data import Dataset
import numpy
import os
from pathlib import Path
from datasets import config as datasets_config
from datasets import load_dataset, load_from_disk
from filelock import FileLock
import pyspng


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


class Ph14TMultiLinglDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        pseudo_gloss_column: str = "pseudo_gloss_strict",
        pipline=None,
    ):
        self.data_root = data_root
        self.mode = mode
        self.pseudo_gloss_column = pseudo_gloss_column
        self.pipline = pipline

        self.hg_dataset = load_dataset(
            "WayenVan/ph14t-multilang",
            split=mode,
        )

    def __len__(self):
        return len(self.hg_dataset)

    def __getitem__(self, idx):
        data_info = self.hg_dataset[idx]

        video_frame_file_name = data_info["frames"]
        video_frame = []

        for frame_file in video_frame_file_name:
            frame_file = frame_file.replace("210x260px", "256x256px")
            with open(os.path.join(self.data_root, frame_file), "rb") as f:
                image = pyspng.load(f.read())
            image = image[:, :, :3]

            video_frame.append(image)

        ret = dict(
            id=data_info["name"],
            # THWC uint8 in [0, 255]; SignVideoProcessor converts it to float32.
            video=numpy.array(video_frame, dtype=numpy.uint8),
            text=data_info["translation"],
            lang=data_info["lang"],
            pseudo_gloss=data_info[self.pseudo_gloss_column],
        )

        if self.pipline:
            ret = self.pipline(ret)

        return ret

    def prepare(self, tokenizer, cache_dir: str | os.PathLike | None = None):
        """Use Datasets' native fingerprint/cache system for token lengths."""

        cache_root = Path(cache_dir or datasets_config.HF_DATASETS_CACHE)
        cache_root = cache_root / "csi_slt" / "ph14t_multilingual"
        cache_root.mkdir(parents=True, exist_ok=True)
        assembled_path = cache_root / (
            f"assembled-{self.mode}-{self.hg_dataset._fingerprint}"
        )
        lock = FileLock(f"{assembled_path}.lock")

        # Dataset(table) is memory-backed and therefore has no directory in which
        # map() can place reusable cache files. Persist it once, then let map()
        # derive cache fingerprints from the function, its arguments (including
        # the tokenizer), and the disk-backed dataset fingerprint.
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
        # 只读取 frames 列的列表长度，不打开任何图片。
        self.video_lengths = [len(frames) for frames in self.hg_dataset["frames"]]

    @classmethod
    def create_prepared_dataset(
        cls, tokenizer, *args, cache_dir: str | os.PathLike | None = None, **kwargs
    ):
        dataset = cls(*args, **kwargs)
        dataset.prepare(tokenizer, cache_dir=cache_dir)
        return dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer

    data_root = "dataset/PHOENIX-2014-T-release-v3"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ph14t_dataset = Ph14TMultiLinglDataset.create_prepared_dataset(
        tokenizer,
        data_root,
        mode="validation",
    )

    print(f"Dataset size: {len(ph14t_dataset)}")
    # print(ph14t_dataset.assemble_df.columns)
    # print(ph14t_dataset.origin_df.columns)

    for i in range(10):
        data_info = ph14t_dataset[i + 10]
        print(data_info["text"])
        print(data_info["video"].shape)
    # print(
    #     f"ID: {data_info['id']}, Video shape: {data_info['video'].shape}, Text: {data_info['text']}"
    # )
