from torch.utils.data import Dataset
import cv2
import numpy
import os
import polars as pl
from datasets import load_dataset


class Ph14TMultiLinglDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        zh_data_root: str,
        en_data_root: str,
        mode: str = "train",
        pipline=None,
    ):
        self.data_root = data_root
        self.zh_data_root = zh_data_root
        self.en_data_root = en_data_root

        self.mode = mode

        self.hg_dataset = load_dataset(
            "WayenVan/PHOENIX-Weather14T",
            split=mode,
            name="video_level",
        )
        self.origin_df = self.hg_dataset.to_polars()

        self.zh_df = pl.read_csv(
            os.path.join(zh_data_root, f"ph14t_{mode}_Chinese.csv"),
            has_header=True,
            separator="|",
        )
        self.en_df = pl.read_csv(
            os.path.join(en_data_root, f"ph14t_{mode}_English.csv"),
            has_header=True,
            separator="|",
        )
        self._create_assemble_df()

        self.pipline = pipline
        self.ids = self.hg_dataset.unique("name")

    def _create_assemble_df(self):
        # Merge
        temp_en = pl.DataFrame(
            {
                "name": self.ids,
                "lang": ["en"] * len(self.ids),
            }
        )
        temp_zh = pl.DataFrame(
            {
                "name": self.ids,
                "lang": ["zh"] * len(self.ids),
            }
        )
        temp_de = pl.DataFrame(
            {
                "name": self.ids,
                "lang": ["de"] * len(self.ids),
            }
        )
        self.assemble_df = temp_en.vstack(temp_zh).vstack(temp_de)

    def __len__(self):
        return len(self.assemble_df)

    def __getitem__(self, idx):
        item = self.assemble_df[idx]

        lang = item["lang"].item()
        id = item["name"].item()

        data_info = self.get_data_info_by_id(id)

        video_frame_file_name = data_info["frames"]
        video_frame = []
        for frame_file in video_frame_file_name:
            image = cv2.imread(os.path.join(self.data_root, frame_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_frame.append(image)

        if lang == "zh":
            text = self.zh_df.filter(pl.col("name") == id)["translation"].to_list()[0]
        elif lang == "en":
            text = self.en_df.filter(pl.col("name") == id)["translation"].to_list()[0]
        elif lang == "de":
            text = data_info["translation"]
        else:
            raise ValueError(f"Unsupported language: {lang}")

        ret = dict(
            id=id,
            # NOTE: [time, height, width, channel], normalized to [0, 1]
            video=numpy.array(video_frame, dtype=numpy.float32) / 255.0,
            text=text,
            lang=lang,
        )

        if self.pipline:
            ret = self.pipline(ret)

        return ret

    def get_data_info_by_id(self, id: str):
        if id not in self.ids:
            raise ValueError(f"ID {id} not found in the dataset.")

        selected = self.origin_df.filter(pl.col("name") == id)

        assert len(selected) == 1, (
            f"Expected one entry for ID {id}, found {len(selected)}."
        )

        data_info = selected.to_dicts()[0]

        return data_info


if __name__ == "__main__":
    data_root = "dataset/PHOENIX-2014-T-release-v3"
    zh_data_root = "large_files/ph14t_chinese"
    en_data_root = "large_files/ph14t_english"

    ph14t_dataset = Ph14TMultiLinglDataset(
        data_root, zh_data_root=zh_data_root, en_data_root=en_data_root, mode="train"
    )
    print(f"Dataset size: {len(ph14t_dataset)}")

    for i in range(10):
        data_info = ph14t_dataset[i + 10000]
        print(data_info["text"])
        # print(
        #     f"ID: {data_info['id']}, Video shape: {data_info['video'].shape}, Text: {data_info['text']}"
        # )
