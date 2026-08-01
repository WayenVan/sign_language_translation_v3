from torch.utils.data import Dataset
import numpy
import os
import polars as pl
from datasets import load_dataset
import pyspng
from datasets import Dataset as HFDataset


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
        self.de_df = self.origin_df.select(["name", "translation"])

        self.ids = self.hg_dataset.unique("name")
        self.pipline = pipline

        self._create_assemble_df()

    def _create_assemble_df(self):
        # Merge
        #
        zh_df = self.zh_df.with_columns(pl.lit("zh").alias("lang"))
        en_df = self.en_df.with_columns(pl.lit("en").alias("lang"))
        de_df = self.de_df.with_columns(pl.lit("de").alias("lang"))

        assemble_df = en_df.vstack(zh_df).vstack(de_df)
        assemble_df = assemble_df.join(
            self.origin_df.drop("translation"), on="name", how="left"
        )
        self.assemble_df = assemble_df

        # 只读取 frames 列的列表长度，不打开任何图片。
        self.lengths = (
            self.assemble_df.select(pl.col("frames").list.len().alias("length"))
            .get_column("length")
            .to_list()
        )

        self.assemble_dataset = HFDataset(self.assemble_df.to_arrow())

    def __len__(self):
        return len(self.assemble_df)

    def __getitem__(self, idx):
        data_info = self.assemble_dataset[idx]

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
            # NOTE: [time, height, width, channel], normalized to [0, 1]
            video=numpy.array(video_frame, dtype=numpy.float32) / 255.0,
            text=data_info["translation"],
            lang=data_info["lang"],
        )

        if self.pipline:
            ret = self.pipline(ret)

        return ret


if __name__ == "__main__":
    data_root = "dataset/PHOENIX-2014-T-release-v3"
    zh_data_root = "large_files/ph14t_chinese"
    en_data_root = "large_files/ph14t_english"

    ph14t_dataset = Ph14TMultiLinglDataset(
        data_root, zh_data_root=zh_data_root, en_data_root=en_data_root, mode="train"
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
