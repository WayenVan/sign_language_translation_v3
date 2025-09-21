from torch.utils.data import Dataset
import numpy
import os
from datasets import load_dataset


class Ph14TGeneralDataset(Dataset):
    def __init__(self, data_root: str, mode: str = "train", pipline=None):
        self.data_root = data_root
        self.hg_dataset = load_dataset(
            "WayenVan/PHOENIX-Weather14T",
            split=mode,
            name="video_level",
        )
        # self.df = self.hg_dataset.to_polars()
        self.mode = mode

        self.pipline = pipline

    def __len__(self):
        return len(self.ids)

    @property
    def ids(self):
        return self.hg_dataset.unique("name")

    def __getitem__(self, idx):
        id = self.ids[idx]
        data_info = self.hg_dataset[idx]

        video_frame_file_name = data_info["frames"]
        video_frame = []
        import cv2

        for frame_file in video_frame_file_name:
            image = cv2.imread(os.path.join(self.data_root, frame_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_frame.append(image)

        ret = dict(
            id=id,
            # NOTE: [time, height, width, channel], normalized to [0, 1]
            video=numpy.array(video_frame, dtype=numpy.float32) / 255.0,
            text=data_info["translation"],
            lang="de",
        )

        if self.pipline:
            ret = self.pipline(ret)

        return ret


if __name__ == "__main__":
    data_root = "dataset/PHOENIX-2014-T-release-v3/"
    ph14t_dataset = Ph14TGeneralDataset(data_root, mode="train")
    print(f"Dataset size: {len(ph14t_dataset)}")
    for i in range(10):
        data_info = ph14t_dataset[i]
        print(
            f"ID: {data_info['id']}, Video shape: {data_info['video'].shape}, Text: {data_info['text']}"
        )
