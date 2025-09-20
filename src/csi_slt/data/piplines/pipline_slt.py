from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ColorJitter,
)
from ..transforms import ToTensorVideo

MEAN = (0.5, 0.5, 0.5)  # to fit the perception encoder
STD = (0.5, 0.5, 0.5)


class SLTGeneralPiplineTrain:
    def __init__(self, height=224, width=224, downsample_rate=1):
        # video transforms
        self.resize_cop = Compose(
            [
                Resize(height=256, width=256),
                RandomCrop(height=height, width=width, p=1.0),
            ],
            p=1.0,
        )

        self.warp = Compose(
            [
                ColorJitter(p=0.75),
                Normalize(
                    mean=MEAN,
                    std=STD,
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )
        self.to_tensor = ToTensorVideo()

        # text transforms
        # self.delete = RandomWordAug(action="delete", aug_p=0.5)
        # self.insert = RandomWordAug(action="insert", aug_p=0.5)
        self.downsample_rate = downsample_rate

    def __call__(self, data):
        video = data["video"]
        text = data["text"]

        video = self.resize_cop(images=video)["images"]
        video = self.warp(images=video)["images"]
        video = video[:: self.downsample_rate]  # downsample video
        video = ToTensorVideo()(video)

        # text = self.delete.augment(text)[0]
        # text = self.insert.augment(text)

        data["augmented_video"] = video
        data["augmented_text"] = text

        return data


class SLTGeneralPiplineTest:
    def __init__(self, height=224, width=224, downsample_rate=1):
        # video transforms
        self.downsample_rate = downsample_rate
        self.resize_cop = Compose(
            [
                Resize(height=256, width=256, p=1.0),
                CenterCrop(height=height, width=width, p=1.0),
            ],
            p=1.0,
        )

        self.warp = Compose(
            [
                Normalize(
                    mean=MEAN,
                    std=STD,
                    max_pixel_value=1.0,
                ),
            ],
            p=1.0,
        )

        self.to_tensor = ToTensorVideo()

    def __call__(self, data):
        video = data["video"]
        text = data["text"]

        video = self.resize_cop(images=video)["images"]
        video = self.warp(images=video)["images"]
        video = video[:: self.downsample_rate]
        video = self.to_tensor(video)

        data["augmented_video"] = video
        data["augmented_text"] = text
        return data
