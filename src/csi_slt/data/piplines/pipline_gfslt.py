from nlpaug.augmenter.word.random import RandomWordAug
from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ShiftScaleRotate,
)
from ..transforms import JitteredUniformSampleVideo, UniformSampleVideo, ToTensorVideo


class GFSLTPiplineTrain:
    def __init__(self, height=224, width=224):
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
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )

        # text transforms
        # self.delete = RandomWordAug(action="delete", aug_p=0.5)
        # self.insert = RandomWordAug(action="insert", aug_p=0.5)

    def __call__(self, data):
        video = data["video"]

        video = self.resize_cop(images=video)["images"]
        video = self.warp(images=video)["images"]

        video = ToTensorVideo()(video)
        # text = self.delete.augment(text)[0]
        # text = self.insert.augment(text)

        data["augmented_video"] = video

        return data


class GFSLTPiplineTest:
    def __init__(self, height=224, width=224, jittered_sample=True):
        # video transforms
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
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
            ],
            p=1.0,
        )

        self.to_tensor = ToTensorVideo()

    def __call__(self, data):
        video = data["video"]

        video = self.resize_cop(images=video)["images"]
        video = self.warp(images=video)["images"]
        video = self.to_tensor(video)

        data["augmented_video"] = video

        return data
