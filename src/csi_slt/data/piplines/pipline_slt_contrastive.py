from nlpaug.augmenter.word.random import RandomWordAug
from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ShiftScaleRotate,
    ColorJitter,
)
from ..transforms import JitteredUniformSampleVideo, UniformSampleVideo, ToTensorVideo


class SLTContrastivePiplineTrain:
    def __init__(self, height=224, width=224, downsample_rate=1):
        # video transforms
        self.resize_norm = Compose(
            [
                Resize(height=256, width=256, p=1.0),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
            ]
        )

        self.aug = Compose(
            [
                RandomCrop(height=height, width=width, p=1.0),
                HorizontalFlip(p=0.5),
                ColorJitter(),
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

        video = video[:: self.downsample_rate]  # downsample video
        video_cropped = self.resize_norm(images=video)["images"]

        video_aug1 = self.aug(images=video_cropped)["images"]
        video_aug2 = self.aug(images=video_cropped)["images"]

        video_aug1 = self.to_tensor(video_aug1)
        video_aug2 = self.to_tensor(video_aug2)

        # text = self.delete.augment(text)[0]
        # text = self.insert.augment(text)

        data["augmented_video_1"] = video_aug1
        data["augmented_video_2"] = video_aug2

        data["augmented_text"] = text
        return data


class SLTContrastivePiplineTest:
    def __init__(self, height=224, width=224, downsample_rate=1):
        # video transforms
        self.resize_norm = Compose(
            [
                Resize(height=256, width=256, p=1.0),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
            ]
        )

        self.aug = Compose(
            [
                RandomCrop(height=height, width=width, p=1.0),
                HorizontalFlip(p=0.5),
                ColorJitter(),
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

        video = video[:: self.downsample_rate]  # downsample video
        video_cropped = self.resize_norm(images=video)["images"]

        video_aug1 = self.aug(images=video_cropped)["images"]
        video_aug2 = self.aug(images=video_cropped)["images"]

        video_aug1 = self.to_tensor(video_aug1)
        video_aug2 = self.to_tensor(video_aug2)

        # text = self.delete.augment(text)[0]
        # text = self.insert.augment(text)

        data["augmented_video_1"] = video_aug1
        data["augmented_video_2"] = video_aug2

        data["augmented_text"] = text
        return data


# class SLTGeneralPiplineTest:
#     def __init__(self, height=224, width=224, downsample_rate=1):
#         # video transforms
#         self.downsample_rate = downsample_rate
#         self.resize_cop = Compose(
#             [
#                 Resize(height=256, width=256, p=1.0),
#                 CenterCrop(height=height, width=width, p=1.0),
#             ],
#             p=1.0,
#         )
#
#         self.warp = Compose(
#             [
#                 Normalize(
#                     mean=(0.485, 0.456, 0.406),
#                     std=(0.229, 0.224, 0.225),
#                     max_pixel_value=1.0,
#                 ),
#             ],
#             p=1.0,
#         )
#
#         self.to_tensor = ToTensorVideo()
#
#     def __call__(self, data):
#         video = data["video"]
#         text = data["text"]
#
#         video = self.resize_cop(images=video)["images"]
#         video = self.warp(images=video)["images"]
#         video = video[:: self.downsample_rate]
#         video = self.to_tensor(video)
#
#         data["augmented_video"] = video
#         data["augmented_text"] = text
#         return data
