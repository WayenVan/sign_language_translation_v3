from transformers.video_processing_utils import BaseVideoProcessor, BatchFeature
from transformers.utils import TensorType, filter_out_non_signature_kwargs
import numpy as np

from typing import Union
import torch

from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ColorJitter,
)


class SignVideoProcessor(BaseVideoProcessor):
    _auto_class = "AutoVideoProcessor"
    model_input_names = ["pixel_values", "pixel_values_lengths"]
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __init__(self, height=224, width=224, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    @property
    def train_transform(self):
        return Compose(
            [
                # Resize(height=256, width=256),
                RandomCrop(height=self.height, width=self.width, p=1.0),
                ColorJitter(p=0.75),
                Normalize(
                    mean=self.image_mean,
                    std=self.image_std,
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )

    @property
    def predict_transform(self):
        return Compose(
            [
                # Resize(height=256, width=256),
                CenterCrop(height=self.height, width=self.width, p=1.0),
                Normalize(
                    mean=self.image_mean,
                    std=self.image_std,
                    max_pixel_value=1.0,
                ),
            ],
            p=1.0,
        )

    @staticmethod
    def pad_dim_to_multiple_of(array, dim, multiple):
        current_size = array.shape[dim]
        remainder = current_size % multiple
        if remainder == 0:
            return array

        pad_size = multiple - remainder

        # 取这个维度的最后一个元素
        index = [slice(None)] * array.ndim
        index[dim] = -1
        last_element = np.take(array, -1, axis=dim)
        last_element = np.expand_dims(last_element, axis=dim)

        # 复制 pad_size 次
        padding = np.repeat(last_element, pad_size, axis=dim)
        return np.concatenate([array, padding], axis=dim)

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        videos: Union[list[np.ndarray], np.ndarray],
        training: bool = True,
        padding_to_multiple_of: int = 4,
    ):
        if isinstance(videos, np.ndarray):
            videos = [videos]

        processs_fn = self.train_transform if training else self.predict_transform

        processed_videos = []
        video_lengths = []
        for video in videos:
            video = self.pad_dim_to_multiple_of(
                video, dim=0, multiple=padding_to_multiple_of
            )

            video_lengths.append(video.shape[0])
            processed = processs_fn(images=video)["images"]
            processed = (
                torch.from_numpy(
                    processed,
                )
                .permute(0, 3, 1, 2)
                .float()
            )  # T, C, H, W conver to tensor
            processed_videos.append(processed)

        video_tensor = torch.cat(
            processed_videos,
            dim=0,
        ).contiguous()
        video_lengths_tensor = torch.tensor(
            video_lengths, dtype=torch.long
        ).contiguous()

        data = {
            "pixel_values": video_tensor,
            "pixel_values_lengths": video_lengths_tensor,
        }
        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)

    def to_dict(self):
        output = super().to_dict()
        output.pop("train_transform", None)
        output.pop("prediction_transform", None)
        return output
