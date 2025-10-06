from transformers.processing_utils import ProcessorMixin
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
from transformers.video_processing_utils import BaseVideoProcessor, BatchFeature
from transformers.utils import TensorType, filter_out_non_signature_kwargs
import numpy as np

from typing import Union

from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ColorJitter,
)


VIDEO_SOFT_TOKEN = "<unused0>"
VIDEO_START_TOKEN = "<unused1>"


class SignVideoProcessor(BaseVideoProcessor):
    # _auto_class = "AutoProcessor"
    model_input_names = ["pixel_values", "pixel_values_lengths"]
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __init__(self, height=224, width=224, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.downsample_rate = downsample_rate

    @property
    def train_transform(self):
        return Compose(
            [
                Resize(height=256, width=256),
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
    def prediction_transform(self):
        return Compose(
            [
                Resize(height=256, width=256),
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
            video = video[:: self.downsample_rate]  # downsample video

            video_lengths.append(video.shape[0])
            processed = processs_fn(images=video)["images"]
            processed_videos.append(processed)

        data = {"pixel_values": processed_videos, "pixel_values_lengths": video_lengths}
        return BatchFeature(data=data, tensor_type=TensorType.NUMPY)

    def to_dict(self):
        output = super().to_dict()
        output.pop("train_transform", None)
        output.pop("prediction_transform", None)
        return output


if __name__ == "__main__":
    processor = SignVideoProcessor()
    processor.train_transform
    processor.prediction_transform
    processor.save_pretrained("outputs/processor_test")
