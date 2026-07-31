from transformers.video_processing_utils import (
    BaseVideoProcessor,
    BatchFeature,
    VideosKwargs,
)
from transformers.processing_utils import Unpack
from transformers.utils import TensorType
import numpy as np

from collections.abc import Sequence
from typing import Union
import torch

from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    ColorJitter,
)


class SignVideoKwargs(VideosKwargs, total=False):
    """Custom kwargs accepted by SignVideoProcessor."""

    padding_to_multiple_of: int


class SignVideoProcessor(BaseVideoProcessor):
    _auto_class = "AutoVideoProcessor"

    model_input_names = ["pixel_values", "pixel_values_lengths"]
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    size = {"height": 224, "width": 224}
    padding_to_multiple_of = 4

    # ------- class attributes for input validation -------
    expected_input_size = {"height": 256, "width": 256}

    valid_kwargs = SignVideoKwargs

    def __init__(self, **kwargs: Unpack[SignVideoKwargs]):
        super().__init__(**kwargs)

    @staticmethod
    def build_train_transform(size, image_mean, image_std):
        return Compose(
            [
                # Resize(height=256, width=256),
                RandomCrop(height=size["height"], width=size["width"], p=1.0),
                ColorJitter(p=0.75),
                Normalize(
                    mean=image_mean,
                    std=image_std,
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )

    @staticmethod
    def build_predict_transform(size, image_mean, image_std):
        return Compose(
            [
                # Resize(height=256, width=256),
                CenterCrop(height=size["height"], width=size["width"], p=1.0),
                Normalize(
                    mean=image_mean,
                    std=image_std,
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

    def preprocess(
        self,
        videos: Union[Sequence[np.ndarray], np.ndarray],
        training: bool = True,
        **kwargs: Unpack[SignVideoKwargs],
    ):
        videos = self._prepare_videos(videos)

        # Instance attributes provide defaults. Values supplied for this call
        # temporarily override them without mutating the processor.
        size = kwargs.pop("size", self.size)
        image_mean = kwargs.pop("image_mean", self.image_mean)
        image_std = kwargs.pop("image_std", self.image_std)
        padding_to_multiple_of = kwargs.pop(
            "padding_to_multiple_of", self.padding_to_multiple_of
        )

        process_fn = (
            self.build_train_transform(size, image_mean, image_std)
            if training
            else self.build_predict_transform(size, image_mean, image_std)
        )

        processed_videos = []
        video_lengths = []
        for video in videos:
            video = self.pad_dim_to_multiple_of(
                video, dim=0, multiple=padding_to_multiple_of
            )

            video_lengths.append(video.shape[0])
            processed = process_fn(images=video)["images"]
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

    def _prepare_videos(
        self, videos: Union[Sequence[np.ndarray], np.ndarray]
    ) -> list[np.ndarray]:
        """Normalize supported inputs to a validated list of THWC videos."""
        if isinstance(videos, np.ndarray):
            if videos.ndim == 4:
                video_list = [videos]
            elif videos.ndim == 5:
                video_list = list(videos)
            else:
                raise ValueError(
                    "A NumPy input must have shape (T, H, W, C) or "
                    f"(B, T, H, W, C), but received {videos.shape}."
                )
        elif isinstance(videos, Sequence) and not isinstance(videos, (str, bytes)):
            video_list = list(videos)
        else:
            raise TypeError(
                "videos must be a NumPy array or a sequence of NumPy arrays, "
                f"but received {type(videos).__name__}."
            )

        if not video_list:
            raise ValueError("videos must contain at least one video.")

        expected_height = self.expected_input_size["height"]
        expected_width = self.expected_input_size["width"]
        for index, video in enumerate(video_list):
            if not isinstance(video, np.ndarray):
                raise TypeError(
                    f"videos[{index}] must be a NumPy array, but received "
                    f"{type(video).__name__}."
                )
            if video.ndim != 4:
                raise ValueError(
                    f"videos[{index}] must have shape (T, H, W, C), but "
                    f"received {video.shape}."
                )
            if video.shape[0] == 0:
                raise ValueError(f"videos[{index}] must contain at least one frame.")
            if video.shape[1:3] != (expected_height, expected_width):
                raise ValueError(
                    f"videos[{index}] must have spatial size "
                    f"({expected_height}, {expected_width}), but received "
                    f"{video.shape[1:3]}."
                )
            if video.shape[3] != 3:
                raise ValueError(
                    f"videos[{index}] must have 3 channels in THWC format, but "
                    f"received {video.shape[3]}."
                )

        return video_list

    def to_dict(self):
        output = super().to_dict()
        return output


if __name__ == "__main__":
    video_processor = SignVideoProcessor(
        padding_to_multiple_of=4,
        size={"height": 256, "width": 256},
    )
    print(video_processor)
