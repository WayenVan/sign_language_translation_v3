from transformers.video_processing_utils import (
    BaseVideoProcessor,
    BatchFeature,
    VideosKwargs,
)
from transformers.processing_utils import Unpack
from transformers.utils import TensorType
from typing import Any
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from csi_slt.data.processors.video_transforms import RandomVideoSpeed


class SignVideoKwargs(VideosKwargs, total=False):
    """Custom kwargs accepted by SignVideoProcessor."""

    padding_to_multiple_of: int
    training: bool
    do_random_speed: bool
    random_speed_range: tuple[float, float]


class SignVideoProcessor(BaseVideoProcessor):
    _auto_class = "AutoVideoProcessor"

    # processor kwargs
    model_input_names = ["pixel_values", "pixel_values_lengths"]
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    crop_size = {"height": 224, "width": 224}
    padding_to_multiple_of = 4
    training = False
    do_normalize = True
    do_resize = False
    do_random_speed = True  # random speed augmentation
    random_speed_range = (0.8, 1.25)
    input_data_format = "channels_last"
    size = {"height": 224, "width": 224}

    valid_kwargs = SignVideoKwargs

    def __init__(self, **kwargs: Unpack[SignVideoKwargs]):
        super().__init__(**kwargs)

    @staticmethod
    def build_train_transform(kwargs: dict[str, Any]):
        transforms = [
            RandomVideoSpeed(speed_range=kwargs["random_speed_range"])
            if kwargs["do_random_speed"]
            else v2.Identity(),
            v2.RandomCrop((kwargs["crop_size"].height, kwargs["crop_size"].width)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize((kwargs["size"].height, kwargs["size"].width))
            if kwargs["do_resize"]
            else v2.Identity(),
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.5,
                    )
                ],
                p=0.75,
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(kwargs["image_mean"], kwargs["image_std"])
            if kwargs["do_normalize"]
            else v2.Identity(),
        ]
        return v2.Compose(transforms)

    @staticmethod
    def build_predict_transform(kwargs: dict[str, Any]):
        transforms = [
            v2.CenterCrop((kwargs["crop_size"].height, kwargs["crop_size"].width)),
            v2.Resize((kwargs["size"].height, kwargs["size"].width))
            if kwargs["do_resize"]
            else v2.Identity(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(kwargs["image_mean"], kwargs["image_std"])
            if kwargs["do_normalize"]
            else v2.Identity(),
        ]
        return v2.Compose(transforms)

    @staticmethod
    def pad_dim_to_multiple_of(tensor: torch.Tensor, dim: int, multiple: int):
        if multiple <= 0:
            raise ValueError("multiple must be greater than zero.")

        current_size = tensor.shape[dim]
        if current_size == 0:
            raise ValueError(
                "Cannot pad an empty dimension by repeating its last item."
            )
        remainder = current_size % multiple
        if remainder == 0:
            return tensor

        pad_size = multiple - remainder
        repeats = [1] * tensor.ndim
        repeats[dim] = pad_size
        padding = tensor.narrow(dim, current_size - 1, 1).repeat(repeats)
        return torch.cat((tensor, padding), dim=dim)

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        **kwargs: Unpack[SignVideoKwargs],
    ):
        training = kwargs["training"]
        process_fn = (
            self.build_train_transform(kwargs)
            if training
            else self.build_predict_transform(kwargs)
        )

        processed_videos = []
        video_lengths = []
        for video in videos:
            # BaseVideoProcessor has already converted each video to TCHW here.
            # Mark it explicitly so torchvision v2 applies video-aware kernels
            # and shares random spatial/color parameters across all frames.
            video = tv_tensors.Video(video)
            processed = process_fn(video).as_subclass(torch.Tensor)
            # Temporal augmentation must run before padding so it cannot sample
            # synthetic repeated frames. Record the final padded length.
            processed = self.pad_dim_to_multiple_of(
                processed,
                dim=0,
                multiple=kwargs["padding_to_multiple_of"],
            )
            video_lengths.append(processed.shape[0])
            processed_videos.append(processed)

        video_tensor = torch.cat(
            processed_videos,
            dim=0,
        ).contiguous()
        video_lengths_tensor = torch.tensor(
            video_lengths, dtype=torch.long, device=video_tensor.device
        ).contiguous()

        data = {
            "pixel_values": video_tensor,
            "pixel_values_lengths": video_lengths_tensor,
        }
        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)


if __name__ == "__main__":
    video_processor = SignVideoProcessor(
        padding_to_multiple_of=4,
        size={"height": 256, "width": 256},
        do_normalize=False,
    )
    print(video_processor)
