from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from csi_slt.data.processors.sign_video_processor import SignVideoProcessor
from csi_slt.data.processors.video_transforms import BottomCenterCrop


def _train_transform_kwargs(*, do_random_resize: bool):
    return {
        "random_speed_range": (0.8, 1.25),
        "do_random_speed": False,
        "do_random_resize": do_random_resize,
        "crop_size": SimpleNamespace(height=224, width=224),
        "size": SimpleNamespace(height=224, width=224),
        "do_resize": False,
        "do_random_gaussian_blur": False,
        "do_random_erasing": False,
        "do_normalize": False,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }


def test_random_resize_is_enabled_by_default():
    processor = SignVideoProcessor()

    assert processor.do_random_resize is True
    assert processor.to_dict()["do_random_resize"] is True


def test_train_transform_uses_random_resized_crop_when_enabled():
    transform = SignVideoProcessor.build_train_transform(
        _train_transform_kwargs(do_random_resize=True)
    )

    assert isinstance(transform.transforms[1], v2.RandomResizedCrop)


def test_train_transform_uses_random_crop_when_random_resize_is_disabled():
    transform = SignVideoProcessor.build_train_transform(
        _train_transform_kwargs(do_random_resize=False)
    )

    assert isinstance(transform.transforms[1], v2.RandomCrop)
    assert not isinstance(transform.transforms[1], v2.RandomResizedCrop)


def _predict_transform_kwargs(*, eval_crop_bottom_aligned: bool):
    return {
        "crop_size": SimpleNamespace(height=224, width=224),
        "size": SimpleNamespace(height=224, width=224),
        "do_resize": False,
        "do_normalize": False,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "eval_crop_bottom_aligned": eval_crop_bottom_aligned,
    }


def _ramp_video(num_frames=3, height=256, width=256):
    """THWC uint8 whose pixel values encode their own row and column."""
    rows = np.arange(height, dtype=np.uint8)[:, None].repeat(width, axis=1)
    cols = np.arange(width, dtype=np.uint8)[None, :].repeat(height, axis=0)
    frame = np.stack([rows, cols, rows // 2 + cols // 2], axis=-1)
    return np.stack([frame] * num_frames)


def test_eval_crop_bottom_aligned_is_disabled_by_default():
    processor = SignVideoProcessor()

    assert processor.eval_crop_bottom_aligned is False
    assert processor.to_dict()["eval_crop_bottom_aligned"] is False


def test_predict_transform_uses_center_crop_by_default():
    # Every PHOENIX-2014-T config and checkpoint predates the flag, so the
    # default must keep evaluating exactly as before.
    transform = SignVideoProcessor.build_predict_transform(
        _predict_transform_kwargs(eval_crop_bottom_aligned=False)
    )

    assert isinstance(transform.transforms[0], v2.CenterCrop)


def test_predict_transform_uses_bottom_center_crop_when_enabled():
    transform = SignVideoProcessor.build_predict_transform(
        _predict_transform_kwargs(eval_crop_bottom_aligned=True)
    )

    assert isinstance(transform.transforms[0], BottomCenterCrop)


def test_bottom_center_crop_keeps_the_bottom_rows_and_centred_columns():
    video = tv_tensors.Video(torch.arange(2 * 3 * 10 * 12).reshape(2, 3, 10, 12))

    cropped = BottomCenterCrop((6, 8))(video)

    assert isinstance(cropped, tv_tensors.Video)
    assert torch.equal(cropped, video[..., 4:10, 2:10])


def test_bottom_center_crop_rejects_inputs_smaller_than_the_crop():
    video = tv_tensors.Video(torch.zeros(1, 3, 100, 100))

    with pytest.raises(ValueError, match="larger than the input"):
        BottomCenterCrop((224, 224))(video)


@pytest.mark.parametrize(
    ("eval_crop_bottom_aligned", "top"),
    [(False, 16), (True, 32)],
)
def test_processor_eval_crop_through_the_call_path(eval_crop_bottom_aligned, top):
    video = _ramp_video()
    processor = SignVideoProcessor(
        do_normalize=False,
        eval_crop_bottom_aligned=eval_crop_bottom_aligned,
    )

    pixel_values = processor([video], training=False).pixel_values

    expected = torch.from_numpy(video[:, top : top + 224, 16:240]).permute(0, 3, 1, 2)
    expected = torch.cat([expected, expected[-1:]])  # padded to a multiple of 4
    assert torch.allclose(pixel_values, expected.float() / 255)
