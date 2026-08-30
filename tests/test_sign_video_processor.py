from types import SimpleNamespace

from torchvision.transforms import v2

from csi_slt.data.processors.sign_video_processor import SignVideoProcessor


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
