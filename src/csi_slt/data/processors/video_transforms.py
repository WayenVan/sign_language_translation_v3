from __future__ import annotations

from typing import Any, Sequence

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


def _is_pure_tensor(inpt: Any) -> bool:
    """只匹配普通 torch.Tensor，不匹配 Image、Video 等 TVTensor 子类。"""
    return type(inpt) is torch.Tensor


class RandomVideoSpeed(v2.Transform):
    """
    通过时间轴帧索引重采样，实现随机视频变速。

    支持输入形状：
        (T, C, H, W)
        (..., T, C, H, W)

    支持输入类型：
        - torch.Tensor
        - torchvision.tv_tensors.Video

    Args:
        speed_range:
            随机速度范围。

            speed > 1:
                加速，输出帧数减少。

            speed < 1:
                减速，输出帧数增加，部分帧会重复。

        p:
            应用变速增强的概率。

    Example:
        RandomVideoSpeed((0.8, 1.25), p=0.5)
    """

    # 只处理明确标记的 Video，或者单独传入的普通 Tensor。
    # 不会把 tv_tensors.Image 当作视频处理。
    _transformed_types = (
        tv_tensors.Video,
        _is_pure_tensor,
    )

    def __init__(
        self,
        speed_range: Sequence[float] = (0.8, 1.25),
        p: float = 1.0,
    ) -> None:
        super().__init__()

        if len(speed_range) != 2:
            raise ValueError(
                f"`speed_range` must contain exactly two values, but got {speed_range}."
            )

        min_speed, max_speed = map(float, speed_range)

        if min_speed <= 0 or max_speed <= 0:
            raise ValueError(
                f"All speed values must be greater than 0, but got {speed_range}."
            )

        if min_speed > max_speed:
            raise ValueError(
                f"`speed_range[0]` must be <= `speed_range[1]`, but got {speed_range}."
            )

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"`p` must be in [0, 1], but got {p}.")

        self.speed_range = (min_speed, max_speed)
        self.p = float(p)

    def make_params(
        self,
        flat_inputs: list[Any],
    ) -> dict[str, Any]:
        apply_transform = bool(torch.rand(()) < self.p)

        if apply_transform:
            speed = float(
                torch.empty(()).uniform_(
                    self.speed_range[0],
                    self.speed_range[1],
                )
            )
        else:
            speed = 1.0

        return {
            "apply_transform": apply_transform,
            "speed": speed,
        }

    def transform(
        self,
        inpt: torch.Tensor,
        params: dict[str, Any],
    ) -> torch.Tensor:
        if not params["apply_transform"]:
            return inpt

        if inpt.ndim < 4:
            raise ValueError(
                "Expected a video with shape (T, C, H, W) or "
                f"(..., T, C, H, W), but got {tuple(inpt.shape)}."
            )

        # torchvision Video 的约定是 (..., T, C, H, W)，
        # 因此时间维始终是倒数第 4 维。
        time_dim = inpt.ndim - 4
        num_frames = inpt.shape[time_dim]

        if num_frames == 0:
            raise ValueError("The input video contains no frames.")

        speed = params["speed"]

        # speed > 1：输出更短
        # speed < 1：输出更长
        new_num_frames = max(
            1,
            round(num_frames / speed),
        )

        # 在完整原始时间轴上均匀取样。
        # round() 相当于最近邻时间插值：
        # - 加速时跳过帧
        # - 减速时重复帧
        indices = torch.linspace(
            0,
            num_frames - 1,
            steps=new_num_frames,
            device=inpt.device,
            dtype=torch.float32,
        )
        indices = indices.round().long()

        output = torch.index_select(
            inpt,
            dim=time_dim,
            index=indices,
        )

        # index_select 默认返回普通 Tensor，
        # 因此恢复 tv_tensors.Video 类型。
        if isinstance(inpt, tv_tensors.Video):
            output = tv_tensors.wrap(output, like=inpt)

        return output


class BottomCenterCrop(v2.Transform):
    """
    底边对齐、水平居中的确定性裁剪。

    手语者的手会从画面底边出去，而不是顶边：居中裁剪会切掉底部
    最常打手语的区域，却保留头顶上方的空墙。这个裁剪把损失挪到顶部。

    输入比裁剪尺寸小时直接报错，而不像 CenterCrop 那样补边：
    从底边往上补边没有合理的语义。

    Args:
        size:
            输出尺寸 (height, width)。

    Example:
        BottomCenterCrop((224, 224))
    """

    _transformed_types = (
        tv_tensors.Video,
        _is_pure_tensor,
    )

    def __init__(self, size: Sequence[int]) -> None:
        super().__init__()

        if len(size) != 2:
            raise ValueError(
                f"`size` must contain exactly two values, but got {size}."
            )

        self.height, self.width = int(size[0]), int(size[1])

    def transform(
        self,
        inpt: torch.Tensor,
        params: dict[str, Any],
    ) -> torch.Tensor:
        in_height, in_width = inpt.shape[-2:]
        if in_height < self.height or in_width < self.width:
            raise ValueError(
                f"Crop {self.height}x{self.width} is larger than the input "
                f"{in_height}x{in_width}."
            )

        # functional.crop 会保留 tv_tensors.Video 类型。
        return v2.functional.crop(
            inpt,
            top=in_height - self.height,
            left=(in_width - self.width) // 2,
            height=self.height,
            width=self.width,
        )
