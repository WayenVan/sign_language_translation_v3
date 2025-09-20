import torch
import os

import albumentations as A


class ToTensorVideo:
    def __init__(self) -> None:
        pass

    def __call__(self, video):
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(
            0, 3, 1, 2
        )  # [time, height, width, channel] -> [time, channel, height, width]
        video = video.contiguous()
        return video


class UniGapSampleVideo:
    def __init__(self, gap=2):
        self.gap = gap

    def __call__(self, video):
        video = video[:: self.gap]
        return video


class UniformSampleVideo:
    def __init__(self, target_len=128):
        self.target_len = target_len

    def __call__(self, video):
        num_frames = video.shape[0]
        indices = self.uniform_sample(num_frames, self.target_len)
        video = video[indices]
        return video

        # 示例：采样视频到固定128帧

    @staticmethod
    def uniform_sample(num_frames, num_samples=128):
        """
        纯等间距采样（不抖动）
        :param num_frames: 视频总帧数
        :param num_samples: 要采样的帧数
        :return: 帧索引列表
        """
        if num_frames < num_samples:
            # 补齐策略：重复最后一帧
            indices = list(range(num_frames)) + [num_frames - 1] * (
                num_samples - num_frames
            )
            return indices

        interval = num_frames / num_samples
        indices = [int(interval * i + interval / 2) for i in range(num_samples)]
        return [min(idx, num_frames - 1) for idx in indices]


class JitteredUniformSampleVideo:
    def __init__(self, target_len=128, jitter_strength=0.5):
        """
        Uniformly samples video frames with jitter

        Args:
            target_len: Number of frames to sample
            jitter_strength: Strength of jitter (0.0-1.0) as a fraction of sampling interval
        """
        self.target_len = target_len
        self.jitter_strength = max(0.0, min(1.0, jitter_strength))  # Clamp between 0-1

    def __call__(self, video):
        num_frames = video.shape[0]

        if num_frames < self.target_len:
            # Pad with last frame when insufficient frames
            indices = list(range(num_frames)) + [num_frames - 1] * (
                self.target_len - num_frames
            )
        else:
            indices = self.jittered_sample(num_frames, self.target_len)

        return video

    def jittered_sample(self, num_frames, num_samples):
        """
        Uniform sampling with jitter applied

        Args:
            num_frames: Total frames in video
            num_samples: Number of frames to sample
        Returns:
            List of sampled frame indices
        """
        interval = num_frames / num_samples
        indices = []

        for i in range(num_samples):
            # Base position at center of segment
            base_pos = i * interval + interval / 2

            # Apply jitter within the segment
            jitter_range = interval * self.jitter_strength
            jitter = torch.rand(1).item() * jitter_range - jitter_range / 2
            pos = base_pos + jitter

            # Clamp position to valid range
            idx = min(max(0, int(round(pos))), num_frames - 1)
            indices.append(idx)

        return indices
