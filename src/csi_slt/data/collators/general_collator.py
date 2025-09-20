import torch
import numpy as np
from typing import List
from torch.nn import functional as F


class GeneralSLTCollator:
    """
    A general collator class for SLT (Speech and Language Tasks).
    This class is designed to handle the collation of data samples
    into batches for training or evaluation.
    """

    def collate_video(self, videos: list):
        """
        Collate function to handle padding of video sequences.
        Supports both numpy arrays (keeps type) and torch tensors.
        """
        v_length = [video.shape[0] for video in videos]
        max_v_length = max(v_length)

        # Check if all inputs are numpy or all are torch
        all_numpy = all(isinstance(v, np.ndarray) for v in videos)
        all_torch = all(isinstance(v, torch.Tensor) for v in videos)

        if not (all_numpy or all_torch):
            raise ValueError("All videos must be either numpy arrays or torch tensors")

        if all_numpy:
            padded_videos = [
                np.pad(
                    video,
                    ((0, max_v_length - video.shape[0]), (0, 0), (0, 0), (0, 0)),
                    mode="constant",
                )
                for video in videos
            ]
            return np.stack(padded_videos), np.array(v_length, dtype=np.int64)
        else:
            padded_videos = [
                F.pad(video, (0, 0, 0, 0, 0, 0, 0, max_v_length - video.shape[0]))
                for video in videos
            ]
            return torch.stack(padded_videos), torch.tensor(v_length).long()

    def __call__(self, batch):
        """
        Collate a batch of samples.

        Args:
            batch (list): A list of samples to collate.

        Returns:
            dict: A dictionary containing the collated inputs and labels.
        """

        ret = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        keys_to_process = [key for key in ret if key in ("video", "augmented_video")]

        for key in keys_to_process:
            if key == "video":
                # Handle video collating
                ret[key], ret["video_length"] = self.collate_video(ret[key])
            if key == "augmented_video":
                # Handle augmented video collating
                ret[key], ret["augmented_video_length"] = self.collate_video(ret[key])
        return ret
