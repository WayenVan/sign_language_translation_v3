import numpy as np
import torch
import random


class MBARTCollator:
    """
    collator for my mbart model, which changes lots of things from the original gfslt collator

    """

    def __init__(
        self,
        tokenizer,
        phase="train",
        training_refurbish=True,
        noise_rate=0.15,
        noise_type="omit_last",
        random_shuffle=False,
    ):
        self.tokenizer = tokenizer
        self.phase = phase
        self.training_refurbish = training_refurbish
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.random_shuffle = random_shuffle

        # Define a mask token for text noise injection
        self.WORD_MASK = getattr(self.tokenizer, "mask_token", None)
        if self.WORD_MASK is None:
            raise ValueError(
                "Tokenizer does not have a mask token defined. "
                "Please set the mask_token attribute in the tokenizer."
            )

    @staticmethod
    def pad_dim_to_multiple_of_4(tensor, dim):
        current_size = tensor.size(dim)
        remainder = current_size % 4
        if remainder == 0:
            return tensor

        pad_size = 4 - remainder

        # 取这个维度的最后一个元素
        index = [slice(None)] * tensor.dim()
        index[dim] = -1
        last_element = tensor[tuple(index)].unsqueeze(dim)

        # 复制 pad_size 次
        padding = last_element.repeat_interleave(pad_size, dim=dim)
        return torch.cat([tensor, padding], dim=dim).contiguous()

    def __call__(self, batch):
        # Collate a batch of samples.
        zbatch = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        # Unpack batch data
        names, videos, texts = (
            zbatch["id"],
            zbatch["augmented_video"],
            zbatch["text"],
        )

        # Stack all videos into single tensor
        # video (T, C, H, W) ...
        #
        videos = [self.pad_dim_to_multiple_of_4(video, dim=0) for video in videos]
        video_lengths = [video.size(0) for video in videos]

        video_tensor = torch.cat(videos, dim=0).contiguous()
        video_lengths_tensor = torch.tensor(video_lengths)

        # Tokenize text
        # the taget sequence will be shifted inside the text decoder, the start toeken is padding id
        text_src_input = self.tokenizer(texts, return_tensors="pt", padding=True)

        # Prepare source input
        video_input = {
            "video": video_tensor,
            "names": names,
            "video_length": video_lengths_tensor,
        }

        # Handle training refurbish if needed
        if self.training_refurbish:
            masked_texts = self.noise_injecting(
                texts,
                self.noise_rate,
                noise_type=self.noise_type,
                random_shuffle=self.random_shuffle,
                is_train=(self.phase == "train"),
            )
            masked_text_src_input = self.tokenizer(
                masked_texts, return_tensors="pt", padding=True
            )
            return video_input, text_src_input, masked_text_src_input

        # text [<language_id>, xx...... </s>]
        return video_input, text_src_input, None

    def noise_injecting(
        self,
        raw_gloss,
        noise_rate=0.15,
        noise_type="omit_last",
        random_shuffle=False,
        is_train=True,
    ):
        new_gloss = []

        for ii, gloss in enumerate(raw_gloss):
            text = gloss.split()

            if noise_type == "omit":
                # del noise
                if random.uniform(0, 1) <= 1.0 and is_train:
                    index = sampler_func(
                        len(text),
                        int(len(text) * (1.0 - noise_rate)),
                        random_choice=is_train,
                    )
                    noise_gloss = []
                    noise_idx = []
                    for i, d in enumerate(text):
                        if i in index:
                            noise_gloss.append(d)
                        else:
                            noise_gloss.append(self.WORD_MASK)
                            noise_idx.append(i)
                else:
                    noise_gloss = [d for d in text]

            elif noise_type == "omit_last":
                if random.uniform(0, 1) <= 1.0 and is_train:
                    index = np.arange(
                        0,
                        len(text)
                        - int(
                            np.ceil(
                                len(text) * (np.random.uniform(0, noise_rate, (1,)))
                            )
                        ),
                        1,
                        dtype=int,
                    )
                    noise_gloss = []
                    for i, d in enumerate(text):
                        if i in index:
                            noise_gloss.append(d)
                        else:
                            noise_gloss.append(self.WORD_MASK)
                else:
                    noise_gloss = [d for d in text]

            if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
                random.shuffle(noise_gloss)  # random shuffle sequence

            new_gloss.append(" ".join(noise_gloss))
        return new_gloss


def sampler_func(clip, sn, random_choice=True):
    """Sample indices from a sequence of length clip into sn segments.

    Args:
        clip: Total length of the sequence
        sn: Number of segments to sample
        random_choice: Whether to choose randomly within segments
    Returns:
        List of sampled indices
    """
    result = []
    for i in range(sn):
        # Calculate segment start and end indices
        start = int(clip * i / sn)
        end = int(clip * (i + 1) / sn)

        # Handle case where segment has no elements
        if start == end:
            result.append(start)
            continue

        # Generate indices within the segment
        segment_indices = list(range(start, end))

        if random_choice:
            # Randomly select one index from segment
            result.append(random.choice(segment_indices))
        else:
            # Select the middle index of the segment
            result.append(segment_indices[len(segment_indices) // 2])

    return result
