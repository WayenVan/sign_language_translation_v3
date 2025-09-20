import numpy as np
import torch
import random


class GLFSLTCollator:
    def __init__(
        self,
        tokenizer,
        phase="train",
        noise_rate=0.15,
        noise_type="omit_last",
        random_shuffle=False,
        training_refurbish=False,
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

    def __call__(self, batch):
        # Collate a batch of samples.
        zbatch = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        # Unpack batch data
        names, videos, texts = (
            zbatch["id"],
            zbatch["augmented_video"],
            zbatch["text"],
        )

        # Process videos
        video_lengths = [len(vid) for vid in videos]
        max_len = max(video_lengths)

        # Calculate padding parameters
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        padded_length = max_len + left_pad + right_pad

        # Create padded video tensor
        padded_videos = []
        for vid, orig_len in zip(videos, video_lengths):
            # Create padding frames
            first_frame = vid[0].unsqueeze(0).expand(left_pad, -1, -1, -1)
            last_frame = (
                vid[-1]
                .unsqueeze(0)
                .expand(padded_length - orig_len - left_pad, -1, -1, -1)
            )

            # Combine with original video
            padded = torch.cat([first_frame, vid, last_frame], dim=0)
            padded_videos.append(padded[:padded_length])  # Ensure uniform length

        # Stack all videos into single tensor
        video_tensor = torch.stack(padded_videos)
        video_lengths_tensor = torch.tensor(video_lengths) + left_pad + right_pad

        # Generate attention mask
        new_lengths = (((video_lengths_tensor - 5 + 1) / 2) - 5 + 1) / 2
        new_lengths = new_lengths.long()
        max_new_len = new_lengths.max().item()
        mask_gen = torch.zeros(len(batch), max_new_len, dtype=torch.long)
        for i, length in enumerate(new_lengths):
            mask_gen[i, :length] = 1
        img_padding_mask = mask_gen

        # Tokenize text
        with self.tokenizer.as_target_tokenizer():
            # the taget sequence will be shifted inside the text decoder, the start toeken is padding id
            tgt_input = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )

        # Prepare source input
        src_input = {
            "input_ids": video_tensor,
            "attention_mask": img_padding_mask,
            "name_batch": names,
            "src_length_batch": video_lengths_tensor,
            "new_src_length_batch": new_lengths,
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
            with self.tokenizer.as_target_tokenizer():
                masked_tgt_input = self.tokenizer(
                    masked_texts, return_tensors="pt", padding=True, truncation=True
                )
            return src_input, tgt_input, masked_tgt_input

        return src_input, tgt_input

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
