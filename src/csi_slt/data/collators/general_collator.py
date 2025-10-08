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

    def __init__(self, processor, debug=False):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.debug = debug

    def __call__(self, batch):
        """
        Collate a batch of samples.

        Args:
            batch (list): A list of samples to collate.

        Returns:
            dict: A dictionary containing the collated inputs and labels.
        """

        # Collate a batch of samples.
        zbatch = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        # Unpack batch data
        names, videos, texts, lang = (
            zbatch["id"],
            zbatch["video"],
            zbatch["text"],
            zbatch["lang"],
        )
        batch_processed = self.processor(videos=videos, text=texts, src_lang=lang)

        input_text = None
        label_text = None
        if self.debug:
            input_text = self.tokenizer.batch_decode(
                batch_processed["input_ids"], skip_special_tokens=False
            )
            label_ids = batch_processed["labels"]
            label_ids = torch.where(
                label_ids == -100,
                torch.full_like(label_ids, self.tokenizer.pad_token_id).to(
                    label_ids.device
                ),
                label_ids,
            )
            label_text = self.tokenizer.batch_decode(
                label_ids,
                skip_special_tokens=False,
            )

        return {
            **batch_processed.data,
            "names": names,
            "lang": lang,
            "original_videos": videos,
            "input_text": input_text,
            "label_text": label_text,
        }
