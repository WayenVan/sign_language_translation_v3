import numpy as np

from nlpaug.augmenter.word.random import RandomWordAug
import polars as pl
import logging

logger = logging.getLogger(__name__)


class SaveOriginalText:
    """
    Save the original text in the data dictionary.
    """

    def __call__(self, data: dict) -> dict:
        data["original_text"] = data["text"]
        return data


class ExtendedPh14TTextAugmentation:
    """
    Extended PH14T Text Augmentation with extended text data by DeekSeek API.
    """

    def __init__(self, extend_csv_dir, random_choice=True):
        self.extend_csv_dir = extend_csv_dir
        self.extended_df = pl.read_csv(
            self.extend_csv_dir,
            separator="|",
        ).select(["id", "text"])
        self.random_choice = random_choice

    def __call__(self, data: dict) -> dict:
        original_text = data["text"]

        extended_texts = self.extended_df.filter(self.extended_df["id"] == data["id"])[
            "text"
        ].to_list()

        if len(extended_texts) == 0:
            logger.warning(
                f"Extended text for id {data['id']} not found in {self.extend_csv_dir}"
            )
            selected_text = original_text
        elif self.random_choice:
            selected_text = np.random.choice(extended_texts)
        else:
            selected_text = original_text

        data["text"] = selected_text
        data["original_text"] = original_text
        data["extended_texts"] = extended_texts
        return data


if __name__ == "__main__":
    # Example usage
    augmenter = RandomWordAugmentation(action="swap", aug_p=0.3)
    data = {"text": "This is a sample text for augmentation."}
    augmented_data = augmenter(data)
    print("Original Text:", data["text"])
    print("Augmented Text:", augmented_data["text"])
