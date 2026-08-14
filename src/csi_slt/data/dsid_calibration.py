"""Text-only dataset view used while calibrating the D-SID threshold."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class DSIDCalibrationDataset(Dataset):
    """Expose only the fields required by the two frozen-teacher paths.

    Datasets may implement ``get_text_item`` to avoid video I/O. The regular
    ``__getitem__`` path remains a compatibility fallback for other datasets.
    """

    required_fields = ("id", "text", "lang", "pseudo_gloss")

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        get_text_item = getattr(self.dataset, "get_text_item", None)
        item = get_text_item(index) if callable(get_text_item) else self.dataset[index]
        missing = [field for field in self.required_fields if field not in item]
        if missing:
            raise KeyError(
                "D-SID calibration sample is missing required fields: "
                + ", ".join(missing)
            )
        return {field: item[field] for field in self.required_fields}
