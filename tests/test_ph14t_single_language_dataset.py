from unittest.mock import patch

import pytest
from datasets import Dataset

from csi_slt.data.ph14t import Ph14TSingleLanguageDataset


def _multilingual_dataset():
    return Dataset.from_dict(
        {
            "name": ["de-1", "en-1", "zh-1", "en-2"],
            "translation": ["de", "en", "zh", "en again"],
            "lang": ["de", "en", "zh", "en"],
            "orig_pseudo_gloss_strict": ["D", "E", "Z", "E2"],
            "frames": [["de.png"], ["en.png"], ["zh.png"], ["en2.png"]],
        }
    )


def test_single_language_dataset_filters_rows_and_preserves_order():
    with patch(
        "csi_slt.data.ph14t.ph14t_torch_dataset_multiling.load_dataset",
        return_value=_multilingual_dataset(),
    ):
        dataset = Ph14TSingleLanguageDataset("frames", language="en")

    assert len(dataset) == 2
    assert dataset.hg_dataset["name"] == ["en-1", "en-2"]
    assert dataset.hg_dataset["lang"] == ["en", "en"]
    assert dataset.cache_namespace == "ph14t_single_language/en"


def test_single_language_dataset_rejects_unknown_language():
    with pytest.raises(ValueError, match="language must be one of"):
        Ph14TSingleLanguageDataset("frames", language="fr")


def test_single_language_dataset_rejects_empty_selection():
    source = _multilingual_dataset().filter(lambda row: row["lang"] != "zh")
    with patch(
        "csi_slt.data.ph14t.ph14t_torch_dataset_multiling.load_dataset",
        return_value=source,
    ):
        with pytest.raises(ValueError, match="No 'zh' samples"):
            Ph14TSingleLanguageDataset("frames", language="zh")
