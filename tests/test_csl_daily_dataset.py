from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from csi_slt.data.csl import CSLDailyDataset


CLIPS = {
    "train": [("S000001_P0000_T00", 3), ("S000001_P0004_T00", 2)],
    "dev": [("S000002_P0001_T00", 4)],
    "test": [("S000003_P0002_T00", 1)],
}


def _write_frames_root(root, *, complete=True):
    """A miniature extract_frames.py output: JPEG frames plus split indexes.

    Each frame is a flat colour whose red channel is the frame's position, so
    a test can tell both frame order and RGB/BGR order from the pixels.
    """
    for split, clips in CLIPS.items():
        rows = []
        for clip_id, num_frames in clips:
            clip_dir = root / split / clip_id
            clip_dir.mkdir(parents=True)
            frames = []
            for position in range(num_frames):
                name = f"images{position + 20:04d}.jpg"
                rgb = np.zeros((8, 8, 3), dtype=np.uint8)
                rgb[..., 0] = 40 * position + 20
                rgb[..., 2] = 200
                cv2.imwrite(str(clip_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frames.append(f"{split}/{clip_id}/{name}")
            rows.append(
                dict(
                    clip_id=clip_id,
                    signer=clip_id.split("_")[1],
                    translation=f"{clip_id} 的翻译",
                    gloss=f"{clip_id} 手语",
                    pseudo_gloss_relaxed=f"{clip_id} 伪词",
                    start_frame=19,
                    end_frame_exclusive=19 + num_frames,
                    num_frames=num_frames,
                    frames=frames,
                )
            )
        pd.DataFrame(rows).to_parquet(root / f"{split}.parquet", index=False)
    if complete:
        (root / ".complete").touch()
    return root


class _CharTokenizer:
    """Stands in for a Hugging Face tokenizer: one token per character."""

    eos_token = "</s>"

    def __call__(self, texts, **kwargs):
        return {"input_ids": [list(range(len(text))) for text in texts]}


@pytest.fixture
def frames_root(tmp_path):
    return _write_frames_root(tmp_path / "full-trim-256x256px")


def test_getitem_returns_the_fields_the_collator_reads(frames_root):
    dataset = CSLDailyDataset(str(frames_root), mode="train")

    item = dataset[0]

    assert len(dataset) == 2
    assert item["id"] == "S000001_P0000_T00"
    assert item["text"] == "S000001_P0000_T00 的翻译"
    assert item["lang"] == "zh"
    assert item["pseudo_gloss"] is None
    assert item["video"].dtype == np.uint8
    assert item["video"].shape == (3, 8, 8, 3)


def test_getitem_returns_the_selected_pseudo_gloss(frames_root):
    dataset = CSLDailyDataset(
        str(frames_root),
        mode="train",
        pseudo_gloss_column="pseudo_gloss_relaxed",
    )

    assert dataset[0]["pseudo_gloss"] == "S000001_P0000_T00 伪词"


def test_rejects_a_missing_pseudo_gloss_column(frames_root):
    with pytest.raises(ValueError, match="pseudo-gloss column.*is missing"):
        CSLDailyDataset(
            str(frames_root),
            mode="train",
            pseudo_gloss_column="missing_gloss",
        )


def test_frames_are_rgb_and_in_index_order(frames_root):
    video = CSLDailyDataset(str(frames_root), mode="train")[0]["video"]

    # JPEG is lossy, so compare against the encoded colours with a tolerance.
    red = video[..., 0].mean(axis=(1, 2))
    blue = video[..., 2].mean(axis=(1, 2))
    np.testing.assert_allclose(red, [20, 60, 100], atol=4)
    np.testing.assert_allclose(blue, [200, 200, 200], atol=4)


def test_takes_of_one_sentence_share_semantic_ids(frames_root):
    dataset = CSLDailyDataset(str(frames_root), mode="train")

    assert dataset[0]["semantic_ids"] == dataset[1]["semantic_ids"] == "S000001"
    assert dataset[0]["id"] != dataset[1]["id"]


@pytest.mark.parametrize("mode", ["validation", "dev"])
def test_validation_reads_the_dev_split(frames_root, mode):
    dataset = CSLDailyDataset(str(frames_root), mode=mode)

    assert dataset.hg_dataset["clip_id"] == ["S000002_P0001_T00"]


def test_get_text_item_does_not_open_frames(frames_root):
    dataset = CSLDailyDataset(str(frames_root), mode="test")

    with patch.object(CSLDailyDataset, "_read_frame", side_effect=AssertionError):
        item = dataset.get_text_item(0)

    assert item["id"] == "S000003_P0002_T00"
    assert "video" not in item


def test_prepare_records_lengths_for_the_bucketing_sampler(frames_root, tmp_path):
    dataset = CSLDailyDataset.create_prepared_dataset(
        _CharTokenizer(),
        str(frames_root),
        mode="train",
        cache_dir=tmp_path / "cache",
    )

    assert dataset.video_lengths == [3, 2]
    label = "S000001_P0000_T00 的翻译</s>"
    assert dataset.label_ids_lengths == [len(label), len(label)]


def test_rejects_an_unfinished_extraction(tmp_path):
    root = _write_frames_root(tmp_path / "partial", complete=False)

    with pytest.raises(FileNotFoundError, match=".complete"):
        CSLDailyDataset(str(root), mode="train")


def test_rejects_unknown_mode(frames_root):
    with pytest.raises(ValueError, match="mode must be one of"):
        CSLDailyDataset(str(frames_root), mode="val")
