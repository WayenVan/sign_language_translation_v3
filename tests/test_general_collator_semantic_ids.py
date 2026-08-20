from types import SimpleNamespace

from csi_slt.data.collators.general_collator import GeneralSLTCollator


class _RecordingProcessor:
    tokenizer = None

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(data={"semantic_ids": kwargs["semantic_ids"]})


def test_collator_forwards_plural_semantic_ids():
    processor = _RecordingProcessor()
    collator = GeneralSLTCollator(processor=processor)
    batch = [
        {
            "id": "sample-a",
            "semantic_ids": "meaning-1",
            "video": "video-a",
            "text": "text-a",
            "lang": "en",
            "pseudo_gloss": "GLOSS-A",
        },
        {
            "id": "sample-b",
            "semantic_ids": "meaning-1",
            "video": "video-b",
            "text": "text-b",
            "lang": "en",
            "pseudo_gloss": "GLOSS-B",
        },
    ]

    output = collator(batch)

    assert processor.kwargs["semantic_ids"] == ("meaning-1", "meaning-1")
    assert output["semantic_ids"] == ("meaning-1", "meaning-1")
    assert "original_videos" not in output
