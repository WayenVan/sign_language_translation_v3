from types import SimpleNamespace

from csi_slt.data.collators.general_collator import GeneralSLTCollator
from csi_slt.engine.prompt_sampler import PromptRecord


class _RecordingProcessor:
    tokenizer = None

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(data={"semantic_ids": kwargs["semantic_ids"]})


class _RecordingResolver:
    def __init__(self):
        self.calls = []

    def resolve(self, row, *, epoch=None):
        self.calls.append((row, epoch))
        return PromptRecord(
            id=f"prompt-{row['id']}",
            target_lang=row["lang"],
            template=f"Instruction for {row['id']}: {{{{ video_start_token }}}}",
        )


def test_collator_forwards_plural_semantic_ids():
    processor = _RecordingProcessor()
    collator = GeneralSLTCollator(
        processor=processor,
        prompt_resolver=_RecordingResolver(),
        training=True,
    )
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


def test_collator_resolves_and_forwards_per_sample_prompt_templates():
    processor = _RecordingProcessor()
    resolver = _RecordingResolver()
    collator = GeneralSLTCollator(
        processor=processor,
        prompt_resolver=resolver,
        training=True,
    )
    collator.set_epoch(4)
    batch = [
        {
            "id": "sample-a",
            "video": "video-a",
            "text": "text-a",
            "lang": "de",
        },
        {
            "id": "sample-b",
            "video": "video-b",
            "text": "text-b",
            "lang": "en",
        },
    ]

    collator(batch)

    assert [epoch for _, epoch in resolver.calls] == [4, 4]
    assert processor.kwargs["prompt_templates"] == (
        "Instruction for sample-a: {{ video_start_token }}",
        "Instruction for sample-b: {{ video_start_token }}",
    )
