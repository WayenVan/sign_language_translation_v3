import json

import pytest

from csi_slt.engine.prompt_resolver import (
    FixedPromptResolver,
    PromptIdFromRowResolver,
    RandomPromptResolver,
)
from csi_slt.engine.prompt_sampler import PromptSampler


def _sampler(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [
        {
            "id": "de_1",
            "target_lang": "de",
            "template": "German one: {{ video_start_token }}",
        },
        {
            "id": "de_2",
            "target_lang": "de",
            "template": "German two: {{ video_start_token }}",
        },
        {
            "id": "en_1",
            "target_lang": "en",
            "template": "English: {{ video_start_token }}",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return PromptSampler(path, seed=7)


def test_random_resolver_uses_row_language_and_epoch(tmp_path):
    resolver = RandomPromptResolver(_sampler(tmp_path))
    row = {"lang": "de"}

    assert resolver.resolve(row, epoch=3).target_lang == "de"
    assert resolver.sampler.epoch == 3


def test_fixed_resolver_validates_mapping_and_resolves_by_language(tmp_path):
    sampler = _sampler(tmp_path)
    resolver = FixedPromptResolver(sampler, {"de": "de_1", "en": "en_1"})

    assert resolver.resolve({"lang": "de"}).id == "de_1"
    with pytest.raises(KeyError, match="No fixed prompt configured"):
        resolver.resolve({"lang": "zh"})
    with pytest.raises(ValueError, match="sample targets 'en'"):
        FixedPromptResolver(sampler, {"en": "de_1"})


def test_prompt_id_from_row_resolver_uses_expanded_row_prompt_id(tmp_path):
    resolver = PromptIdFromRowResolver(_sampler(tmp_path))

    assert resolver.resolve({"lang": "de", "prompt_id": "de_2"}).id == "de_2"
    with pytest.raises(ValueError, match="sample targets 'en'"):
        resolver.resolve({"lang": "en", "prompt_id": "de_2"})


def test_resolvers_report_missing_columns(tmp_path):
    sampler = _sampler(tmp_path)

    with pytest.raises(KeyError, match="missing required column 'lang'"):
        RandomPromptResolver(sampler).resolve({})
    with pytest.raises(KeyError, match="missing required column 'prompt_id'"):
        PromptIdFromRowResolver(sampler).resolve({"lang": "de"})
