import json

import pytest

from csi_slt.engine.prompt_sampler import PromptSampler


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompt(prompt_id, target_lang):
    return {
        "id": prompt_id,
        "target_lang": target_lang,
        "template": f"{prompt_id}: {{{{ video_start_token }}}}",
    }


def test_random_draws_are_reproducible_but_can_vary_per_call(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_prompt(f"de_{index}", "de") for index in range(16)])
    first = PromptSampler(path, seed=7)
    second = PromptSampler(path, seed=7)

    first_sequence = [first.random("de").id for _ in range(20)]
    second_sequence = [second.random("de").id for _ in range(20)]

    assert first_sequence == second_sequence
    assert len(set(first_sequence)) > 1

    first.set_epoch(3)
    epoch_sequence = [first.random("de").id for _ in range(10)]
    first.set_epoch(3)
    assert [first.random("de").id for _ in range(10)] == epoch_sequence


def test_random_stays_inside_target_language_group(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [_prompt("de_1", "de"), _prompt("de_2", "de"), _prompt("en_1", "en")],
    )
    sampler = PromptSampler(path, seed=3)

    assert sampler.random("de").target_lang == "de"
    assert sampler.random("en").id == "en_1"


def test_by_id_resolves_and_validates_target_language(tmp_path):
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(path, [_prompt("canonical_de", "de"), _prompt("heldout_en", "en")])
    sampler = PromptSampler(path)

    assert sampler.by_id("canonical_de", target_lang="de").id == "canonical_de"
    with pytest.raises(KeyError, match="Unknown prompt id"):
        sampler.by_id("missing")
    with pytest.raises(ValueError, match="sample targets 'en'"):
        sampler.by_id("canonical_de", target_lang="en")


def test_loads_multiple_non_overlapping_banks(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_prompt("de_1", "de")])
    _write_jsonl(second, [_prompt("en_1", "en")])

    sampler = PromptSampler([first, second])

    assert {record.id for record in sampler.records} == {"de_1", "en_1"}


def test_rejects_duplicate_ids_across_banks(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_prompt("duplicate", "de")])
    _write_jsonl(second, [_prompt("duplicate", "en")])

    with pytest.raises(ValueError, match="Duplicate prompt id"):
        PromptSampler([first, second])


def test_rejects_unused_fields_and_bad_sentinel(tmp_path):
    extra = tmp_path / "extra.jsonl"
    row = _prompt("extra", "de")
    row["split"] = "train"
    _write_jsonl(extra, [row])
    with pytest.raises(ValueError, match="unsupported fields"):
        PromptSampler(extra)

    bad = tmp_path / "bad.jsonl"
    row = _prompt("bad", "de")
    row["template"] = "missing sentinel"
    _write_jsonl(bad, [row])
    with pytest.raises(ValueError, match="exactly one"):
        PromptSampler(bad)


def test_real_prompt_banks_load():
    train = PromptSampler("prompts/train.jsonl")
    heldout = PromptSampler("prompts/heldout.jsonl")
    wrong_task = PromptSampler("prompts/wrong_task.jsonl")
    unrelated = PromptSampler("prompts/unrelated.jsonl")

    assert len(train.records) == 24
    assert len(heldout.records) == 12
    assert len(wrong_task.records) == 3
    assert len(unrelated.records) == 3
    assert train.by_id("canonical_en_de_001", target_lang="de")
