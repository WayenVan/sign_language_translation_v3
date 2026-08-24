import json

import pytest

from csi_slt.engine.prompt_sampler import PromptSampler


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompt(prompt_id, instruction_lang, target_lang, split="train"):
    return {
        "id": prompt_id,
        "split": split,
        "intent": "sign_translation",
        "instruction_lang": instruction_lang,
        "target_lang": target_lang,
        "template": f"{prompt_id}: {{{{ video_start_token }}}}",
    }


def test_random_sampling_is_stable_within_epoch_and_changes_across_epochs(tmp_path):
    prompt_path = tmp_path / "train.jsonl"
    _write_jsonl(
        prompt_path,
        [_prompt(f"en_de_{index}", "en", "de") for index in range(16)],
    )
    sampler = PromptSampler(prompt_path, strategy="random", seed=7)

    first = sampler.sample("de", "sample-a", instruction_lang="en")
    assert sampler.sample("de", "sample-a", instruction_lang="en") == first

    observed = {first.id}
    for epoch in range(1, 12):
        sampler.set_epoch(epoch)
        observed.add(
            sampler.sample("de", "sample-a", instruction_lang="en").id
        )
    assert len(observed) > 1


def test_random_sampling_can_choose_any_instruction_language(tmp_path):
    prompt_path = tmp_path / "train.jsonl"
    _write_jsonl(
        prompt_path,
        [
            _prompt("de_de", "de", "de"),
            _prompt("en_de", "en", "de"),
            _prompt("zh_de", "zh", "de"),
        ],
    )
    sampler = PromptSampler(prompt_path, strategy="random", seed=3)

    record = sampler.sample("de", "sample-a")
    assert record.target_lang == "de"
    assert record.instruction_lang in {"de", "en", "zh"}


def test_fixed_sampling_uses_first_matching_id_and_allows_override(tmp_path):
    prompt_path = tmp_path / "train.jsonl"
    _write_jsonl(
        prompt_path,
        [
            _prompt("en_de_002", "en", "de"),
            _prompt("en_de_001", "en", "de"),
        ],
    )
    sampler = PromptSampler(prompt_path, strategy="fixed")

    assert sampler.sample("de", instruction_lang="en").id == "en_de_001"
    assert (
        sampler.sample(
            "de", instruction_lang="en", prompt_id="en_de_002"
        ).id
        == "en_de_002"
    )


def test_manifest_sampling_resolves_and_validates_assignment(tmp_path):
    prompt_path = tmp_path / "heldout.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        prompt_path,
        [
            _prompt("zh_de_heldout", "zh", "de", split="heldout"),
            _prompt("en_zh_heldout", "en", "zh", split="heldout"),
        ],
    )
    _write_jsonl(
        manifest_path,
        [
            {
                "eval_id": "sample-de:prompt-1",
                "translation_id": "sample-de",
                "prompt_id": "zh_de_heldout",
            },
            {
                "eval_id": "sample-de:prompt-2",
                "translation_id": "sample-de",
                "prompt_id": "zh_de_heldout",
            },
            {
                "eval_id": "sample-zh:prompt-1",
                "translation_id": "sample-zh",
                "prompt_id": "en_zh_heldout",
            },
        ],
    )
    sampler = PromptSampler(
        prompt_path,
        strategy="manifest",
        manifest_path=manifest_path,
        expected_split="heldout",
    )

    record = sampler.sample(
        "de", instruction_lang="zh", eval_id="sample-de:prompt-1"
    )
    assert record.id == "zh_de_heldout"
    assignment = sampler.get_assignment("sample-de:prompt-2")
    assert assignment.translation_id == "sample-de"
    with pytest.raises(KeyError, match="No prompt assignment for eval_id"):
        sampler.sample("de", eval_id="missing")
    with pytest.raises(ValueError, match="targets 'de'"):
        sampler.sample("zh", eval_id="sample-de:prompt-1")


def test_rejects_wrong_split_and_bad_sentinel(tmp_path):
    prompt_path = tmp_path / "bad.jsonl"
    row = _prompt("bad", "en", "de", split="heldout")
    row["template"] = "missing sentinel"
    _write_jsonl(prompt_path, [row])

    with pytest.raises(ValueError, match="expected 'train'"):
        PromptSampler(prompt_path, expected_split="train")
    with pytest.raises(ValueError, match="exactly one"):
        PromptSampler(prompt_path, expected_split="heldout")


def test_real_prompt_banks_load():
    train = PromptSampler(
        "promts/train.jsonl", strategy="random", expected_split="train"
    )
    heldout = PromptSampler(
        "promts/heldout.jsonl", strategy="fixed", expected_split="heldout"
    )
    adversarial = PromptSampler(
        "promts/adversarial.jsonl",
        strategy="fixed",
        expected_split="adversarial",
    )

    assert len(train.records) == 54
    assert len(heldout.records) == 36
    assert len(adversarial.records) == 30
    assert adversarial.sample(None, instruction_lang="zh").target_lang is None
