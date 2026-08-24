from pathlib import Path

import pytest

from csi_slt.commands.build_prompt_assignments import (
    build_assignments,
    read_prompt_bank,
)


def test_builds_four_same_language_assignments_per_dataset_row():
    dataset = [{"id": "sample-de", "lang": "de"}, {"id": "sample-en", "lang": "en"}]
    prompts = [
        {
            "id": f"heldout_{target}_{index}",
            "target_lang": target,
        }
        for target in ("de", "en", "zh")
        for index in range(1, 5)
    ]

    assignments = build_assignments(dataset, prompts)

    assert len(assignments) == 8
    assert assignments[0] == {
        "eval_id": "sample-de:heldout_de_1",
        "translation_id": "sample-de",
        "prompt_id": "heldout_de_1",
    }
    assert {row["prompt_id"].split("_")[1] for row in assignments[:4]} == {"de"}
    assert {row["prompt_id"].split("_")[1] for row in assignments[4:]} == {"en"}


def test_rejects_duplicate_dataset_ids():
    prompts = [{"id": "heldout_de_1", "target_lang": "de"}]
    with pytest.raises(ValueError, match="duplicate dataset id"):
        build_assignments([{"id": "same", "lang": "de"}] * 2, prompts)


@pytest.mark.parametrize(
    ("bank_path", "assignments_per_row"),
    [
        ("prompts/heldout.jsonl", 4),
        ("prompts/wrong_task.jsonl", 1),
        ("prompts/unrelated.jsonl", 1),
    ],
)
def test_real_evaluation_banks_expand_by_target_language(
    bank_path, assignments_per_row
):
    dataset = [
        {"id": "sample-de", "lang": "de"},
        {"id": "sample-en", "lang": "en"},
        {"id": "sample-zh", "lang": "zh"},
    ]

    assignments = build_assignments(dataset, read_prompt_bank(Path(bank_path)))

    assert len(assignments) == len(dataset) * assignments_per_row
