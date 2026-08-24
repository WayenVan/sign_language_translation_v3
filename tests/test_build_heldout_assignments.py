import pytest

from csi_slt.commands.build_heldout_assignments import build_assignments


def test_builds_four_same_language_assignments_per_dataset_row():
    dataset = [{"id": "sample-de", "lang": "de"}, {"id": "sample-en", "lang": "en"}]
    prompts = [
        {
            "id": f"{instruction}_{target}_{index}",
            "instruction_lang": instruction,
            "target_lang": target,
        }
        for instruction in ("de", "en", "zh")
        for target in ("de", "en", "zh")
        for index in range(1, 5)
    ]

    assignments = build_assignments(dataset, prompts)

    assert len(assignments) == 8
    assert assignments[0] == {
        "eval_id": "sample-de:de_de_1",
        "translation_id": "sample-de",
        "prompt_id": "de_de_1",
    }
    assert {row["prompt_id"].split("_")[0] for row in assignments[:4]} == {"de"}
    assert {row["prompt_id"].split("_")[1] for row in assignments[4:]} == {"en"}


def test_rejects_duplicate_dataset_ids():
    prompts = [{"id": "de_de_1", "instruction_lang": "de", "target_lang": "de"}]
    with pytest.raises(ValueError, match="duplicate dataset id"):
        build_assignments([{"id": "same", "lang": "de"}] * 2, prompts)
