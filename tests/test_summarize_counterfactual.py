import json
from pathlib import Path

import pytest

from csi_slt.commands.summarize_counterfactual import (
    Condition,
    LanguageDetector,
    check_design,
    compute_bsa,
    group_rows_by_language,
    _distractor_of,
)
from csi_slt.engine.prompt_sampler import PromptSampler


LANGUAGES = ("de", "en", "zh")


def _condition(template, target, distractor):
    return Condition(
        template=template,
        variant=f"{template}_001",
        target=target,
        distractor=distractor,
        prompt_id=f"{template}_en_{target}_001",
        directory=Path("unused"),
    )


# --------------------------------------------------------------------------- #
# BSA
# --------------------------------------------------------------------------- #


def test_bsa_is_zero_for_a_model_with_a_fixed_language_preference():
    # Always answers German: right whenever German is the target, wrong on the
    # reverse direction of the very same pair. LAcc says 0.5, BSA says 0.0 --
    # which is the whole reason BSA is the headline number.
    hits = {
        ("cf_first", "de", "en"): [True] * 4,
        ("cf_first", "en", "de"): [False] * 4,
    }

    bsa = compute_bsa(hits, ("cf_first",), ("de", "en"))

    assert bsa["overall"] == 0.0
    assert bsa["by_template"]["cf_first"] == 0.0
    assert bsa["by_pair"]["de/en"] == 0.0
    assert bsa["num_pair_instances"] == 4


def test_bsa_counts_a_video_only_when_both_directions_succeed():
    hits = {
        ("cf_first", "de", "en"): [True, False, True],
        ("cf_first", "en", "de"): [True, True, False],
    }

    bsa = compute_bsa(hits, ("cf_first",), ("de", "en"))

    assert bsa["overall"] == pytest.approx(1 / 3)
    assert bsa["by_template_pair"]["cf_first"]["de/en"] == pytest.approx(1 / 3)


def test_bsa_pools_templates_and_pairs_separately():
    hits = {
        ("cf_first", "de", "en"): [True, True],
        ("cf_first", "en", "de"): [True, True],
        ("cf_last", "de", "en"): [True, True],
        ("cf_last", "en", "de"): [False, False],
    }

    bsa = compute_bsa(hits, ("cf_first", "cf_last"), ("de", "en"))

    assert bsa["by_template"] == {"cf_first": 1.0, "cf_last": 0.0}
    assert bsa["by_pair"]["de/en"] == 0.5
    assert bsa["overall"] == 0.5
    assert bsa["num_pair_instances"] == 4


def test_bsa_rejects_directions_of_unequal_length():
    hits = {
        ("cf_first", "de", "en"): [True, True],
        ("cf_first", "en", "de"): [True],
    }

    with pytest.raises(ValueError, match="videos in one direction"):
        compute_bsa(hits, ("cf_first",), ("de", "en"))


# --------------------------------------------------------------------------- #
# Design completeness
# --------------------------------------------------------------------------- #


def test_check_design_accepts_the_complete_six_cell_template():
    conditions = [
        _condition("cf_first", target, distractor)
        for target in LANGUAGES
        for distractor in LANGUAGES
        if target != distractor
    ]

    check_design(conditions, LANGUAGES)


def test_check_design_rejects_a_missing_direction():
    conditions = [
        _condition("cf_first", target, distractor)
        for target in LANGUAGES
        for distractor in LANGUAGES
        if target != distractor
    ]

    with pytest.raises(ValueError, match="missing"):
        check_design(conditions[:-1], LANGUAGES)


def test_check_design_rejects_a_duplicated_condition():
    conditions = [
        _condition("cf_first", "de", "en"),
        _condition("cf_first", "de", "en"),
    ]

    with pytest.raises(ValueError, match="appears twice"):
        check_design(conditions, LANGUAGES)


# --------------------------------------------------------------------------- #
# The shipped prompt banks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("bank_path", "family"),
    [
        ("prompts/generic/counterfactual_first.jsonl", "cf_first"),
        ("prompts/generic/counterfactual_last.jsonl", "cf_last"),
    ],
)
def test_shipped_bank_covers_every_ordered_language_pair(bank_path, family):
    sampler = PromptSampler(bank_path, supported_languages=LANGUAGES)

    conditions = [
        _condition(
            family,
            record.target_lang,
            _distractor_of(sampler, record.id, record.target_lang),
        )
        for record in sampler.records
    ]

    assert len(conditions) == 6
    check_design(conditions, LANGUAGES)


@pytest.mark.parametrize(
    "bank_path",
    [
        "prompts/generic/counterfactual_first.jsonl",
        "prompts/generic/counterfactual_last.jsonl",
    ],
)
def test_shipped_bank_names_each_language_exactly_once(bank_path):
    # Repeating the target language name -- as the canonical template does in
    # "only the German translation" -- would make it both the most frequent and
    # the last-mentioned language name, and a keyword heuristic would pass the
    # experiment without reading the negation.
    sampler = PromptSampler(bank_path, supported_languages=LANGUAGES)

    for record in sampler.records:
        instruction, _, tail = record.template.partition("\n")
        for name in ("German", "English", "Chinese"):
            assert instruction.count(name) <= 1
            assert name not in tail


def test_distractor_of_rejects_a_prompt_naming_one_language(tmp_path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text(
        json.dumps(
            {
                "id": "cf_first_en_de_001",
                "target_lang": "de",
                "template": "Translate into German.\n{{ video_start_token }}",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sampler = PromptSampler(bank, supported_languages=LANGUAGES)

    with pytest.raises(ValueError, match="must name exactly two"):
        _distractor_of(sampler, "cf_first_en_de_001", "de")


# --------------------------------------------------------------------------- #
# Row identity
# --------------------------------------------------------------------------- #


def test_group_rows_by_language_splits_in_dataset_order():
    row_languages = ("en", "en", "de", "de")
    rows = [
        {"index": index, "language": language, "prediction": f"p{index}"}
        for index, language in enumerate(row_languages)
    ]
    labels = ["en", "de", "de", "de"]

    grouped = group_rows_by_language(rows, labels, row_languages, Path("variant"))

    assert [row["index"] for row, _ in grouped["en"]] == [0, 1]
    assert [label for _, label in grouped["en"]] == ["en", "de"]
    assert [row["index"] for row, _ in grouped["de"]] == [2, 3]


def test_group_rows_by_language_rejects_a_reordered_predictions_file():
    row_languages = ("en", "de")
    rows = [
        {"index": 0, "language": "de", "prediction": "x"},
        {"index": 1, "language": "en", "prediction": "y"},
    ]

    with pytest.raises(ValueError, match="no longer matches"):
        group_rows_by_language(rows, ["de", "en"], row_languages, Path("variant"))


def test_group_rows_by_language_rejects_a_short_predictions_file():
    rows = [{"index": 0, "language": "en", "prediction": "x"}]

    with pytest.raises(ValueError, match="holds 1 rows"):
        group_rows_by_language(rows, ["en"], ("en", "de"), Path("variant"))


# --------------------------------------------------------------------------- #
# Detection cache
# --------------------------------------------------------------------------- #


class _StubDetector(LanguageDetector):
    """A detector that never loads a model and records how often it ran."""

    def __init__(self, labels):
        self.model_name = "stub"
        self.batch_size = 1
        self._pipeline = None
        self._labels = labels
        self.calls = 0

    def detect(self, predictions):
        self.calls += 1
        return list(self._labels)


def _rows():
    return [
        {"index": 0, "language": "de", "prediction": "Regen"},
        {"index": 1, "language": "en", "prediction": "Rain"},
    ]


def test_language_detection_is_cached_next_to_the_predictions(tmp_path):
    detector = _StubDetector(["de", "en"])

    first = detector.labels_for(tmp_path, _rows())
    second = detector.labels_for(tmp_path, _rows())

    assert first == second == ["de", "en"]
    assert detector.calls == 1
    cached = [
        json.loads(line)
        for line in (tmp_path / "predictions_langid.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert cached[0] == {"index": 0, "language": "de", "detected": "de"}


def test_language_detection_cache_is_rebuilt_when_the_predictions_change(tmp_path):
    detector = _StubDetector(["de", "en"])
    detector.labels_for(tmp_path, _rows())

    changed = _rows()[:1]
    detector._labels = ["zh"]
    assert detector.labels_for(tmp_path, changed) == ["zh"]
    assert detector.calls == 2


def test_force_langid_recomputes_a_valid_cache(tmp_path):
    detector = _StubDetector(["de", "en"])
    detector.labels_for(tmp_path, _rows())

    detector.labels_for(tmp_path, _rows(), force=True)

    assert detector.calls == 2


def test_blank_predictions_are_never_a_hit():
    detector = LanguageDetector.__new__(LanguageDetector)
    detector.batch_size = 8
    detector._pipeline = object()  # must not be touched: nothing is detectable

    assert detector.detect(["", "   ", "\n"]) == ["<empty>"] * 3
