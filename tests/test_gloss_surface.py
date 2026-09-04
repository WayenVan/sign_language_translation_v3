import pytest

from csi_slt.commands.train import resolve_llm_token_ids
from csi_slt.data.ph14t.gloss_surface import phoenix_gloss_surface_candidates


def first(gloss: str) -> str:
    return phoenix_gloss_surface_candidates(gloss)[0]


def test_every_candidate_starts_a_word():
    for gloss in ("REGEN", "neg-KAUM", "Z+D+F", "WIE-AUSSEHEN"):
        assert all(
            candidate.startswith(" ")
            for candidate in phoenix_gloss_surface_candidates(gloss)
        )


def test_negation_prefixes_become_an_explicit_nicht():
    assert first("neg-REGEN") == " nicht regen"
    assert first("negalp-MUSS") == " nicht muss"


def test_possessive_prefix_and_sense_suffix_are_dropped():
    assert first("poss-MEIN") == " mein"
    assert first("HABEN2") == " haben"


def test_compounds_split_and_fingerspelling_joins():
    assert first("WIE-AUSSEHEN") == " wie aussehen"
    assert first("Z+D+F") == " zdf"


def test_umlaut_restored_form_is_offered_first_and_the_raw_one_survives():
    candidates = phoenix_gloss_surface_candidates("KOENNEN")

    assert candidates[0] == " können"
    # ``NEUE``-style false positives are why the transliterated spelling stays
    # available for the tokenizer to pick instead.
    assert " koennen" in candidates


def test_capitalized_variant_is_offered_for_german_nouns():
    assert " Regen" in phoenix_gloss_surface_candidates("REGEN")


def test_rejects_a_gloss_without_a_surface_form():
    with pytest.raises(ValueError):
        phoenix_gloss_surface_candidates("+++")
    with pytest.raises(TypeError):
        phoenix_gloss_surface_candidates(None)


class _PieceTokenizer:
    """Splits on a fixed vocabulary of known words, else one id per char."""

    known = {" können": [100], " regen": [101]}

    def encode(self, text, add_special_tokens=False):
        if text in self.known:
            return list(self.known[text])
        return [ord(character) for character in text.strip()]


def test_resolve_keeps_the_candidate_with_the_fewest_pieces():
    ids = resolve_llm_token_ids(
        "KOENNEN",
        llm_tokenizer=_PieceTokenizer(),
        surface_candidates=phoenix_gloss_surface_candidates,
    )

    assert ids == [100]


def test_resolve_falls_back_to_the_raw_token_without_a_mapping():
    ids = resolve_llm_token_ids(
        "KOENNEN",
        llm_tokenizer=_PieceTokenizer(),
        surface_candidates=None,
    )

    assert ids == [ord(character) for character in "KOENNEN"]


def test_resolve_breaks_ties_in_the_datasets_own_order():
    class _UniformTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    ids = resolve_llm_token_ids(
        "REGEN",
        llm_tokenizer=_UniformTokenizer(),
        surface_candidates=lambda gloss: (" regen", " Regen"),
    )

    assert ids == [1, 2]
