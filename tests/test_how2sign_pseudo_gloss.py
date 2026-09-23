import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "preprocess/how2sign/build_pseudo_gloss.py"
)
SPEC = importlib.util.spec_from_file_location("how2sign_pseudo_gloss", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gloss_row_uses_lemmas_and_pos_variants():
    row = MODULE.gloss_row(
        "It was signing quickly.",
        ["It", "was", "signing", "quickly", "."],
        ["it", "be", "sign", "quickly", "."],
        ["PRON", "AUX", "VERB", "ADV", "PUNCT"],
    )

    assert row["tokens"] == ["It", "was", "signing", "quickly", "."]
    assert row["token_offsets"] == [(0, 2), (3, 6), (7, 14), (15, 22), (22, 23)]
    assert row["content_mask"] == [1, 0, 1, 1, 0]
    assert row["pseudo_gloss_default"] == "it sign quickly"
    assert row["pseudo_gloss_strict"] == "sign quickly"
    assert row["pseudo_gloss_relaxed"] == "it be sign quickly"
    assert row["pseudo_gloss_offsets"] == [(0, 2), (7, 14), (15, 22)]


def test_gloss_row_falls_back_to_surface_for_missing_lemma():
    row = MODULE.gloss_row("OpenAI", ["OpenAI"], ["_"], ["PROPN"])
    assert row["pseudo_gloss_default"] == "openai"


def test_gloss_row_keeps_interjection_when_label_would_be_empty():
    row = MODULE.gloss_row(
        "Hi.", ["Hi", "."], ["hi", "."], ["INTJ", "PUNCT"]
    )
    assert row["content_mask"] == [1, 0]
    assert row["pseudo_gloss_default"] == "hi"
    assert row["pseudo_gloss_strict"] == "hi"
    assert row["pseudo_gloss_relaxed"] == "hi"
    assert row["pseudo_gloss_offsets"] == [(0, 2)]


def test_gloss_row_keeps_alphabetic_function_words_when_label_would_be_empty():
    row = MODULE.gloss_row(
        "Not all.",
        ["Not", "all", "."],
        ["not", "all", "."],
        ["PART", "DET", "PUNCT"],
    )
    assert row["content_mask"] == [1, 1, 0]
    assert row["pseudo_gloss_relaxed"] == "not all"
    assert row["pseudo_gloss_offsets"] == [(0, 3), (4, 7)]


def test_gloss_row_rejects_misaligned_model_outputs():
    with pytest.raises(RuntimeError, match="output length mismatch"):
        MODULE.gloss_row("hello", ["hello"], [], ["INTJ"])


def test_token_offsets_rejects_unalignable_tokens():
    with pytest.raises(RuntimeError, match="cannot align"):
        MODULE.token_offsets("can't", ["can", "not"])
