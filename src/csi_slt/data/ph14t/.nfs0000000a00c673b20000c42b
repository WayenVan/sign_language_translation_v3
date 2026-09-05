"""PHOENIX-2014-T gloss surface forms, used only to seed the CTC codebook.

The CTC vocabulary is built from the dataset's own gloss annotation ("orth"),
whose spelling conventions are PHOENIX's, not German's: everything is upper
case, umlauts are transliterated (``KOENNEN``), compounds and multi-word signs
are hyphenated (``WIE-AUSSEHEN``), fingerspelling is plus-separated
(``Z+D+F``), and a few morphological markers are carried as lowercase prefixes
(``neg-``, ``negalp-``, ``poss-``).

An LLM tokenizer has never seen that spelling. ``KOENNEN`` splits into
``['KO', 'ENN', 'EN']``, so averaging those sub-token embeddings to initialize
a codebook row lands nowhere near where the language model keeps "können".
Rewriting the same gloss as ``" können"`` makes it a single known token whose
nearest neighbours are ``kann``/``müssen``.

This module owns that rewriting, and nothing else: it is a pure
string-to-strings function with no knowledge of any model, tokenizer or
embedding. Everything downstream of it (choosing among the candidates,
resolving them to sub-token ids, averaging the embeddings) is dataset-agnostic
and lives outside. Another gloss corpus with different annotation conventions
supplies its own function here instead of touching the codebook.
"""

import re


__all__ = ["phoenix_gloss_surface_candidates"]


# ``neg-`` and ``negalp-`` mark a negated sign (negalp additionally marks the
# alpha/headshake variant). Both are rendered as an explicit "nicht", which
# keeps the negation in the initialization instead of collapsing a sign onto
# its own opposite; "nicht" costs a single sub-token. The eight negalp- glosses
# include ``negalp-KEIN`` -> "nicht kein", a double negation left as-is rather
# than special-cased: it is 0.09% of gloss tokens and phase B trains the row.
_NEGATION_PREFIX = re.compile(r"^(?:negalp|neg)-")
# ``poss-`` marks a possessive sign whose stem is already the possessive word
# (``poss-MEIN`` -> "mein"), so the marker is simply dropped.
_POSSESSIVE_PREFIX = re.compile(r"^poss-")
# Sense-disambiguating suffix on otherwise identical glosses (``HABEN2``). The
# digit is not part of the word, and keeping it would average in the embedding
# of a numeral.
_SENSE_SUFFIX = re.compile(r"(?<=[A-Za-z])\d+$")

# PHOENIX transliterates umlauts. Restoring them is only ever a candidate --
# the caller keeps whichever form its tokenizer segments best -- because the
# digraphs also occur without an umlaut behind them (``NEUE``, ``BAUER``).
_UMLAUTS = (("ue", "ü"), ("oe", "ö"), ("ae", "ä"))


def phoenix_gloss_surface_candidates(gloss: str) -> tuple[str, ...]:
    """Return German surface forms of ``gloss``, best-first.

    Every candidate carries a leading space, which is what marks a word start
    for a byte-level BPE tokenizer and roughly doubles how many glosses stay a
    single token.

    The candidates differ only in spelling choices this module cannot settle
    on its own -- which transliterated digraphs are really umlauts, and whether
    a gloss is a capitalized German noun. The caller resolves that by keeping
    the candidate its tokenizer splits into the fewest pieces, falling back to
    this order on a tie. Umlaut-restored forms come first because they are the
    correct German spelling wherever they apply.
    """
    if not isinstance(gloss, str):
        raise TypeError(f"gloss must be a string, got {type(gloss).__name__}")

    stem = _NEGATION_PREFIX.sub("nicht ", gloss)
    stem = _POSSESSIVE_PREFIX.sub("", stem)
    stem = _SENSE_SUFFIX.sub("", stem)
    # Fingerspelled letter chains become one word; hyphens separate the words
    # of a compound or multi-word sign.
    stem = stem.replace("+", "").replace("-", " ").lower()
    stem = " ".join(stem.split())
    if not stem:
        raise ValueError(f"gloss {gloss!r} has no surface form")

    candidates: list[str] = []
    for spelling in _umlaut_variants(stem):
        for form in (spelling, _capitalize_words(spelling)):
            candidate = " " + form
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _umlaut_variants(stem: str) -> list[str]:
    """All-digraphs-restored first, then one at a time, then the raw stem."""
    variants = [stem]
    for source, target in _UMLAUTS:
        variants.append(stem.replace(source, target))
    fully_restored = stem
    for source, target in _UMLAUTS:
        fully_restored = fully_restored.replace(source, target)

    ordered = [fully_restored] + [v for v in variants if v != fully_restored]
    seen: set[str] = set()
    return [v for v in ordered if not (v in seen or seen.add(v))]


def _capitalize_words(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())
