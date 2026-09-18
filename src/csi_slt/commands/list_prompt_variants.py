"""Enumerate the aligned prompt variants of one prompt bank.

Prompt IDs follow ``<family>_<instruction lang>_<target lang>_<number>``, so
``<family>_<number>`` -- ``diverse_003``, ``heldout_001`` -- names the same
instruction variant across target languages. One line is printed per variant,
TAB-separated, the variant key first and then its prompt ID per language:

    diverse_001<TAB>diverse_en_de_001<TAB>diverse_en_en_001<TAB>diverse_en_zh_001

Evaluation launchers loop over this output and hand one line's IDs to
``FixedPromptResolver``, which is what keeps every target language on the same
instruction variant within one evaluation run. A variant that is missing in any
requested language is a hard error rather than a silently shorter suite: the
suites are reported as mean +- standard deviation across variants, and that
number is only comparable when every language contributes the same variants.

Examples:

    python -m csi_slt.commands.list_prompt_variants \
        --prompt-bank prompts/generic/train.jsonl --families canonical,diverse

    python -m csi_slt.commands.list_prompt_variants \
        --prompt-bank prompts/generic/heldout.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from csi_slt.engine.prompt_sampler import PromptSampler


DEFAULT_LANGUAGES = ("de", "en", "zh")

# canonical_en_de_001 -> family=canonical, instruction=en, target=de, number=001
_ID_PATTERN = re.compile(
    r"^(?P<family>[a-z][a-z0-9]*(?:_[a-z0-9]+)*?)"
    r"_(?P<instruction_lang>[a-z]{2})"
    r"_(?P<target_lang>[a-z]{2})"
    r"_(?P<number>\d+)$"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List prompt-bank variants shared by every target language."
    )
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument(
        "--families",
        default="",
        help=(
            "Comma-separated ID families to keep, in output order "
            "(default: every family in the bank, alphabetically)."
        ),
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated target languages, in output column order.",
    )
    return parser.parse_args(argv)


def _split_option(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def collect_variants(
    sampler: PromptSampler,
    languages: tuple[str, ...],
    families: tuple[str, ...],
) -> list[tuple[str, dict[str, str]]]:
    """Group a bank's prompt IDs into per-variant, per-language ID triples."""

    requested_languages = set(languages)
    allowed_families = set(families)
    # (family, number) keeps the numeric ordering of variants inside a family
    # while the family's own position follows ``--families``.
    variants: dict[tuple[str, int], dict[str, str]] = {}
    seen_families: list[str] = []

    for record in sampler.records:
        match = _ID_PATTERN.match(record.id)
        if match is None:
            raise ValueError(
                f"prompt id {record.id!r} does not follow "
                "<family>_<instruction lang>_<target lang>_<number>"
            )
        if match.group("target_lang") != record.target_lang:
            raise ValueError(
                f"prompt id {record.id!r} encodes target language "
                f"{match.group('target_lang')!r} but declares "
                f"{record.target_lang!r}"
            )
        family = match.group("family")
        if allowed_families and family not in allowed_families:
            continue
        if record.target_lang not in requested_languages:
            continue
        if family not in seen_families:
            seen_families.append(family)
        key = (family, int(match.group("number")))
        by_language = variants.setdefault(key, {})
        if record.target_lang in by_language:
            raise ValueError(
                f"variant {family}_{match.group('number')} has two "
                f"{record.target_lang!r} prompts: "
                f"{by_language[record.target_lang]!r} and {record.id!r}"
            )
        by_language[record.target_lang] = record.id

    if not variants:
        requested = ", ".join(sorted(allowed_families)) if allowed_families else "any"
        raise ValueError(
            f"no prompts in {', '.join(str(p) for p in sampler.prompt_paths)} "
            f"match families [{requested}] and languages "
            f"[{', '.join(languages)}]"
        )

    unknown_families = allowed_families.difference(seen_families)
    if unknown_families:
        raise ValueError(
            f"families not present in the bank: {sorted(unknown_families)}"
        )

    incomplete = {
        f"{family}_{number:03d}": sorted(requested_languages - set(by_language))
        for (family, number), by_language in sorted(variants.items())
        if len(by_language) != len(requested_languages)
    }
    if incomplete:
        details = "; ".join(
            f"{variant} missing {missing}" for variant, missing in incomplete.items()
        )
        raise ValueError(f"prompt variants are not aligned across languages: {details}")

    family_rank = {
        family: index
        for index, family in enumerate(families if families else sorted(seen_families))
    }
    ordered_keys = sorted(
        variants, key=lambda key: (family_rank[key[0]], key[1], key[0])
    )
    return [
        (f"{family}_{number:03d}", variants[(family, number)])
        for family, number in ordered_keys
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    languages = _split_option(args.languages) or DEFAULT_LANGUAGES
    families = _split_option(args.families)

    # PromptSampler validates the bank itself: unique IDs, supported target
    # languages, exactly one video sentinel per template.
    sampler = PromptSampler(args.prompt_bank, supported_languages=languages)
    for variant, prompt_ids in collect_variants(sampler, languages, families):
        print("\t".join([variant, *(prompt_ids[language] for language in languages)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
