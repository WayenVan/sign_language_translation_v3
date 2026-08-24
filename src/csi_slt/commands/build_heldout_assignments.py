"""Build the deterministic held-out evaluation prompt manifest.

Each Hugging Face dataset row is paired with every held-out prompt whose
instruction and target languages both match the row's target language.  With
the current prompt bank this produces four evaluation assignments per row.

Example:

    python -m csi_slt.commands.build_heldout_assignments \
        --output promts/heldout_assignments.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path


def read_prompt_bank(path: Path) -> list[dict]:
    prompts: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                prompt = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if prompt.get("split") != "heldout":
                raise ValueError(
                    f"{path}:{line_number}: expected split='heldout', "
                    f"got {prompt.get('split')!r}"
                )
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"prompt bank is empty: {path}")
    return prompts


def build_assignments(
    dataset: Iterable[Mapping[str, object]],
    prompts: Iterable[Mapping[str, object]],
    *,
    id_column: str = "id",
    language_column: str = "lang",
) -> list[dict[str, str]]:
    """Pair every dataset row with its same-language held-out prompts."""

    prompts_by_language: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for prompt in prompts:
        prompt_id = prompt.get("id")
        target_lang = prompt.get("target_lang")
        instruction_lang = prompt.get("instruction_lang")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("every prompt must have a non-empty string id")
        # Matching both fields is intentional: target-only matching would select
        # 12 prompts per row from the current 3x3 prompt bank instead of four.
        if target_lang == instruction_lang and isinstance(target_lang, str):
            prompts_by_language[target_lang].append(prompt)

    for language in prompts_by_language:
        prompts_by_language[language].sort(key=lambda row: str(row["id"]))

    assignments: list[dict[str, str]] = []
    seen_dataset_ids: set[str] = set()
    for row_number, row in enumerate(dataset, start=1):
        if id_column not in row or language_column not in row:
            raise KeyError(
                f"dataset row {row_number} lacks {id_column!r} or {language_column!r}"
            )
        translation_id = str(row[id_column])
        target_lang = row[language_column]
        if not translation_id:
            raise ValueError(f"dataset row {row_number} has an empty id")
        if translation_id in seen_dataset_ids:
            raise ValueError(f"duplicate dataset id: {translation_id!r}")
        seen_dataset_ids.add(translation_id)

        matching_prompts = prompts_by_language.get(str(target_lang), [])
        if not matching_prompts:
            raise ValueError(
                f"no same-language held-out prompts for dataset row "
                f"{translation_id!r} (target language {target_lang!r})"
            )
        for prompt in matching_prompts:
            prompt_id = str(prompt["id"])
            assignments.append(
                {
                    "eval_id": f"{translation_id}:{prompt_id}",
                    "translation_id": translation_id,
                    "prompt_id": prompt_id,
                }
            )
    return assignments


def write_jsonl(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign every multilingual validation row its four held-out prompts."
    )
    parser.add_argument("--dataset", default="WayenVan/ph14t-multilang")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--prompt-bank", type=Path, default=Path("promts/heldout.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("promts/heldout_assignments.jsonl")
    )
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--language-column", default="lang")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    prompts = read_prompt_bank(args.prompt_bank)
    assignments = build_assignments(
        dataset,
        prompts,
        id_column=args.id_column,
        language_column=args.language_column,
    )
    write_jsonl(args.output, assignments)
    print(
        f"Wrote {len(assignments)} assignments for {len(dataset)} dataset rows "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
