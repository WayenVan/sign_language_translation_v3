"""Add English pseudo-gloss columns to the preprocessed How2Sign splits.

The pseudo-glosses are lower-cased lemmas selected with the same universal-POS
rules used for CSL-Daily: the default variant keeps nouns, verbs, adjectives,
adverbs, numbers, pronouns and proper nouns; strict removes pronouns; relaxed
adds auxiliaries.

Examples::

    # Preview 20 rows without changing any parquet file.
    python preprocess/how2sign/build_pseudo_gloss.py --dry-run --limit 20

    # Process all splits in place, retaining each original as <split>.backup.
    python preprocess/how2sign/build_pseudo_gloss.py

The first run downloads Stanza's English models.  They supply tokenization,
multi-word-token expansion, lemmatization, universal POS tags and exact source
offsets.  If the normal content-word rule would produce an empty label, INTJ
tokens are retained as a fallback (for example, ``Hi.`` -> ``hi``).
"""

import argparse
import json
import os
from pathlib import Path

import polars as pl


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "dataset/how2sign-front-clips/preprocessed/sq720-x+50-24fps-256x256px"
)
SPLITS = ("train", "validation", "test")
TEXT_COLUMN = "translation"
GLOSS_COLUMNS = (
    "tokens",
    "token_offsets",
    "content_mask",
    "pseudo_gloss_default",
    "pseudo_gloss_strict",
    "pseudo_gloss_relaxed",
    "pseudo_gloss_offsets",
)

STANZA_PROCESSORS = "tokenize,mwt,pos,lemma"

POS_DEFAULT = frozenset({"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"})
POS_STRICT = POS_DEFAULT - {"PRON"}
POS_RELAXED = POS_DEFAULT | {"AUX"}


def load_stanza():
    """Load Stanza's English tokenizer, MWT expander, tagger and lemmatizer."""
    import stanza
    import torch

    return stanza.Pipeline(
        lang="en",
        processors=STANZA_PROCESSORS,
        use_gpu=torch.cuda.is_available(),
        tokenize_no_ssplit=True,
        verbose=False,
        download_method=stanza.DownloadMethod.REUSE_RESOURCES,
    )


def token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Locate surface tokens in ``text`` from left to right."""
    offsets = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start == -1:
            # HanLP normally preserves surface forms.  A case-insensitive retry
            # handles a defensive normalization without silently inventing spans.
            start = text.lower().find(token.lower(), cursor)
        if start == -1:
            raise RuntimeError(
                f"cannot align HanLP token {token!r} after offset {cursor} in {text!r}"
            )
        cursor = start + len(token)
        offsets.append((start, cursor))
    return offsets


def gloss_row(
    text: str,
    tokens: list[str],
    lemmas: list[str],
    upos: list[str],
    offsets: list[tuple[int, int]] | None = None,
) -> dict:
    """Build one row of pseudo-gloss fields from aligned HanLP outputs."""
    lengths = {len(tokens), len(lemmas), len(upos)}
    if len(lengths) != 1:
        raise RuntimeError(
            f"HanLP output length mismatch on {text!r}: "
            f"tokens={len(tokens)}, lemmas={len(lemmas)}, upos={len(upos)}"
        )

    if offsets is None:
        offsets = token_offsets(text, tokens)
    elif len(offsets) != len(tokens):
        raise RuntimeError(
            f"Stanza offset length mismatch on {text!r}: "
            f"tokens={len(tokens)}, offsets={len(offsets)}"
        )
    normalized = [
        (lemma if lemma and lemma != "_" else token).lower()
        for token, lemma in zip(tokens, lemmas)
    ]

    # CTC targets must not be empty.  For short utterances such as ``Up.`` or
    # ``Not all.``, preserve every token containing a letter when the standard
    # content-word rule would otherwise discard the whole utterance.
    fallback = (
        [any(char.isalpha() for char in token) for token in tokens]
        if not any(pos in POS_DEFAULT for pos in upos)
        else [False] * len(tokens)
    )

    def selected(keep: frozenset[str]) -> list[bool]:
        return [pos in keep or use_fallback for pos, use_fallback in zip(upos, fallback)]

    def gloss(keep: frozenset[str]) -> str:
        mask = selected(keep)
        return " ".join(word for word, keep_word in zip(normalized, mask) if keep_word)

    content_mask = [int(keep) for keep in selected(POS_DEFAULT)]
    return {
        "tokens": tokens,
        "token_offsets": offsets,
        "content_mask": content_mask,
        "pseudo_gloss_default": gloss(POS_DEFAULT),
        "pseudo_gloss_strict": gloss(POS_STRICT),
        "pseudo_gloss_relaxed": gloss(POS_RELAXED),
        "pseudo_gloss_offsets": [
            offset for offset, keep in zip(offsets, content_mask) if keep
        ],
    }


def annotate_texts(texts: list[str], nlp, batch_size: int) -> list[dict]:
    """Annotate texts in bounded batches while preserving empty rows and order."""
    rows: list[dict | None] = [None] * len(texts)
    nonempty = [index for index, text in enumerate(texts) if text]

    for start in range(0, len(nonempty), batch_size):
        indices = nonempty[start : start + batch_size]
        batch = [texts[index] for index in indices]
        docs = nlp.bulk_process(batch)
        if len(docs) != len(batch):
            raise RuntimeError("Stanza changed the number of documents in a batch")
        for index, doc in zip(indices, docs):
            words = [word for sentence in doc.sentences for word in sentence.words]
            rows[index] = gloss_row(
                texts[index],
                [word.text for word in words],
                [word.lemma for word in words],
                [word.upos for word in words],
                [(word.start_char, word.end_char) for word in words],
            )

    empty = gloss_row("", [], [], [])
    return [row if row is not None else empty.copy() for row in rows]


def add_gloss_columns(df: pl.DataFrame, nlp, batch_size: int) -> pl.DataFrame:
    """Append pseudo-gloss columns derived from ``translation``."""
    texts = [(text or "").strip() for text in df[TEXT_COLUMN].to_list()]
    rows = annotate_texts(texts, nlp, batch_size)
    return df.with_columns(
        pl.Series(name, [row[name] for row in rows]) for name in GLOSS_COLUMNS
    )


def preview_split(
    data_root: Path, split: str, nlp, batch_size: int, limit: int
) -> None:
    """Print annotations without changing the source parquet."""
    path = data_root / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path, columns=["clip_id", TEXT_COLUMN], n_rows=limit)
    out = add_gloss_columns(df, nlp, batch_size)
    for row in out.iter_rows(named=True):
        print(json.dumps(row, ensure_ascii=False))


def process_split(
    data_root: Path, split: str, nlp, batch_size: int, force: bool
) -> None:
    """Safely replace one parquet split while retaining its original backup."""
    current = data_root / f"{split}.parquet"
    backup = data_root / f"{split}.backup"
    tmp = data_root / f".{split}.parquet.tmp"

    if backup.exists():
        if current.exists():
            has_gloss = set(GLOSS_COLUMNS) <= set(pl.read_parquet_schema(current))
            if not has_gloss:
                raise RuntimeError(
                    f"{current} and {backup} both exist but {current.name} has no "
                    "gloss columns; resolve manually which file is the original"
                )
            if not force:
                print(f"[{split}] already has gloss columns, skipping (--force to redo)")
                return
        source = backup
    elif current.exists():
        source = current
    else:
        raise FileNotFoundError(f"neither {current} nor {backup} exists")

    df = pl.read_parquet(source)
    clashes = set(GLOSS_COLUMNS) & set(df.columns)
    if clashes:
        raise RuntimeError(f"{source} already has gloss columns: {sorted(clashes)}")
    print(f"[{split}] {len(df)} rows from {source.name}")

    out = add_gloss_columns(df, nlp, batch_size)
    if out.height != df.height:
        raise RuntimeError(f"[{split}] row count changed: {df.height} -> {out.height}")
    if not out["clip_id"].equals(df["clip_id"]):
        raise RuntimeError(f"[{split}] clip_id order changed")
    if not out.select(df.columns).equals(df):
        raise RuntimeError(f"[{split}] original columns changed")
    if out.columns != df.columns + list(GLOSS_COLUMNS):
        raise RuntimeError(f"[{split}] unexpected column layout: {out.columns}")

    out.write_parquet(tmp)
    if source == current:
        os.replace(current, backup)
    os.replace(tmp, current)
    print(f"[{split}] wrote {current} (original kept as {backup.name})")


def repair_empty_labels(
    data_root: Path, split: str, nlp, batch_size: int
) -> None:
    """Re-annotate only rows whose relaxed pseudo-gloss is empty."""
    current = data_root / f"{split}.parquet"
    tmp = data_root / f".{split}.parquet.tmp"
    if not current.exists():
        raise FileNotFoundError(current)

    df = pl.read_parquet(current)
    missing = set(GLOSS_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"{current} lacks gloss columns: {sorted(missing)}")
    indices = [
        index
        for index, value in enumerate(df["pseudo_gloss_relaxed"].to_list())
        if value is None or not value.strip()
    ]
    if not indices:
        print(f"[{split}] no empty pseudo-gloss labels")
        return

    texts = [(df[TEXT_COLUMN][index] or "").strip() for index in indices]
    repairs = annotate_texts(texts, nlp, batch_size)
    columns = {name: df[name].to_list() for name in GLOSS_COLUMNS}
    for index, row in zip(indices, repairs):
        for name in GLOSS_COLUMNS:
            columns[name][index] = row[name]
    out = df.with_columns(pl.Series(name, columns[name]) for name in GLOSS_COLUMNS)
    remaining = sum(
        value is None or not value.strip()
        for value in out["pseudo_gloss_relaxed"].to_list()
    )
    if remaining:
        raise RuntimeError(f"[{split}] {remaining} empty labels remain after repair")
    original_columns = [name for name in df.columns if name not in GLOSS_COLUMNS]
    if not out.select(original_columns).equals(df.select(original_columns)):
        raise RuntimeError(f"[{split}] original columns changed during repair")

    out.write_parquet(tmp)
    os.replace(tmp, current)
    print(f"[{split}] repaired {len(indices)} empty pseudo-gloss labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append English pseudo-gloss columns to How2Sign parquet splits."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLITS), choices=SPLITS
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dry-run", action="store_true", help="print annotations without writing files"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="rows per split shown by --dry-run"
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild an annotated split from its backup"
    )
    parser.add_argument(
        "--repair-empty",
        action="store_true",
        help="only rebuild rows whose pseudo_gloss_relaxed value is empty",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    nlp = load_stanza()
    for split in args.splits:
        if args.dry_run:
            preview_split(args.data_root, split, nlp, args.batch_size, args.limit)
        elif args.repair_empty:
            repair_empty_labels(args.data_root, split, nlp, args.batch_size)
        else:
            process_split(args.data_root, split, nlp, args.batch_size, args.force)


if __name__ == "__main__":
    main()
