"""Add pseudo-gloss columns to the CSL-Daily-HG parquet splits, in place.

Usage::

    # all three splits of the default data root
    python preprocess/csl_daily/build_pseudo_gloss.py

    # another data root / a subset of splits
    python preprocess/csl_daily/build_pseudo_gloss.py \\
        --data-root /path/to/full-trim-256x256px --splits dev test

Requires ``pip install hanlp[pt]``; the HanLP CTB9 models download themselves
into ``~/.hanlp`` on first use, so the first run needs network access.

The labels follow the zh path of ``WayenVan/ph14t-multilang``
(``scripts/extract_gloss_columns.py``), so CSL-Daily and PH14T share one
pseudo-gloss scheme: HanLP ``CTB9_TOK_ELECTRA_SMALL`` segmentation,
``CTB9_POS_ELECTRA_SMALL`` tagging, the same CTB -> UPOS table and the VTaMo
filters below.  Chinese has no lemmatizer, so a gloss token is the lowercased
surface word.

Appended columns (every original column is kept unchanged)::

    tokens                  HanLP word segmentation of ``translation``
    token_offsets           [start, end) char offsets, aligned with tokens
    content_mask            1/0 per token: kept by the default rule?
    pseudo_gloss_default    NOUN/VERB/ADJ/ADV/NUM/PRON/PROPN, space-joined
    pseudo_gloss_strict     default minus PRON
    pseudo_gloss_relaxed    default plus AUX
    pseudo_gloss_offsets    [start, end) char offsets of the default tokens

``translation`` is already the source-language text, so there is no ``orig_``
copy: in PH14T that prefix marks the German original next to a translated
``translation``, and here the two would be identical.

File layout per split: the original ``<split>.parquet`` is renamed to
``<split>.backup`` and the new ``<split>.parquet`` takes its name, so readers
keep loading ``<split>.parquet``.  The backup name is chosen so that
``load_dataset(<data_root>)`` does not pick it up: its split auto-detection
also matches ``train.parquet.backup`` and ``train.bak.parquet`` and would merge
them into the split.

The script is safe to re-run.  It always reads the backup when one exists,
writes to a hidden temp file first, and swaps names only after checking that the
row count, ``clip_id`` order and every original column are unchanged.  A split
whose ``<split>.parquet`` already has the gloss columns is skipped unless
``--force`` is given.
"""

import argparse
import os
from pathlib import Path

import polars as pl

DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "dataset/CSL-Daily-HG/preprocessed/full-trim-256x256px"
)
SPLITS = ("train", "dev", "test")
TEXT_COLUMN = "translation"
GLOSS_COLUMNS = (
    "tokens", "token_offsets", "content_mask", "pseudo_gloss_default",
    "pseudo_gloss_strict", "pseudo_gloss_relaxed", "pseudo_gloss_offsets",
)

HANLP_TOKENIZER_MODEL = "CTB9_TOK_ELECTRA_SMALL"
HANLP_POS_MODEL = "CTB9_POS_ELECTRA_SMALL"

# POS rules (universal UPOS tags, VTaMo paper Sec. 3.1)
POS_DEFAULT = frozenset({"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"})
POS_STRICT = POS_DEFAULT - {"PRON"}
POS_RELAXED = POS_DEFAULT | {"AUX"}

# CTB -> UPOS (Universal Dependencies Chinese conventions).  Tags missing here
# map to "X"; only the content-word rows can reach a filter set.
CTB_TO_UPOS = {
    # content words
    "NR": "PROPN",   # 专有名词
    "NN": "NOUN",    # 普通名词
    "NT": "NOUN",    # 时间名词 (UD zh: NOUN)
    "M": "NOUN",     # 量词 (UD zh: NOUN)
    "ON": "NOUN",    # 拟声词 (UD zh: NOUN)
    "VV": "VERB",    # 动词
    "VC": "AUX",     # 系动词"是" (UD Chinese: copular auxiliary)
    "VE": "VERB",    # 动词"有" (存在/拥有)
    "VA": "ADJ",     # 谓词性形容词
    "JJ": "ADJ",     # 定语形容词
    "AD": "ADV",     # 副词
    "CD": "NUM",     # 数词
    "OD": "NUM",     # 序数词
    "PN": "PRON",    # 代词
    # function words / others
    "P": "ADP",      # 介词
    "LC": "ADP",     # 方位词 (UD zh: ADP)
    "LB": "ADP",     # 长"被"(被/给,被动标记)
    "CC": "CCONJ",   # 并列连词
    "CS": "SCONJ",   # 从属连词 (虽然/因为...)
    "DT": "DET",     # 限定词
    "DEC": "PART",   # 的 (补语/名词化)
    "DEG": "PART",   # 的 (定语)
    "DER": "PART",   # 得
    "DEV": "PART",   # 地
    "AS": "PART",    # 了/着/过
    "MSP": "PART",   # 其他结构助词
    "SP": "PART",    # 句末语气词
    "ETC": "PART",   # 等
    "SB": "PART",    # 短"被"
    "BA": "PART",    # 把
    "IJ": "INTJ",    # 感叹词
    "PU": "PUNCT",   # 标点
    "FW": "X",       # 外语词
    "X": "X",        # 其他
    "URL": "X",      # 链接
}


def load_hanlp():
    """Load the HanLP CTB9 tokenizer and POS tagger."""
    import hanlp
    from transformers import PreTrainedTokenizerBase

    # HanLP 2.1.3 still calls encode_plus / batch_encode_plus, which
    # Transformers 5 removed; forward them to the tokenizer's __call__.
    if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
        PreTrainedTokenizerBase.encode_plus = (
            lambda self, text, text_pair=None, **kw:
            self(text=text, text_pair=text_pair, **kw)
        )
    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        PreTrainedTokenizerBase.batch_encode_plus = (
            lambda self, batch, **kw: self(text=batch, **kw)
        )

    tokenizer = hanlp.load(getattr(hanlp.pretrained.tok, HANLP_TOKENIZER_MODEL))
    tagger = hanlp.load(getattr(hanlp.pretrained.pos, HANLP_POS_MODEL))
    return tokenizer, tagger


def gloss_row(text: str, tokens: list, ctb_tags: list) -> dict:
    """Build one row's gloss columns from its HanLP tokens and CTB tags."""
    if len(tokens) != len(ctb_tags):
        raise RuntimeError(
            f"HanLP tokenizer/POS length mismatch on {text!r}: "
            f"{len(tokens)} tokens vs {len(ctb_tags)} tags"
        )

    # HanLP keeps surface forms, so offsets come from scanning the text.
    offsets = []
    cursor = 0
    for word in tokens:
        start = text.find(word, cursor)
        if start == -1:  # defensive: should never happen
            start = cursor
        cursor = start + len(word)
        offsets.append((start, cursor))

    upos = [CTB_TO_UPOS.get(tag, "X") for tag in ctb_tags]

    def gloss(keep: frozenset) -> str:
        return " ".join(t.lower() for t, p in zip(tokens, upos) if p in keep)

    content_mask = [int(p in POS_DEFAULT) for p in upos]
    return {
        "tokens": tokens,
        "token_offsets": offsets,
        "content_mask": content_mask,
        "pseudo_gloss_default": gloss(POS_DEFAULT),
        "pseudo_gloss_strict": gloss(POS_STRICT),
        "pseudo_gloss_relaxed": gloss(POS_RELAXED),
        "pseudo_gloss_offsets": [o for o, m in zip(offsets, content_mask) if m],
    }


def add_gloss_columns(df: pl.DataFrame, tokenizer, tagger) -> pl.DataFrame:
    """Append GLOSS_COLUMNS to ``df``, computed from TEXT_COLUMN."""
    texts = [(t or "").strip() for t in df[TEXT_COLUMN].to_list()]
    nonempty = [i for i, t in enumerate(texts) if t]

    tokens = [[] for _ in texts]
    tags = [[] for _ in texts]
    if nonempty:
        batch_tokens = tokenizer([texts[i] for i in nonempty])
        batch_tags = tagger(batch_tokens)
        for i, tok, tag in zip(nonempty, batch_tokens, batch_tags):
            tokens[i], tags[i] = tok, tag

    rows = [gloss_row(*args) for args in zip(texts, tokens, tags)]
    return df.with_columns(
        pl.Series(key, [row[key] for row in rows]) for key in GLOSS_COLUMNS
    )


def process_split(data_root: Path, split: str, tokenizer, tagger, force: bool) -> None:
    current = data_root / f"{split}.parquet"
    backup = data_root / f"{split}.backup"
    # Hidden, so load_dataset(<data_root>) ignores it if a run dies mid-write.
    tmp = data_root / f".{split}.parquet.tmp"

    if backup.exists():
        if current.exists():
            has_gloss = set(GLOSS_COLUMNS) <= set(pl.read_parquet_schema(current))
            if not has_gloss:
                raise RuntimeError(
                    f"{current} and {backup} both exist but {current.name} has no "
                    f"gloss columns; resolve by hand which one is the original"
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

    out = add_gloss_columns(df, tokenizer, tagger)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append pseudo-gloss columns to the CSL-Daily-HG parquet splits."
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help=f"Directory holding <split>.parquet (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
        help="Splits to process (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild splits that already have gloss columns, from their backup",
    )
    args = parser.parse_args()

    tokenizer, tagger = load_hanlp()
    for split in args.splits:
        process_split(args.data_root, split, tokenizer, tagger, args.force)


if __name__ == "__main__":
    main()
