"""Aggregate one prompt-suite directory into summary.json and summary.md.

A prompt suite is one directory holding one subdirectory per prompt variant,
each written by ``csi_slt.commands.evaluate``::

    outputs/eval/<run>/<checkpoint-step>/diverse/
        variants.tsv                 # what scripts/eval asked to run
        canonical_001/predictions_metrics.json
        diverse_001/predictions_metrics.json
        ...

This command reads those per-variant metric files and reports the suite the way
refine-logs/PROMPT_PROTOCOL.md requires it: per variant, then mean +- standard
deviation across variants (sample standard deviation, ddof=1). ``variants.tsv``
-- the output of ``csi_slt.commands.list_prompt_variants`` -- is the expected
variant list, so a variant whose run died is reported as missing instead of
quietly shrinking the mean.

Both outputs are derived files: deleting them and re-running this command
reproduces them from the untouched predictions.

The ``--diagnostic`` suites (wrong_task, unrelated) instruct the model to do
something other than translate the signing, so their BLEU/ROUGE against the
translation reference measures nothing. Those runs are still summarized, but
the metric is target-language accuracy -- did the model answer in the language
the instruction asked for -- and everything scored against the reference is
marked as appendix-only. ``--language-distribution`` additionally reports what
language the outputs were actually in, using the same detector the metric uses.

Examples:

    python -m csi_slt.commands.summarize_prompt_suite \
        outputs/eval/<run>/checkpoint-180000/diverse

    python -m csi_slt.commands.summarize_prompt_suite \
        outputs/eval/<run>/checkpoint-180000/wrong_task \
        --diagnostic --language-distribution
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LANGUAGES = ("de", "en", "zh")

# Reported per language, in this order. ``num_samples`` rides along as a count
# so a variant that evaluated the wrong number of samples is visible, but it is
# never averaged across variants.
METRIC_KEYS = ("bleu4", "bleu1", "rougeL", "bert_score_f1", "lacc")
COUNT_KEYS = ("num_samples",)

# Suite-level columns. ``bleu4``/``bert_score_f1``/``lacc`` come from the
# metric's own macro average over languages; ``rougeL`` has no macro variant and
# is the pooled overall score, which is why it is listed separately here.
MACRO_SOURCES = {
    "bleu4": "test_overall_macro_bleu4",
    "bert_score_f1": "test_overall_macro_bert_score_f1",
    "lacc": "test_overall_macro_lacc",
    "rougeL": "test_overall_rougeL",
}

# For a diagnostic suite, only these survive as a real measurement.
DIAGNOSTIC_PRIMARY = ("lacc",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one prompt-suite directory of evaluation runs."
    )
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated target languages, in report column order.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Treat reference-scored metrics as appendix-only and report "
            "target-language accuracy as the primary result (wrong_task, "
            "unrelated)."
        ),
    )
    parser.add_argument(
        "--language-distribution",
        action="store_true",
        help=(
            "Also detect the output language of every prediction and report "
            "its distribution per requested language (loads the same detector "
            "the metric uses)."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=-1,
        help=(
            "Quote this many predictions per language for qualitative reading "
            "(default: 3 for --diagnostic, 0 otherwise)."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def _split_option(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def read_variants_file(path: Path) -> tuple[list[str], dict[str, dict[str, str]], str]:
    """Read the expected variant list written next to the suite's runs."""

    variants: list[str] = []
    prompt_ids: dict[str, dict[str, str]] = {}
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    bank = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line.lstrip("# ").partition("=")
            if key.strip() == "bank":
                bank = value.strip()
            elif key.strip() == "languages":
                languages = _split_option(value)
            continue
        fields = line.split("\t")
        variant, ids = fields[0], fields[1:]
        variants.append(variant)
        prompt_ids[variant] = dict(zip(languages, ids, strict=False))
    return variants, prompt_ids, bank


def extract_metrics(
    raw: dict[str, Any], languages: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Reshape one ``predictions_metrics.json`` into per-language groups."""

    metrics: dict[str, dict[str, float]] = {}
    for language in languages:
        group = {}
        for key in METRIC_KEYS + COUNT_KEYS:
            value = raw.get(f"test_{language}_{key}")
            if value is not None:
                group[key] = value
        if group:
            metrics[language] = group
    macro = {
        name: raw[source] for name, source in MACRO_SOURCES.items() if source in raw
    }
    if macro:
        metrics["macro"] = macro
    return metrics


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sample_predictions(
    rows: list[dict[str, Any]], languages: tuple[str, ...], count: int
) -> dict[str, list[dict[str, str]]]:
    samples: dict[str, list[dict[str, str]]] = {}
    for language in languages:
        picked = [row for row in rows if row.get("language") == language][:count]
        if picked:
            samples[language] = [
                {
                    "prediction": row.get("prediction", ""),
                    "reference": row.get("reference", ""),
                }
                for row in picked
            ]
    return samples


class LanguageDistribution:
    """Detected-output-language histogram, per requested target language."""

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        from csi_slt.engine.sft.metrics import SLTMetric

        # Reuse the metric's own defaults so a distribution reported here and
        # the lacc reported by the evaluation agree on what "German" means.
        defaults = SLTMetric.__init__.__kwdefaults__ or {}
        self.model_name = model_name or defaults.get(
            "language_detector_model_type",
            "papluca/xlm-roberta-base-language-detection",
        )
        self.batch_size = batch_size
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            from transformers import pipeline

            self._detector = pipeline(
                "text-classification", model=self.model_name, device=-1
            )
        return self._detector

    def __call__(
        self, rows: list[dict[str, Any]], languages: tuple[str, ...]
    ) -> dict[str, dict[str, int]]:
        distribution: dict[str, dict[str, int]] = {}
        for language in languages:
            predictions = [
                str(row.get("prediction", ""))
                for row in rows
                if row.get("language") == language
            ]
            if not predictions:
                continue
            counter: Counter[str] = Counter()
            nonempty = [text for text in predictions if text.strip()]
            counter["<empty>"] = len(predictions) - len(nonempty)
            if nonempty:
                results = self._get_detector()(
                    nonempty, batch_size=self.batch_size, truncation=True
                )
                for result in results:
                    label = str(result["label"]).strip().lower().replace("_", "-")
                    counter[label] += 1
            distribution[language] = {
                label: count for label, count in counter.most_common() if count
            }
        return distribution


def aggregate(
    variant_metrics: list[dict[str, dict[str, float]]],
    languages: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean, sample standard deviation, min and max across prompt variants."""

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for group in (*languages, "macro"):
        group_summary: dict[str, dict[str, float]] = {}
        for key in METRIC_KEYS:
            values = [
                float(metrics[group][key])
                for metrics in variant_metrics
                if group in metrics and key in metrics[group]
            ]
            if not values:
                continue
            group_summary[key] = {
                "mean": statistics.fmean(values),
                # One variant has no spread; report 0.0 rather than crash so a
                # single-variant suite shares the multi-variant schema.
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        if group_summary:
            aggregated[group] = group_summary
    return aggregated


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    suite_dir: Path = args.suite_dir
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"suite directory not found: {suite_dir}")

    languages = _split_option(args.languages) or DEFAULT_LANGUAGES
    samples = args.samples if args.samples >= 0 else (3 if args.diagnostic else 0)

    variants_file = suite_dir / "variants.tsv"
    if variants_file.is_file():
        expected, prompt_ids, bank = read_variants_file(variants_file)
    else:
        expected = sorted(
            path.name for path in suite_dir.iterdir() if path.is_dir()
        )
        prompt_ids, bank = {}, ""

    distribution_tool = (
        LanguageDistribution() if args.language_distribution else None
    )

    checkpoint_dir = ""
    variants: list[dict[str, Any]] = []
    missing: list[str] = []
    for variant in expected:
        metrics_path = suite_dir / variant / "predictions_metrics.json"
        if not metrics_path.is_file():
            missing.append(variant)
            continue
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {
            "variant": variant,
            "prompt_ids": prompt_ids.get(variant, {}),
            "metrics": extract_metrics(raw, languages),
        }

        # evaluate.py dumps the resolved config next to its predictions; it is
        # the authoritative record of what this run actually ran.
        config_path = suite_dir / variant / "eval_config.yaml"
        if config_path.is_file():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            checkpoint_dir = checkpoint_dir or (
                (config.get("model") or {}).get("checkpoint_dir") or ""
            )
            resolved_ids = (
                ((config.get("prompt") or {}).get("test") or {}).get("prompt_ids")
                or {}
            )
            if resolved_ids:
                entry["prompt_ids"] = resolved_ids

        predictions_path = suite_dir / variant / "predictions.jsonl"
        if (samples or distribution_tool) and predictions_path.is_file():
            rows = _read_predictions(predictions_path)
            if samples:
                entry["sample_predictions"] = _sample_predictions(
                    rows, languages, samples
                )
            if distribution_tool is not None:
                entry["language_distribution"] = distribution_tool(rows, languages)
        variants.append(entry)

    if not variants:
        raise FileNotFoundError(
            f"no variant holds predictions_metrics.json under {suite_dir}"
        )

    summary: dict[str, Any] = {
        "suite": suite_dir.name,
        "suite_dir": str(suite_dir),
        "checkpoint_dir": checkpoint_dir,
        "prompt_bank": bank,
        "languages": list(languages),
        "diagnostic": bool(args.diagnostic),
        "primary_metrics": list(
            DIAGNOSTIC_PRIMARY if args.diagnostic else METRIC_KEYS
        ),
        "appendix_metrics": [
            key for key in METRIC_KEYS if key not in DIAGNOSTIC_PRIMARY
        ]
        if args.diagnostic
        else [],
        "variants": variants,
        "aggregate": aggregate([entry["metrics"] for entry in variants], languages),
        "missing": missing,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_commit": _git_commit(),
    }
    if distribution_tool is not None:
        totals: dict[str, Counter[str]] = {}
        for entry in variants:
            for language, counts in entry.get("language_distribution", {}).items():
                totals.setdefault(language, Counter()).update(counts)
        summary["language_distribution"] = {
            language: dict(counter.most_common())
            for language, counter in totals.items()
        }
    return summary


def _format_value(value: float | None, key: str) -> str:
    del key
    if value is None:
        return "--"
    # Every metric here is a 0-1 fraction: SLTMetric divides sacrebleu's 0-100
    # score by 100, so BLEU-4 reads 0.2627, and four decimals keep the same
    # precision a 0-100 BLEU would show with two.
    return f"{value:.4f}"


def render_markdown(summary: dict[str, Any]) -> str:
    languages = tuple(summary["languages"])
    groups = (*languages, "macro")
    lines: list[str] = []

    lines.append(f"# Prompt suite: {summary['suite']}")
    lines.append("")
    lines.append(f"- suite dir: `{summary['suite_dir']}`")
    if summary["checkpoint_dir"]:
        lines.append(f"- checkpoint: `{summary['checkpoint_dir']}`")
    if summary["prompt_bank"]:
        lines.append(f"- prompt bank: `{summary['prompt_bank']}`")
    lines.append(f"- variants: {len(summary['variants'])}")
    if summary["missing"]:
        lines.append(f"- **missing variants**: {', '.join(summary['missing'])}")
    if summary["diagnostic"]:
        lines.append(
            "- **diagnostic suite**: the instruction asks for something other "
            "than a translation, so only "
            f"{', '.join(summary['primary_metrics'])} is a measurement; "
            "everything scored against the translation reference is "
            "appendix-only."
        )
    lines.append(f"- generated: {summary['generated_at']} ({summary['git_commit']})")
    lines.append("")

    primary = list(summary["primary_metrics"])
    appendix = list(summary["appendix_metrics"])

    def metric_tables(keys: list[str]) -> None:
        for key in keys:
            present = [
                entry
                for entry in summary["variants"]
                if any(key in entry["metrics"].get(group, {}) for group in groups)
            ]
            if not present:
                continue
            lines.append(f"### {key}")
            lines.append("")
            lines.append("| variant | " + " | ".join(groups) + " |")
            lines.append("|---" * (len(groups) + 1) + "|")
            for entry in present:
                cells = [
                    _format_value(entry["metrics"].get(group, {}).get(key), key)
                    for group in groups
                ]
                lines.append(f"| {entry['variant']} | " + " | ".join(cells) + " |")
            stats = [summary["aggregate"].get(group, {}).get(key) for group in groups]
            # A one-variant suite has no spread to report; the single row above
            # already is the result.
            if len(summary["variants"]) > 1 and any(stats):
                mean_cells = [
                    "--"
                    if stat is None
                    else (
                        f"{_format_value(stat['mean'], key)} ± "
                        f"{_format_value(stat['std'], key)}"
                    )
                    for stat in stats
                ]
                lines.append("| **mean ± std** | " + " | ".join(mean_cells) + " |")
                range_cells = [
                    "--"
                    if stat is None
                    else (
                        f"{_format_value(stat['min'], key)} – "
                        f"{_format_value(stat['max'], key)}"
                    )
                    for stat in stats
                ]
                lines.append("| min – max | " + " | ".join(range_cells) + " |")
            lines.append("")

    lines.append("## Results")
    lines.append("")
    metric_tables(primary)

    samples_present = any("num_samples" in entry["metrics"].get(languages[0], {}) for entry in summary["variants"])
    if samples_present:
        lines.append("### num_samples")
        lines.append("")
        lines.append("| variant | " + " | ".join(languages) + " |")
        lines.append("|---" * (len(languages) + 1) + "|")
        for entry in summary["variants"]:
            cells = [
                str(int(entry["metrics"].get(language, {}).get("num_samples", 0)) or "--")
                for language in languages
            ]
            lines.append(f"| {entry['variant']} | " + " | ".join(cells) + " |")
        lines.append("")

    if summary.get("language_distribution"):
        lines.append("## Output-language distribution")
        lines.append("")
        lines.append("| requested | detected (count) |")
        lines.append("|---|---|")
        for language, counts in summary["language_distribution"].items():
            detected = ", ".join(f"{label} {count}" for label, count in counts.items())
            lines.append(f"| {language} | {detected} |")
        lines.append("")

    for entry in summary["variants"]:
        if not entry.get("sample_predictions"):
            continue
        lines.append(f"## Sample predictions: {entry['variant']}")
        lines.append("")
        for language, rows in entry["sample_predictions"].items():
            lines.append(f"**{language}**")
            lines.append("")
            for row in rows:
                lines.append(f"- pred: {row['prediction']}")
                lines.append(f"  - ref: {row['reference']}")
            lines.append("")

    if appendix:
        lines.append("## Appendix: reference-scored metrics (not comparable)")
        lines.append("")
        metric_tables(appendix)

    if summary["variants"] and summary["variants"][0]["prompt_ids"]:
        lines.append("## Prompt IDs")
        lines.append("")
        lines.append("| variant | " + " | ".join(languages) + " |")
        lines.append("|---" * (len(languages) + 1) + "|")
        for entry in summary["variants"]:
            cells = [entry["prompt_ids"].get(language, "--") for language in languages]
            lines.append(f"| {entry['variant']} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args)

    json_path = args.output_json or args.suite_dir / "summary.json"
    md_path = args.output_md or args.suite_dir / "summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if summary["missing"]:
        print(
            f"WARNING: {len(summary['missing'])} variant(s) have no metrics: "
            f"{', '.join(summary['missing'])}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
