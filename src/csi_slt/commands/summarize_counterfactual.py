"""Aggregate the counterfactual instruction-switching suites of one checkpoint.

The two suites written by ``scripts/eval/run_eval_cf_{first,last}_prompt.sh``
hold 12 conditions per video -- 2 templates x 3 unordered language pairs x 2
target directions -- laid out as two prompt variants per suite::

    outputs/eval/<run>/<checkpoint-step>/
        cf_first/{cf_first_001,cf_first_002}/predictions.jsonl
        cf_last/ {cf_last_001, cf_last_002}/predictions.jsonl

``summarize_prompt_suite`` already reports each suite on its own, but the
headline number of this experiment is Bidirectional Switch Accuracy, and BSA is
not a per-suite quantity: it pairs the two *directions* of one language pair,
which live in two different variants, and it pairs them *per video*::

    BSA = mean over (template, pair {A,B}, video v) of
          1[ output(v, target=A, distractor=B) is A
             and output(v, target=B, distractor=A) is B ]

A model with a fixed language preference wins one direction of every pair and
therefore scores near zero, which is the point of the measure.

Pairing the two directions means joining rows *across* target languages, so row
identity has to survive the join. ``predictions.jsonl`` carries an ``index`` but
no video id, and the two directions of a pair are different dataset rows (one
carries the German reference, the other the English one), so the reference text
cannot be the key either. What makes the join exact is that the gathered
prediction order is the dataset order: with ``per_device_eval_batch_size`` rows
per rank, ``gather_for_metrics`` concatenates rank 0's slice before rank 1's, so
each gathered batch is a contiguous ascending run and the permutation is the
identity. Row ``i`` therefore describes dataset row ``i``, whose video name the
dataset itself supplies. Both halves of that claim are asserted rather than
assumed: the per-row language column must match the dataset's, and the video
order must be identical inside every target-language group.

Output languages are detected with the same model ``SLTMetric`` uses for lacc,
so a hit here and a hit there mean the same thing. Detection is the only slow
step, so it is cached next to the predictions as ``predictions_langid.jsonl``
and reused until the predictions change (``--force-langid`` recomputes).

Everything written is derived: deleting ``counterfactual_summary.{json,md}`` and
the langid caches and re-running this command reproduces them from the
untouched predictions.

Examples:

    python -m csi_slt.commands.summarize_counterfactual \
        outputs/eval/4b-multilang-diverse-eval/<run>/checkpoint-37254

    python -m csi_slt.commands.summarize_counterfactual \
        outputs/eval/<run>/checkpoint-38064 --samples 5 --force-langid
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from csi_slt.commands.summarize_prompt_suite import read_variants_file
from csi_slt.engine.prompt_sampler import PromptSampler


DEFAULT_LANGUAGES = ("de", "en", "zh")

# Suite directory -> how the template positions the requested target language.
# The two templates exist to make position a control: a model that answers in
# the first language name it reads scores high on one and near zero on the
# other, and only instruction semantics score high on both.
DEFAULT_TEMPLATES = {"cf_first": "target first", "cf_last": "target last"}

# The instruction language is English throughout
# (refine-logs/PROMPT_PROTOCOL.md), so the distractor of a condition is
# recovered by finding the English language names in the prompt template.
LANGUAGE_NAMES = {"de": "German", "en": "English", "zh": "Chinese"}

# Where the canonical-prompt column of the main table comes from. Canonical lacc
# is not re-measured here: it is read from the run that already produced it, so
# the table's first column is literally the number the diverse suite reported.
CANONICAL_CANDIDATES = ("diverse/canonical_001", "fixed/multi/canonical_001")

# Marker used for a prediction that is empty or whitespace only. It is never a
# hit, and it is kept distinct from a detected language so an output-collapse
# failure is not silently reported as "answered in the wrong language".
EMPTY = "<empty>"


@dataclass(frozen=True)
class Condition:
    """One (template, target, distractor) cell of the 12-condition design."""

    template: str
    variant: str
    target: str
    distractor: str
    prompt_id: str
    directory: Path

    @property
    def label(self) -> str:
        return f"{self.target}<-{self.distractor}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the cf_first/cf_last suites of one checkpoint into "
            "per-condition lacc, Bidirectional Switch Accuracy and a "
            "target-language confusion matrix."
        )
    )
    parser.add_argument(
        "step_dir",
        type=Path,
        help="the <run>/<checkpoint-step> directory holding the suite directories",
    )
    parser.add_argument(
        "--suites",
        default=",".join(DEFAULT_TEMPLATES),
        help="Comma-separated suite directories to read, in report order.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated target languages, in report column order.",
    )
    parser.add_argument(
        "--canonical",
        default="auto",
        help=(
            "Directory of the canonical-prompt run for the first table column: "
            "'auto' searches "
            f"{', '.join(CANONICAL_CANDIDATES)} under the step directory, "
            "'none' omits the column, or pass a path."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Failed predictions to quote per condition (0 disables).",
    )
    parser.add_argument(
        "--force-langid",
        action="store_true",
        help="Recompute the cached output-language detection.",
    )
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        default=32,
        help="Batch size for the output-language detector.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def _split_option(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_language_code(language: str) -> str:
    """Mirror ``SLTMetric._normalize_language_code`` exactly."""

    return language.strip().lower().replace("_", "-")


# --------------------------------------------------------------------------- #
# Output-language detection
# --------------------------------------------------------------------------- #


class LanguageDetector:
    """Per-prediction output-language labels, cached next to the predictions.

    The model defaults to the one ``SLTMetric`` uses, so a hit counted here and
    the lacc reported by ``evaluate`` agree on what "German" means.
    """

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        from csi_slt.engine.sft.metrics import SLTMetric

        defaults = SLTMetric.__init__.__kwdefaults__ or {}
        self.model_name = model_name or defaults.get(
            "language_detector_model_type",
            "papluca/xlm-roberta-base-language-detection",
        )
        self.batch_size = batch_size
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification", model=self.model_name, device=-1
            )
        return self._pipeline

    def detect(self, predictions: list[str]) -> list[str]:
        """Label every prediction, marking blank output as ``<empty>``."""

        labels: list[str] = [EMPTY] * len(predictions)
        indices = [
            index for index, text in enumerate(predictions) if str(text).strip()
        ]
        if not indices:
            return labels
        results = self._get_pipeline()(
            [str(predictions[index]) for index in indices],
            batch_size=self.batch_size,
            truncation=True,
        )
        if len(results) != len(indices):
            raise RuntimeError(
                "Language detector returned an unexpected number of results: "
                f"{len(results)} for {len(indices)} inputs"
            )
        for index, result in zip(indices, results, strict=True):
            labels[index] = _normalize_language_code(str(result["label"]))
        return labels

    def labels_for(
        self, variant_dir: Path, rows: list[dict[str, Any]], *, force: bool = False
    ) -> list[str]:
        """Return this variant's labels, reading or refreshing its cache."""

        cache_path = variant_dir / "predictions_langid.jsonl"
        if not force and cache_path.is_file():
            cached = _read_jsonl(cache_path)
            # The cache is keyed to one exact predictions file. Anything that
            # does not line up means the predictions were re-run, so recompute
            # rather than report last week's answers against this week's text.
            if len(cached) == len(rows) and all(
                entry.get("index") == row.get("index")
                and entry.get("language") == row.get("language")
                for entry, row in zip(cached, rows, strict=True)
            ):
                return [str(entry["detected"]) for entry in cached]

        labels = self.detect([row.get("prediction", "") for row in rows])
        with cache_path.open("w", encoding="utf-8") as handle:
            for row, label in zip(rows, labels, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "index": row.get("index"),
                            "language": row.get("language"),
                            "detected": label,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return labels


# --------------------------------------------------------------------------- #
# Reading one suite
# --------------------------------------------------------------------------- #


def _load_eval_config(variant_dir: Path) -> dict[str, Any]:
    config_path = variant_dir / "eval_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{variant_dir} has no eval_config.yaml; it is the authoritative "
            "record of which prompts this run actually used."
        )
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _distractor_of(sampler: PromptSampler, prompt_id: str, target: str) -> str:
    """Recover a counterfactual prompt's distractor from its template text."""

    record = sampler.by_id(prompt_id, target_lang=target)
    mentioned = {
        code: record.template.index(name)
        for code, name in LANGUAGE_NAMES.items()
        if name in record.template
    }
    if len(mentioned) != 2:
        raise ValueError(
            f"prompt {prompt_id!r} names {sorted(mentioned)} language(s); a "
            "counterfactual prompt must name exactly two"
        )
    if target not in mentioned:
        raise ValueError(
            f"prompt {prompt_id!r} targets {target!r} but never names "
            f"{LANGUAGE_NAMES[target]!r}"
        )
    (distractor,) = set(mentioned) - {target}
    return distractor


def collect_conditions(
    step_dir: Path, suites: tuple[str, ...], languages: tuple[str, ...]
) -> tuple[list[Condition], str]:
    """Enumerate the conditions of every requested suite, with provenance."""

    conditions: list[Condition] = []
    checkpoint_dir = ""
    for suite in suites:
        suite_dir = step_dir / suite
        if not suite_dir.is_dir():
            raise FileNotFoundError(
                f"suite directory not found: {suite_dir}; run "
                f"scripts/eval/run_eval_{suite}_prompt.sh first"
            )
        variants_file = suite_dir / "variants.tsv"
        if not variants_file.is_file():
            raise FileNotFoundError(f"{suite_dir} has no variants.tsv")
        expected, _, bank = read_variants_file(variants_file)
        if not bank:
            raise ValueError(f"{variants_file} does not record its prompt bank")
        sampler = PromptSampler(bank, supported_languages=languages)

        for variant in expected:
            variant_dir = suite_dir / variant
            if not (variant_dir / "predictions.jsonl").is_file():
                raise FileNotFoundError(
                    f"{variant_dir} holds no predictions.jsonl; the suite is "
                    "incomplete and BSA would be computed over a partial design"
                )
            config = _load_eval_config(variant_dir)
            checkpoint_dir = checkpoint_dir or (
                (config.get("model") or {}).get("checkpoint_dir") or ""
            )
            prompt_ids = (
                ((config.get("prompt") or {}).get("test") or {}).get("prompt_ids")
                or {}
            )
            for target in languages:
                prompt_id = prompt_ids.get(target)
                if not prompt_id:
                    raise ValueError(
                        f"{variant_dir}/eval_config.yaml has no prompt id for "
                        f"target language {target!r}"
                    )
                conditions.append(
                    Condition(
                        template=suite,
                        variant=variant,
                        target=target,
                        distractor=_distractor_of(sampler, prompt_id, target),
                        prompt_id=prompt_id,
                        directory=variant_dir,
                    )
                )
    return conditions, checkpoint_dir


def check_design(conditions: list[Condition], languages: tuple[str, ...]) -> None:
    """Refuse to report BSA over anything but the complete 12-cell design."""

    expected = set(itertools.permutations(languages, 2))
    by_template: dict[str, set[tuple[str, str]]] = {}
    for condition in conditions:
        cell = (condition.target, condition.distractor)
        cells = by_template.setdefault(condition.template, set())
        if cell in cells:
            raise ValueError(
                f"{condition.template}: condition {condition.label} appears twice"
            )
        cells.add(cell)
    for template, cells in by_template.items():
        missing = expected - cells
        if missing:
            raise ValueError(
                f"{template} is missing {sorted(missing)}; every ordered "
                "language pair is needed before BSA means anything"
            )


# --------------------------------------------------------------------------- #
# Row identity
# --------------------------------------------------------------------------- #


def load_video_order(
    data_root: str, languages: tuple[str, ...]
) -> tuple[tuple[str, ...], list[str]]:
    """Return the test split's per-language row languages and video order.

    The returned video order is shared by every target language: BSA joins a
    video's German row to its English row, so the two groups must enumerate the
    same videos in the same sequence. That is a property of the dataset, not
    something this command can repair, so it is asserted here.
    """

    from csi_slt.data.ph14t import Ph14TMultiLinglDataset

    dataset = Ph14TMultiLinglDataset(data_root=data_root, mode="test")
    row_languages = [str(language) for language in dataset.hg_dataset["lang"]]
    names = [str(name) for name in dataset.hg_dataset["name"]]

    per_language = {
        language: [
            name
            for name, row_language in zip(names, row_languages, strict=True)
            if row_language == language
        ]
        for language in languages
    }
    missing = [language for language, group in per_language.items() if not group]
    if missing:
        raise ValueError(f"test split has no rows for language(s) {missing}")

    reference_language = languages[0]
    reference_order = per_language[reference_language]
    for language in languages[1:]:
        if per_language[language] != reference_order:
            raise ValueError(
                f"the {language!r} rows of the test split do not enumerate the "
                f"same videos in the same order as the {reference_language!r} "
                "rows; BSA cannot pair the two directions of a language pair"
            )
    return tuple(row_languages), reference_order


def group_rows_by_language(
    rows: list[dict[str, Any]],
    labels: list[str],
    row_languages: tuple[str, ...],
    variant_dir: Path,
) -> dict[str, list[tuple[dict[str, Any], str]]]:
    """Split one variant's predictions into per-language, video-ordered groups.

    ``predictions.jsonl`` is written in dataset order (see the module docstring),
    which is what lets the k-th row of a language group stand for the k-th video.
    The language column is compared against the dataset's row by row, so a run
    whose order ever stopped matching fails loudly instead of silently pairing
    the wrong videos.
    """

    if len(rows) != len(row_languages):
        raise ValueError(
            f"{variant_dir}/predictions.jsonl holds {len(rows)} rows but the "
            f"test split has {len(row_languages)}"
        )
    grouped: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for position, (row, label) in enumerate(zip(rows, labels, strict=True)):
        expected_language = row_languages[position]
        if row.get("language") != expected_language:
            raise ValueError(
                f"{variant_dir}/predictions.jsonl row {position} is "
                f"{row.get('language')!r} but the test split's row {position} is "
                f"{expected_language!r}; the prediction order no longer matches "
                "the dataset order"
            )
        grouped.setdefault(expected_language, []).append((row, label))
    return grouped


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _mean(values: list[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_bsa(
    hits: dict[tuple[str, str, str], list[bool]],
    templates: tuple[str, ...],
    languages: tuple[str, ...],
) -> dict[str, Any]:
    """Bidirectional Switch Accuracy over every (template, pair, video)."""

    by_template_pair: dict[str, dict[str, float]] = {}
    by_template: dict[str, float] = {}
    by_pair: dict[str, float] = {}
    pooled: list[bool] = []
    pair_pool: dict[str, list[bool]] = {}

    for template in templates:
        template_pool: list[bool] = []
        for first, second in itertools.combinations(languages, 2):
            pair = f"{first}/{second}"
            forward = hits[(template, first, second)]
            backward = hits[(template, second, first)]
            if len(forward) != len(backward):
                raise ValueError(
                    f"{template} {pair}: {len(forward)} videos in one direction "
                    f"and {len(backward)} in the other"
                )
            both = [a and b for a, b in zip(forward, backward, strict=True)]
            by_template_pair.setdefault(template, {})[pair] = _mean(both)
            template_pool.extend(both)
            pair_pool.setdefault(pair, []).extend(both)
            pooled.extend(both)
        by_template[template] = _mean(template_pool)
    for pair, values in pair_pool.items():
        by_pair[pair] = _mean(values)

    return {
        "overall": _mean(pooled),
        "by_template": by_template,
        "by_pair": by_pair,
        "by_template_pair": by_template_pair,
        "num_pair_instances": len(pooled),
    }


def read_canonical(step_dir: Path, choice: str, languages: tuple[str, ...]):
    """Read the canonical-prompt lacc the diverse or fixed suite already wrote."""

    if choice == "none":
        return None
    if choice == "auto":
        candidates = [step_dir / candidate for candidate in CANONICAL_CANDIDATES]
        found = next(
            (
                path
                for path in candidates
                if (path / "predictions_metrics.json").is_file()
            ),
            None,
        )
        if found is None:
            return None
        canonical_dir = found
    else:
        canonical_dir = Path(choice)
        metrics_path = canonical_dir / "predictions_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"no predictions_metrics.json under {canonical_dir}")

    raw = json.loads(
        (canonical_dir / "predictions_metrics.json").read_text(encoding="utf-8")
    )
    lacc = {
        language: raw[f"test_{language}_lacc"]
        for language in languages
        if f"test_{language}_lacc" in raw
    }
    macro = raw.get("test_overall_macro_lacc")
    if macro is not None:
        lacc["macro"] = macro
    return {
        "source": str(canonical_dir),
        "lacc": lacc,
        "bleu4": raw.get("test_overall_macro_bleu4"),
    }


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
    step_dir: Path = args.step_dir
    if not step_dir.is_dir():
        raise FileNotFoundError(f"step directory not found: {step_dir}")
    languages = _split_option(args.languages) or DEFAULT_LANGUAGES
    templates = _split_option(args.suites) or tuple(DEFAULT_TEMPLATES)

    conditions, checkpoint_dir = collect_conditions(step_dir, templates, languages)
    check_design(conditions, languages)

    # Every condition of every suite ran against the same test split, so the
    # data root of any one of them settles the video order for all of them.
    first_config = _load_eval_config(conditions[0].directory)
    data_root = ((first_config.get("data") or {}).get("data_root")) or ""
    if not data_root:
        raise ValueError(
            f"{conditions[0].directory}/eval_config.yaml records no data.data_root"
        )
    row_languages, video_order = load_video_order(data_root, languages)

    detector = LanguageDetector(batch_size=args.detector_batch_size)

    # One variant holds all three target languages, so read and detect once per
    # variant rather than once per condition.
    grouped_by_variant: dict[tuple[str, str], dict[str, list]] = {}
    for condition in conditions:
        key = (condition.template, condition.variant)
        if key in grouped_by_variant:
            continue
        rows = _read_jsonl(condition.directory / "predictions.jsonl")
        labels = detector.labels_for(
            condition.directory, rows, force=args.force_langid
        )
        grouped_by_variant[key] = group_rows_by_language(
            rows, labels, row_languages, condition.directory
        )

    hits: dict[tuple[str, str, str], list[bool]] = {}
    condition_reports: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = {
        language: Counter() for language in languages
    }
    confusion_by_template: dict[str, dict[str, Counter[str]]] = {
        template: {language: Counter() for language in languages}
        for template in templates
    }

    for condition in conditions:
        group = grouped_by_variant[(condition.template, condition.variant)][
            condition.target
        ]
        if len(group) != len(video_order):
            raise ValueError(
                f"{condition.directory}: {len(group)} {condition.target!r} rows "
                f"but the test split has {len(video_order)} videos"
            )
        condition_hits = [label == condition.target for _, label in group]
        hits[(condition.template, condition.target, condition.distractor)] = (
            condition_hits
        )
        for _, label in group:
            confusion[condition.target][label] += 1
            confusion_by_template[condition.template][condition.target][label] += 1

        failures = [
            {
                "video": video_order[position],
                "detected": label,
                "prediction": row.get("prediction", ""),
                "reference": row.get("reference", ""),
            }
            for position, (row, label) in enumerate(group)
            if label != condition.target
        ]
        condition_reports.append(
            {
                "template": condition.template,
                "variant": condition.variant,
                "prompt_id": condition.prompt_id,
                "target": condition.target,
                "distractor": condition.distractor,
                "condition": condition.label,
                "num_samples": len(group),
                "lacc": _mean(condition_hits),
                "num_failures": len(failures),
                "failure_samples": failures[: args.samples] if args.samples else [],
            }
        )

    lacc_by_template = {
        template: _mean(
            [
                hit
                for (condition_template, _, _), values in hits.items()
                if condition_template == template
                for hit in values
            ]
        )
        for template in templates
    }
    lacc_by_template_language = {
        template: {
            language: _mean(
                [
                    hit
                    for (condition_template, target, _), values in hits.items()
                    if condition_template == template and target == language
                    for hit in values
                ]
            )
            for language in languages
        }
        for template in templates
    }
    lacc_by_pair = {
        f"{first}/{second}": _mean(
            [
                hit
                for (_, target, distractor), values in hits.items()
                if {target, distractor} == {first, second}
                for hit in values
            ]
        )
        for first, second in itertools.combinations(languages, 2)
    }

    summary: dict[str, Any] = {
        "step_dir": str(step_dir),
        "checkpoint_dir": checkpoint_dir,
        "templates": {
            template: DEFAULT_TEMPLATES.get(template, template)
            for template in templates
        },
        "languages": list(languages),
        "num_videos": len(video_order),
        "num_conditions": len(conditions),
        "detector": detector.model_name,
        "canonical": read_canonical(step_dir, args.canonical, languages),
        "conditions": condition_reports,
        "lacc": {
            "by_template": lacc_by_template,
            "by_template_language": lacc_by_template_language,
            "by_pair": lacc_by_pair,
            "overall": _mean([hit for values in hits.values() for hit in values]),
        },
        "bsa": compute_bsa(hits, templates, languages),
        "confusion": {
            language: dict(counter.most_common())
            for language, counter in confusion.items()
        },
        "confusion_by_template": {
            template: {
                language: dict(counter.most_common())
                for language, counter in per_language.items()
            }
            for template, per_language in confusion_by_template.items()
        },
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_commit": _git_commit(),
    }
    return summary


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _format(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def render_markdown(summary: dict[str, Any]) -> str:
    languages = tuple(summary["languages"])
    templates = tuple(summary["templates"])
    lines: list[str] = []

    lines.append("# Counterfactual instruction switching")
    lines.append("")
    lines.append(f"- step dir: `{summary['step_dir']}`")
    if summary["checkpoint_dir"]:
        lines.append(f"- checkpoint: `{summary['checkpoint_dir']}`")
    lines.append(
        f"- design: {summary['num_conditions']} conditions x "
        f"{summary['num_videos']} videos"
    )
    lines.append(f"- language detector: `{summary['detector']}`")
    if summary["canonical"]:
        lines.append(f"- canonical lacc from: `{summary['canonical']['source']}`")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    header = ["Canonical LAcc"] + [
        f"{summary['templates'][template].title()} LAcc" for template in templates
    ]
    canonical_lacc = (
        (summary["canonical"] or {}).get("lacc", {}).get("macro")
        if summary["canonical"]
        else None
    )
    values = [_format(canonical_lacc)] + [
        _format(summary["lacc"]["by_template"][template]) for template in templates
    ]
    lines.append("| " + " | ".join(header + ["BSA"]) + " |")
    lines.append("|" + "---|" * (len(header) + 1))
    lines.append(
        "| " + " | ".join(values + [_format(summary["bsa"]["overall"])]) + " |"
    )
    lines.append("")
    lines.append(
        f"BSA is counted over {summary['bsa']['num_pair_instances']} "
        "(template, language pair, video) instances; a pair counts only when "
        "both directions produce the requested language."
    )
    lines.append("")

    lines.append("## LAcc per condition")
    lines.append("")
    lines.append("| Template | Variant | Target | Distractor | Prompt | n | LAcc |")
    lines.append("|---|---|---|---|---|---:|---:|")
    for entry in summary["conditions"]:
        lines.append(
            f"| {entry['template']} | {entry['variant']} | {entry['target']} | "
            f"{entry['distractor']} | `{entry['prompt_id']}` | "
            f"{entry['num_samples']} | {_format(entry['lacc'])} |"
        )
    lines.append("")

    lines.append("## LAcc per template and target language")
    lines.append("")
    lines.append("| Template | " + " | ".join(languages) + " | macro |")
    lines.append("|---|" + "---:|" * (len(languages) + 1))
    for template in templates:
        per_language = summary["lacc"]["by_template_language"][template]
        lines.append(
            f"| {template} | "
            + " | ".join(_format(per_language[language]) for language in languages)
            + f" | {_format(summary['lacc']['by_template'][template])} |"
        )
    lines.append("")

    lines.append("## BSA per language pair")
    lines.append("")
    pairs = sorted(summary["bsa"]["by_pair"])
    lines.append("| Template | " + " | ".join(pairs) + " | all |")
    lines.append("|---|" + "---:|" * (len(pairs) + 1))
    for template in templates:
        per_pair = summary["bsa"]["by_template_pair"][template]
        lines.append(
            f"| {template} | "
            + " | ".join(_format(per_pair.get(pair)) for pair in pairs)
            + f" | {_format(summary['bsa']['by_template'][template])} |"
        )
    lines.append(
        "| all | "
        + " | ".join(_format(summary["bsa"]["by_pair"][pair]) for pair in pairs)
        + f" | {_format(summary['bsa']['overall'])} |"
    )
    lines.append("")

    lines.append("## Target-language confusion")
    lines.append("")
    lines.append("Rows are the requested language, columns the detected one.")
    lines.append("")
    detected = sorted(
        {label for counts in summary["confusion"].values() for label in counts}
    )
    lines.append("| requested | " + " | ".join(detected) + " |")
    lines.append("|---|" + "---:|" * len(detected))
    for language in languages:
        counts = summary["confusion"][language]
        lines.append(
            f"| {language} | "
            + " | ".join(str(counts.get(label, 0)) for label in detected)
            + " |"
        )
    lines.append("")

    if any(entry["failure_samples"] for entry in summary["conditions"]):
        lines.append("## Failed predictions")
        lines.append("")
        for entry in summary["conditions"]:
            if not entry["failure_samples"]:
                continue
            lines.append(
                f"### {entry['template']} / {entry['condition']} "
                f"({entry['num_failures']} of {entry['num_samples']} failed)"
            )
            lines.append("")
            for sample in entry["failure_samples"]:
                lines.append(
                    f"- `{sample['video']}` detected **{sample['detected']}**: "
                    f"{sample['prediction']!r}"
                )
            lines.append("")

    lines.append("## Main table row")
    lines.append("")
    lines.append("```latex")
    lines.append(
        "  -- & -- & "
        + " & ".join(values + [_format(summary["bsa"]["overall"])])
        + r" \\"
    )
    lines.append("```")
    lines.append("")
    lines.append(f"Generated {summary['generated_at']} at {summary['git_commit']}.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args)

    output_json = args.output_json or args.step_dir / "counterfactual_summary.json"
    output_md = args.output_md or args.step_dir / "counterfactual_summary.md"
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    # One line of headline so a sweep log says what happened without opening
    # the report; everything else lives in the two files above.
    headline = "  ".join(
        f"{template} lacc={_format(value)}"
        for template, value in summary["lacc"]["by_template"].items()
    )
    print(f"{headline}  BSA={_format(summary['bsa']['overall'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
