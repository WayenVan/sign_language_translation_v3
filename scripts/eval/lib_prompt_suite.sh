#! /bin/bash
#
# Shared driver for the prompt-suite launchers in this directory. Not meant to
# be run directly: scripts/eval/run_eval_*_prompt.sh source it and call
#
#   run_prompt_suite <suite> <bank> <ID families> <diagnostic> <allow languages> -- "$@"
#
#   suite            output subdirectory name (fixed, diverse, unseen, ...)
#   bank             prompt bank JSONL, relative to the project directory
#   families         comma-separated prompt-ID families to run, in order
#   diagnostic       true for suites whose instruction is not "translate this",
#                    so BLEU against the translation reference means nothing
#   allow languages  true if the suite accepts de|en|zh|multi (fixed only);
#                    the paraphrase suites are multilingual by construction
#
# One suite = one output directory holding one subdirectory per prompt variant,
# plus the variants.tsv saying which variants the suite is made of and the
# summary.{json,md} aggregated from them:
#
#   outputs/eval/<run>/<checkpoint-step>/<suite>/
#       variants.tsv
#       <variant>/predictions.jsonl, prompts.jsonl, predictions_metrics.json,
#                 eval_config.yaml
#       summary.json, summary.md
#
# Within one variant all three target languages run the same instruction
# variant, which is what makes "mean +- std across prompt variants"
# (refine-logs/PROMPT_PROTOCOL.md) a statement about prompt wording instead of
# a mix of wording and language.
#
# Variants already holding predictions_metrics.json are skipped (FORCE=1
# re-runs them), and a variant that fails does not abort the rest of the suite.

set -euo pipefail

PROJECT_DIR=/users/2533494w/projects/sign_language_translation_v3
if [[ ! -d "$PROJECT_DIR" ]]; then
  # tubbs checkout; scripts/eval/run_eval.sh resolves the same two hosts.
  PROJECT_DIR=/home/2533494W/project/sign_language_translation_v3
fi
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

suite_usage() {
  sed -n '/^# Usage:$/,/^# Help:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

run_suite_python() {
  (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_PYTHON" "$@")
}

run_prompt_suite() {
  local suite="$1" bank="$2" families="$3" diagnostic="$4" allow_languages="$5"
  shift 5
  [[ "${1:-}" == "--" ]] && shift

  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    suite_usage
    exit 0
  fi

  local worker="$PROJECT_DIR/scripts/eval/run_eval.sh"
  if [[ ! -f "$worker" ]]; then
    echo "eval worker not found: $worker" >&2
    exit 3
  fi
  if [[ ! -f "$PROJECT_DIR/$bank" ]]; then
    echo "prompt bank not found: $PROJECT_DIR/$bank" >&2
    exit 3
  fi

  local checkpoint_dir="${1:-}"
  if [[ -z "$checkpoint_dir" ]]; then
    echo "a checkpoint directory is required as the first argument." >&2
    suite_usage
    exit 2
  fi
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "checkpoint directory not found: $checkpoint_dir" >&2
    exit 3
  fi
  checkpoint_dir="$(cd "$checkpoint_dir" && pwd)"
  shift

  local dry_run=false
  local lang_tag=multi
  local extra_worker_args=()
  local arg
  for arg in "$@"; do
    case "$arg" in
    share) extra_worker_args+=(share) ;;
    dry-run | --dry-run)
      dry_run=true
      extra_worker_args+=(dry-run)
      ;;
    de | en | zh)
      if [[ "$allow_languages" != true ]]; then
        # These suites paraphrase one and the same instruction, so they only
        # say something against a checkpoint with three target languages to
        # keep apart -- the same reason scripts/run_llm_lora_inject.sh refuses
        # single-language diverse training.
        echo "$suite evaluates de+en+zh jointly; single-language runs are not supported." >&2
        exit 2
      fi
      lang_tag="$arg"
      ;;
    multi | multilang) lang_tag=multi ;;
    *)
      echo "Unknown argument: $arg (supported:$([[ "$allow_languages" == true ]] && echo " de, en, zh, multi,") share, dry-run)" >&2
      exit 2
      ;;
    esac
  done

  # Pre-flight the checkpoint's training language once. run_eval.sh performs
  # the authoritative check per run, but a multilingual-only suite pointed at a
  # single-language checkpoint would otherwise print the same refusal once per
  # prompt variant.
  local ckpt_language=unknown
  if [[ -f "$checkpoint_dir/hydra_config.yaml" ]]; then
    ckpt_language="$(run_suite_python -c \
      'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])) or {}; print((cfg.get("data") or {}).get("language") or "multilang")' \
      "$checkpoint_dir/hydra_config.yaml")"
  fi
  if [[ "$ckpt_language" != unknown && "$ckpt_language" != multilang &&
    "$lang_tag" != "$ckpt_language" ]]; then
    if [[ "${FORCE_LANG:-0}" == "1" ]]; then
      echo "WARNING: FORCE_LANG=1; running the $suite suite ($lang_tag) against a $ckpt_language-only checkpoint." >&2
    else
      echo "checkpoint/suite mismatch: $checkpoint_dir was trained with language=$ckpt_language, but the $suite suite evaluates $lang_tag. Set FORCE_LANG=1 to override." >&2
      exit 5
    fi
  fi

  local ckpt_step ckpt_run_name suite_dir suite_dir_abs summarizer_languages
  ckpt_step="$(basename "$checkpoint_dir")"
  ckpt_run_name="$(basename "$(dirname "$checkpoint_dir")")"
  suite_dir="${EVAL_OUTPUT_ROOT:-outputs/eval}/${ckpt_run_name}/${ckpt_step}/${suite}"
  summarizer_languages=de,en,zh
  if [[ "$allow_languages" == true ]]; then
    # Keep one directory per evaluated language: a de-only run and a joint run
    # are different measurements of the same checkpoint.
    suite_dir="$suite_dir/$lang_tag"
    [[ "$lang_tag" != multi ]] && summarizer_languages="$lang_tag"
  fi

  # The variant list comes from the bank itself, and is a hard error when the
  # three languages do not offer the same variants.
  local variant_lines
  variant_lines="$(run_suite_python -m csi_slt.commands.list_prompt_variants \
    --prompt-bank "$bank" --families "$families")"

  # PROMPT_VARIANTS runs or previews a subset: PROMPT_VARIANTS="diverse_003".
  local wanted="${PROMPT_VARIANTS:-}"
  local selected=()
  local line variant
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    variant="${line%%$'\t'*}"
    if [[ -n "$wanted" && " $wanted " != *" $variant "* ]]; then
      continue
    fi
    selected+=("$line")
  done <<<"$variant_lines"

  if ((${#selected[@]} == 0)); then
    echo "no prompt variant of $bank matches PROMPT_VARIANTS='$wanted'" >&2
    exit 2
  fi

  # EVAL_OUTPUT_ROOT may be absolute; run_eval.sh and the summarizer both run
  # with the project directory as their working directory, so the relative form
  # is what they are handed either way.
  if [[ "$suite_dir" == /* ]]; then
    suite_dir_abs="$suite_dir"
  else
    suite_dir_abs="$PROJECT_DIR/$suite_dir"
  fi

  echo "Suite: $suite   languages: $lang_tag"
  echo "Checkpoint: $checkpoint_dir"
  echo "Prompt bank: $bank (families: $families)"
  echo "Variants to run: ${#selected[@]} of $(grep -c . <<<"$variant_lines")"
  echo "Suite dir: $suite_dir_abs"
  echo

  if [[ "$dry_run" != true ]]; then
    mkdir -p "$suite_dir_abs"
    # Record the whole suite, not the selected subset: this is what the suite
    # consists of, and summarize_prompt_suite reports the rest as missing.
    {
      echo "# bank=$bank"
      echo "# families=$families"
      echo "# languages=de,en,zh"
      echo "$variant_lines"
    } >"$suite_dir_abs/variants.tsv"
  fi

  local de_id en_id zh_id status failed=0
  for line in "${selected[@]}"; do
    IFS=$'\t' read -r variant de_id en_id zh_id <<<"$line"
    echo "--- $suite/$variant: de=$de_id en=$en_id zh=$zh_id"
    status=0
    EVAL_OUTPUT_DIR="$suite_dir/$variant" \
      EVAL_PROMPT_BANK="$bank" \
      EVAL_PROMPT_IDS="de=$de_id,en=$en_id,zh=$zh_id" \
      bash "$worker" "$checkpoint_dir" "$lang_tag" "${extra_worker_args[@]+"${extra_worker_args[@]}"}" || status=$?
    if ((status != 0)); then
      failed=$((failed + 1))
      echo "FAILED: $suite/$variant (exit $status); continuing with the remaining variants." >&2
    fi
    echo
  done

  local summarize_command=(-m csi_slt.commands.summarize_prompt_suite "$suite_dir"
    --languages "$summarizer_languages")
  if [[ "$diagnostic" == true ]]; then
    summarize_command+=(--diagnostic)
  fi

  # The output-language distribution is the confusion matrix behind lacc, and is
  # independent of whether the suite's BLEU means anything: a diagnostic suite
  # wants it because lacc is its only real metric, and the counterfactual suites
  # want it alongside a BLEU that does count. Hence the default follows
  # $diagnostic and EVAL_LANGUAGE_DISTRIBUTION overrides it either way.
  local want_distribution="$diagnostic"
  case "${EVAL_LANGUAGE_DISTRIBUTION:-}" in
  1 | true) want_distribution=true ;;
  0 | false) want_distribution=false ;;
  "") ;;
  *)
    echo "EVAL_LANGUAGE_DISTRIBUTION must be 0 or 1, got '${EVAL_LANGUAGE_DISTRIBUTION}'" >&2
    exit 2
    ;;
  esac
  if [[ "$want_distribution" == true ]]; then
    summarize_command+=(--language-distribution)
  fi

  if [[ "$dry_run" == true ]]; then
    echo "Dry run only. Omit dry-run to evaluate; the summary step would be:"
    printf '  python'
    printf ' %q' "${summarize_command[@]}"
    printf '\n'
    return 0
  fi

  echo "Summarizing $suite_dir"
  run_suite_python "${summarize_command[@]}"

  if ((failed > 0)); then
    echo "$failed of ${#selected[@]} variants failed; see the warnings above." >&2
    return 1
  fi
}
