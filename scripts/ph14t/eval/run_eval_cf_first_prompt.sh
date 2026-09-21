#! /bin/bash
#
# Usage:
#   bash scripts/ph14t/eval/run_eval_cf_first_prompt.sh <CKPT_DIR>          # 2 variants
#   bash scripts/ph14t/eval/run_eval_cf_first_prompt.sh <CKPT_DIR> share    # in-repo dataset
#   bash scripts/ph14t/eval/run_eval_cf_first_prompt.sh <CKPT_DIR> dry-run  # preview only
#
# <CKPT_DIR> first, then order-free keywords: share, dry-run. Always de+en+zh.
#
# Environment: see scripts/ph14t/eval/run_eval.sh (EVAL_NUM_PROCESSES, FORCE, ...)
#   PROMPT_VARIANTS="cf_first_001" runs a subset.
#   EVAL_LANGUAGE_DISTRIBUTION=0 skips the output-language detection pass.
#   EVAL_OUTPUT_ROOT overrides outputs/eval.
#
# Help: scripts/ph14t/eval/run_eval_cf_first_prompt.sh --help
#
# Counterfactual instruction switching, target named first:
#
#   Translate the signing into {TARGET}, not {DISTRACTOR}.
#
# Every prompt names exactly two languages, exactly once each, so a counterfactual
# pair -- "German, not English" against "English, not German" -- differs only in
# which language plays the target role. Keyword detection alone therefore cannot
# separate the two directions; only the instruction's semantics can. The tail of
# the template drops the language name the canonical prompts repeat there
# ("only the German translation" -> "only the translation"), because repeating it
# would make the target both the most frequent and the last-mentioned language
# name and hand the experiment away.
#
# This suite is the "target first" half. run_eval_cf_last_prompt.sh is the
# "target last" half, and the two together cover 3 language pairs x 2 directions
# x 2 templates = 12 conditions per video. Bidirectional Switch Accuracy pairs
# the two directions across variants and is computed from the predictions of
# both suites, not from either summary alone.
#
# BLEU/ROUGE here are scored against the reference in the requested target
# language and do count, so the suite is not diagnostic; the reading is
# target-language accuracy plus the output-language distribution
# (refine-logs/PROMPT_PROTOCOL.md, "adversarial prompts": negated distractor).
#
# This is a held-out diagnostic bank: never train on it, never select
# checkpoints or hyperparameters with it.
#
#   outputs/eval/<run>/<step>/cf_first/{cf_first_001,cf_first_002}/

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_prompt_suite.sh"

# lacc is the primary reading here, so its confusion matrix comes along by
# default even though this suite is not diagnostic.
: "${EVAL_LANGUAGE_DISTRIBUTION:=1}"

run_prompt_suite cf_first prompts/generic/counterfactual_first.jsonl cf_first false false -- "$@"
