#! /bin/bash
#
# Usage:
#   bash scripts/eval/run_eval_cf_last_prompt.sh <CKPT_DIR>          # 2 variants
#   bash scripts/eval/run_eval_cf_last_prompt.sh <CKPT_DIR> share    # in-repo dataset
#   bash scripts/eval/run_eval_cf_last_prompt.sh <CKPT_DIR> dry-run  # preview only
#
# <CKPT_DIR> first, then order-free keywords: share, dry-run. Always de+en+zh.
#
# Environment: see scripts/eval/run_eval.sh (EVAL_NUM_PROCESSES, FORCE, ...)
#   PROMPT_VARIANTS="cf_last_001" runs a subset.
#   EVAL_LANGUAGE_DISTRIBUTION=0 skips the output-language detection pass.
#   EVAL_OUTPUT_ROOT overrides outputs/eval.
#
# Help: scripts/eval/run_eval_cf_last_prompt.sh --help
#
# Counterfactual instruction switching, target named last:
#
#   Do not translate the signing into {DISTRACTOR}; use {TARGET}.
#
# The mirror image of run_eval_cf_first_prompt.sh: the same language pairs and
# the same two directions, with the target moved to the end of the instruction.
# Running both halves is what makes position a control rather than a confound --
# a model that simply answers in the first language name it sees scores near 1.0
# on cf_first and near 0.0 here, and a model that answers in the last one scores
# the other way round. Only instruction semantics score high on both.
#
# See run_eval_cf_first_prompt.sh for the shared design notes (neutral template
# tail, held-out status, why this suite is not diagnostic).
#
#   outputs/eval/<run>/<step>/cf_last/{cf_last_001,cf_last_002}/

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_prompt_suite.sh"

# lacc is the primary reading here, so its confusion matrix comes along by
# default even though this suite is not diagnostic.
: "${EVAL_LANGUAGE_DISTRIBUTION:=1}"

run_prompt_suite cf_last prompts/generic/counterfactual_last.jsonl cf_last false false -- "$@"
