#! /bin/bash
#
# Usage:
#   bash scripts/eval/run_eval_diverse_prompt.sh <CKPT_DIR>          # 8 variants
#   bash scripts/eval/run_eval_diverse_prompt.sh <CKPT_DIR> share    # in-repo dataset
#   bash scripts/eval/run_eval_diverse_prompt.sh <CKPT_DIR> dry-run  # preview only
#
# <CKPT_DIR> first, then order-free keywords: share, dry-run. Always de+en+zh.
#
# Environment: see scripts/eval/run_eval.sh (EVAL_NUM_PROCESSES, FORCE, ...)
#   PROMPT_VARIANTS="diverse_003 diverse_005" runs a subset.
#   EVAL_OUTPUT_ROOT overrides outputs/eval.
#
# Help: scripts/eval/run_eval_diverse_prompt.sh --help
#
# Seen-prompt robustness: the canonical prompt plus all seven diverse training
# paraphrases, one evaluation per variant, all three languages on the same
# variant within a run. canonical_001 is run here too rather than borrowed from
# the fixed suite, so all eight points in the mean +- std come from one
# identically configured sweep.
#
#   outputs/eval/<run>/<step>/diverse/{canonical_001,diverse_001..diverse_007}/

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_prompt_suite.sh"

run_prompt_suite diverse prompts/generic/train.jsonl canonical,diverse false false -- "$@"
