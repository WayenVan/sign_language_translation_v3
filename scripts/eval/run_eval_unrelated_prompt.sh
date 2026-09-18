#! /bin/bash
#
# Usage:
#   bash scripts/eval/run_eval_unrelated_prompt.sh <CKPT_DIR>          # 1 variant
#   bash scripts/eval/run_eval_unrelated_prompt.sh <CKPT_DIR> share    # in-repo dataset
#   bash scripts/eval/run_eval_unrelated_prompt.sh <CKPT_DIR> dry-run  # preview only
#
# <CKPT_DIR> first, then order-free keywords: share, dry-run. Always de+en+zh.
#
# Environment: see scripts/eval/run_eval.sh (EVAL_NUM_PROCESSES, FORCE, ...)
#   EVAL_LANGUAGE_DISTRIBUTION=0 skips the output-language detection pass.
#   EVAL_OUTPUT_ROOT overrides outputs/eval.
#
# Help: scripts/eval/run_eval_unrelated_prompt.sh --help
#
# Adversarial instruction diagnostic: the prompt asks a general-knowledge
# question in the target language and tells the model to ignore the video. There
# is no correct translation, so this suite is an appendix diagnostic only -- the
# reading is whether the output still lands in the requested language
# (refine-logs/PROMPT_PROTOCOL.md).
#
#   outputs/eval/<run>/<step>/unrelated/unrelated_001/

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_prompt_suite.sh"

run_prompt_suite unrelated prompts/generic/unrelated.jsonl unrelated true false -- "$@"
