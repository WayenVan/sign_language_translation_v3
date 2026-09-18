#! /bin/bash
#
# Usage:
#   bash scripts/eval/run_eval_wrong_task_prompt.sh <CKPT_DIR>          # 1 variant
#   bash scripts/eval/run_eval_wrong_task_prompt.sh <CKPT_DIR> share    # in-repo dataset
#   bash scripts/eval/run_eval_wrong_task_prompt.sh <CKPT_DIR> dry-run  # preview only
#
# <CKPT_DIR> first, then order-free keywords: share, dry-run. Always de+en+zh.
#
# Environment: see scripts/eval/run_eval.sh (EVAL_NUM_PROCESSES, FORCE, ...)
#   EVAL_LANGUAGE_DISTRIBUTION=0 skips the output-language detection pass.
#   EVAL_OUTPUT_ROOT overrides outputs/eval.
#
# Help: scripts/eval/run_eval_wrong_task_prompt.sh --help
#
# Adversarial instruction diagnostic: the prompt asks the model to describe the
# signer's clothing in the target language instead of translating the signing.
# BLEU/ROUGE against the translation reference therefore measure nothing and the
# summary files mark them appendix-only; the reading is target-language accuracy
# plus the output-language distribution (refine-logs/PROMPT_PROTOCOL.md).
#
#   outputs/eval/<run>/<step>/wrong_task/wrong_task_001/

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_prompt_suite.sh"

run_prompt_suite wrong_task prompts/generic/wrong_task.jsonl wrong_task true false -- "$@"
