#! /bin/bash
#
# Usage:
#   bash scripts/eval/run_eval_fixed_prompt.sh <CKPT_DIR>              # de+en+zh, canonical prompt
#   bash scripts/eval/run_eval_fixed_prompt.sh <CKPT_DIR> de           # one target language
#   bash scripts/eval/run_eval_fixed_prompt.sh <CKPT_DIR> multi share  # in-repo dataset
#   bash scripts/eval/run_eval_fixed_prompt.sh <CKPT_DIR> dry-run      # preview only
#
# <CKPT_DIR> first, then order-free keywords: de|en|zh|multi (default multi),
# share, dry-run.
#
# Environment: see scripts/eval/run_eval.sh (EVAL_NUM_PROCESSES, FORCE, ...)
#   EVAL_OUTPUT_ROOT overrides outputs/eval.
#
# Help: scripts/eval/run_eval_fixed_prompt.sh --help
#
# Standard translation quality: every model is scored on the same canonical
# prompt per target language, which is the condition behind the main
# Mono-versus-Unified tables (refine-logs/PROMPT_PROTOCOL.md). This is the only
# suite that accepts a single target language.
#
#   outputs/eval/<run>/<step>/fixed/<multi|de|en|zh>/canonical_001/

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_prompt_suite.sh"

run_prompt_suite fixed prompts/generic/train.jsonl canonical false true -- "$@"
