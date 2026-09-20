#! /bin/bash
#
# Usage:
#   bash scripts/eval/run_eval_unseen_prompt.sh <CKPT_DIR>          # 8 variants
#   bash scripts/eval/run_eval_unseen_prompt.sh <CKPT_DIR> share    # in-repo dataset
#   bash scripts/eval/run_eval_unseen_prompt.sh <CKPT_DIR> dry-run  # preview only
#
# <CKPT_DIR> first, then order-free keywords: share, dry-run. Always de+en+zh.
#
# Environment: see scripts/eval/run_eval.sh (EVAL_NUM_PROCESSES, FORCE, ...)
#   PROMPT_VARIANTS="heldout_002" runs a subset.
#   EVAL_OUTPUT_ROOT overrides outputs/eval.
#
# Help: scripts/eval/run_eval_unseen_prompt.sh --help
#
# Held-out instruction generalization: prompts/generic/heldout.jsonl is absent
# from every training and validation prompt pool, so this is the final
# unseen-instruction measurement and must not be used to pick checkpoints or
# hyperparameters (refine-logs/PROMPT_PROTOCOL.md).
#
#   outputs/eval/<run>/<step>/unseen/heldout_001..heldout_008/

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_prompt_suite.sh"

run_prompt_suite unseen prompts/generic/heldout.jsonl heldout false false -- "$@"
