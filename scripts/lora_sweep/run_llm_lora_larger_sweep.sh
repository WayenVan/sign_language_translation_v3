#! /bin/bash
#
# Larger-rank qkvo sweep wrapper for scripts/run_llm_lora_ablation.sh.
#
# Sweep dimension:
#   RANKS   -- LoRA rank: 32, 64, and 128 by default
# Fixed parameters:
#   targets -- qkvo attention projections
#   depth   -- all 36 Qwen3-4B decoder blocks
#   alpha   -- derived by the child launcher as 2 * rank
#   EPOCHS  -- empty by default, so the child config keeps its default (12)
#
# Usage:
#   bash scripts/swaps/run_llm_lora_larger_sweep.sh             # run sequentially
#   bash scripts/swaps/run_llm_lora_larger_sweep.sh --sbatch    # submit to Slurm
#   bash scripts/swaps/run_llm_lora_larger_sweep.sh --dry-run   # preview only
#
# Environment overrides:
#   RANKS="32 64 128" EPOCHS=20 \
#     bash scripts/swaps/run_llm_lora_larger_sweep.sh
#
# Optional child flags:
#   EXTRA_ARGS="share"  (or "debug share" when running the child directly)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHILD_SCRIPT="$SCRIPT_DIR/run_llm_lora_ablation.sh"

RANKS="${RANKS:-32 64 128}"
EPOCHS="${EPOCHS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

MODE=direct
case "${1:-}" in
  "") ;;
  --sbatch) MODE=sbatch ;;
  --dry-run) MODE=dry-run ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown argument: $1 (supported: --sbatch, --dry-run)" >&2; exit 2 ;;
esac

if [[ ! -f "$CHILD_SCRIPT" ]]; then
  echo "Child launcher not found: $CHILD_SCRIPT" >&2
  exit 3
fi

read -r -a RANK_VALUES <<< "$RANKS"
read -r -a CHILD_EXTRA_ARGS <<< "$EXTRA_ARGS"

if (( ${#RANK_VALUES[@]} == 0 )); then
  echo "RANKS must contain at least one value." >&2
  exit 2
fi

if [[ -n "$EPOCHS" ]] && { [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || (( EPOCHS < 1 )); }; then
  echo "EPOCHS must be empty or a positive integer, got: $EPOCHS" >&2
  exit 2
fi

echo "Mode: $MODE"
echo "Sweep: fixed TARGETS=qkvo x RANK={${RANKS// /,}} (${#RANK_VALUES[@]} jobs)"
echo "Fixed: all 36 layers EPOCHS=${EPOCHS:-config-default-12} EXTRA_ARGS=${EXTRA_ARGS:-none}"

for rank in "${RANK_VALUES[@]}"; do
  if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( rank < 1 )); then
    echo "Each RANK must be a positive integer, got: $rank" >&2
    exit 2
  fi

  alpha=$(( 2 * rank ))
  job_name="slt_llora_de_all_qkvo_r${rank}a${alpha}"
  child_args=(qkvo "$rank")
  [[ -n "$EPOCHS" ]] && child_args+=("$EPOCHS")
  child_args+=("${CHILD_EXTRA_ARGS[@]}")

  if [[ "$MODE" == sbatch ]]; then
    command=(sbatch -J "$job_name" "$CHILD_SCRIPT" "${child_args[@]}")
  else
    command=(bash "$CHILD_SCRIPT" "${child_args[@]}")
  fi

  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$MODE" != dry-run ]]; then
    "${command[@]}"
  fi
done

if [[ "$MODE" == dry-run ]]; then
  echo "Dry run only. Omit --dry-run to execute sequentially, or use --sbatch."
fi
