#! /bin/bash
#
# Sweep wrapper for scripts/run_llm_lora_ablation.sh.
#
# Sweep dimensions (Cartesian product):
#   TARGETS -- qv or qkvo attention projections
#   RANKS   -- LoRA rank; the child launcher derives alpha = 2 * rank
# Fixed parameter:
#   depth   -- all 36 Qwen3-4B decoder blocks
#   EPOCHS  -- empty by default, so the child config keeps its default (12)
#
# Usage:
#   bash scripts/swaps/run_llm_lora_ablation_sweep.sh             # run sequentially
#   bash scripts/swaps/run_llm_lora_ablation_sweep.sh --sbatch    # submit to Slurm
#   bash scripts/swaps/run_llm_lora_ablation_sweep.sh --dry-run   # preview only
#
# Environment overrides (space-separated lists):
#   TARGETS="qv qkvo" RANKS="8 16" EPOCHS=20 \
#     bash scripts/swaps/run_llm_lora_ablation_sweep.sh
#
# Optional child flags:
#   EXTRA_ARGS="share"  (or "debug share" when running the child directly)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHILD_SCRIPT="$SCRIPT_DIR/run_llm_lora_ablation.sh"

TARGETS="${TARGETS:-qv qkvo}"
RANKS="${RANKS:-8 16}"
EPOCHS="${EPOCHS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
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

read -r -a TARGET_VALUES <<< "$TARGETS"
read -r -a RANK_VALUES <<< "$RANKS"
read -r -a CHILD_EXTRA_ARGS <<< "$EXTRA_ARGS"

if (( ${#TARGET_VALUES[@]} == 0 || ${#RANK_VALUES[@]} == 0 )); then
  echo "TARGETS and RANKS must each contain at least one value." >&2
  exit 2
fi

if [[ -n "$EPOCHS" ]] && { [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || (( EPOCHS < 1 )); }; then
  echo "EPOCHS must be empty or a positive integer, got: $EPOCHS" >&2
  exit 2
fi

job_count=$(( ${#TARGET_VALUES[@]} * ${#RANK_VALUES[@]} ))
echo "Mode: $MODE"
echo "Sweep: TARGETS={${TARGETS// /,}} x RANK={${RANKS// /,}} ($job_count jobs)"
echo "Fixed: all 36 layers EPOCHS=${EPOCHS:-config-default-12} EXTRA_ARGS=${EXTRA_ARGS:-none}"

for targets in "${TARGET_VALUES[@]}"; do
  if [[ "$targets" != qv && "$targets" != qkvo ]]; then
    echo "Each TARGETS value must be qv or qkvo, got: $targets" >&2
    exit 2
  fi

  for rank in "${RANK_VALUES[@]}"; do
    if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( rank < 1 )); then
      echo "Each RANK must be a positive integer, got: $rank" >&2
      exit 2
    fi

    alpha=$(( 2 * rank ))
    job_name="slt_llora_de_all_${targets}_r${rank}a${alpha}"
    child_args=("$targets" "$rank")
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
done

if [[ "$MODE" == dry-run ]]; then
  echo "Dry run only. Omit --dry-run to execute sequentially, or use --sbatch."
fi
