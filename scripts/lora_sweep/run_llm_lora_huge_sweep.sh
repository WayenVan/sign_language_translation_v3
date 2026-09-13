#! /bin/bash
#
# Usage:
#   bash scripts/swaps/run_llm_lora_huge_sweep.sh             # run sequentially
#   bash scripts/swaps/run_llm_lora_huge_sweep.sh --sbatch    # submit to Slurm
#   bash scripts/swaps/run_llm_lora_huge_sweep.sh --dry-run   # preview only
#
# Environment overrides:
#   RANKS="256 512" EPOCHS=20 \
#     bash scripts/swaps/run_llm_lora_huge_sweep.sh
#
# Optional child flags:
#   EXTRA_ARGS="share"  (or "debug share" when running the child directly)
#
# Huge-rank qkvo sweep wrapper for scripts/run_llm_lora_ablation.sh.
#
# Sweep dimension:
#   RANKS   -- LoRA rank: 256 and 512 by default
# Fixed parameters:
#   targets -- qkvo attention projections
#   depth   -- all 36 Qwen3-4B decoder blocks
#   alpha   -- derived by the child launcher as 2 * rank (scale alpha/r = 2,
#              the same as every earlier rank in this family)
#   EPOCHS  -- empty by default, so the child config keeps its default (12)

set -euo pipefail

# --------------------------------------------------------------------------- #
# Why go this far: dev BLEU-4 has risen monotonically with rank across the
# qkvo runs so far -- r8 0.1860, r16 0.1876, r32 0.1894, r64 0.1976,
# r128 0.2034 -- with no sign of a plateau yet.
#
# Size: qkvo LoRA costs r * 20,480 parameters per block (q 2560->4096,
# k/v 2560->1024, o 4096->2560), r * 737,280 over 36 blocks:
#   r=256 -> 188.7M    r=512 -> 377.5M    (Qwen3-4B itself: ~4.0B)
# At r=512 the k/v adapters are half their matrices' full rank (1024), so this
# end of the sweep is close to full fine-tuning of k_proj/v_proj.
#
# Resources: r16 / r32 ran on H100 NVL (node12) in ~7.2 h, peaking at
# 31.2 GiB per GPU. r=512 adds up to ~6 GB of fp32 LoRA weights, gradients and
# Adam state, which fits an H100 but is tight on the child's default gpu-l40s
# partition (~45 GiB usable).
# --------------------------------------------------------------------------- #

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHILD_SCRIPT="$SCRIPT_DIR/run_llm_lora_ablation.sh"

RANKS="${RANKS:-256 512}"
EPOCHS="${EPOCHS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  # The comment block from line 2 to the first blank line.
  sed -n '2,/^$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
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

# Validate every rank before launching any, so a typo cannot leave a
# half-submitted sweep behind.
for rank in "${RANK_VALUES[@]}"; do
  if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( rank < 1 )); then
    echo "Each RANK must be a positive integer, got: $rank" >&2
    exit 2
  fi
done

echo "Mode: $MODE"
echo "Sweep: fixed TARGETS=qkvo x RANK={${RANKS// /,}} (${#RANK_VALUES[@]} jobs)"
echo "Fixed: all 36 layers EPOCHS=${EPOCHS:-config-default-12} EXTRA_ARGS=${EXTRA_ARGS:-none}"

for rank in "${RANK_VALUES[@]}"; do
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
