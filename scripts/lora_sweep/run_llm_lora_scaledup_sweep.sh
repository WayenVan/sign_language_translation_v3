#! /bin/bash
#
# Usage:
#   bash scripts/lora_sweep/run_llm_lora_scaledup_sweep.sh             # sequential
#   bash scripts/lora_sweep/run_llm_lora_scaledup_sweep.sh --sbatch    # submit to Slurm
#   bash scripts/lora_sweep/run_llm_lora_scaledup_sweep.sh --dry-run   # preview only
#
# Qwen3-14B scaled-up sweep. It continues from the checkpoint-48000 in the
# requested stage-1 run and injects q/k/v/o LoRA into all 40 decoder blocks.
# The only sweep dimension is rank: 256 and 512. Alpha remains 2 * rank.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHILD_SCRIPT="$SCRIPT_DIR/run_llm_lora_ablation.sh"

export LLM_LORA_CKPT_LEAF="v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-de-0911.224x224/checkpoint-48000"
export LLM_LORA_CHECKPOINT_DIR="/users/2533494w/projects/sign_language_translation_v3/outputs/${LLM_LORA_CKPT_LEAF}"
export LLM_LORA_CKPT_TAG="ckpt48k"
export LLM_LORA_MODEL_RUN_SLUG="qwen3-14b-cradio-l-nextframe-handroi-cls-31m"
export LLM_LORA_MODEL_WANDB_TAG="qwen3-14b"
export LLM_LORA_LAYER_COUNT=40
export LLM_LORA_OUTPUT_DATE_TAG=0913

RANKS=(256 512)
EPOCHS="${EPOCHS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
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
if [[ -n "$EPOCHS" ]] && { [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || (( EPOCHS < 1 )); }; then
  echo "EPOCHS must be empty or a positive integer, got: $EPOCHS" >&2
  exit 2
fi
read -r -a CHILD_EXTRA_ARGS <<< "$EXTRA_ARGS"

echo "Mode: $MODE"
echo "Sweep: fixed TARGETS=qkvo x RANK={256,512} (2 jobs)"
echo "Fixed: Qwen3-14B, all 40 layers, checkpoint-48000, EPOCHS=${EPOCHS:-config-default-12}"

for rank in "${RANKS[@]}"; do
  alpha=$((2 * rank))
  job_name="slt_14b_llora_de_all_qkvo_r${rank}a${alpha}"
  child_args=(qkvo "$rank")
  [[ -n "$EPOCHS" ]] && child_args+=("$EPOCHS")
  child_args+=("${CHILD_EXTRA_ARGS[@]}")

  if [[ "$MODE" == sbatch ]]; then
    # Command-line resource flags override the 4B launcher's SBATCH defaults.
    command=(sbatch --partition=gpu-h100 --gres=gpu:2 --time=2-12:00:00 \
      -J "$job_name" "$CHILD_SCRIPT" "${child_args[@]}")
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
