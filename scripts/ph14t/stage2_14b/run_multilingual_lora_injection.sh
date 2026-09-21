#! /bin/bash
#
# Inject q/k/v/o LoRA into the best Qwen3-14B multilingual stage-1 checkpoints.
# Runs the diverse-prompt and fixed-prompt models as two independent jobs.
#
# Usage:
#   bash scripts/ph14t/stage2_14b/run_multilingual_lora_injection.sh             # sequential
#   bash scripts/ph14t/stage2_14b/run_multilingual_lora_injection.sh --sbatch    # submit 2 Slurm jobs
#   bash scripts/ph14t/stage2_14b/run_multilingual_lora_injection.sh --dry-run   # preview only
#
# Environment overrides:
#   RANK=768 EPOCHS=11 EXTRA_ARGS="share"
#   DIVERSE_CHECKPOINT_STEP=checkpoint-180000 FIXED_CHECKPOINT_STEP=checkpoint-126000

set -euo pipefail

PROJECT_DIR=/users/2533494w/projects/sign_language_translation_v3
CHILD_SCRIPT="$PROJECT_DIR/scripts/ph14t/run_llm_lora_inject.sh"
OUTPUT_ROOT="$PROJECT_DIR/outputs/v5.0-14b-final-ckpts"

RANK="${RANK:-768}"
EPOCHS="${EPOCHS:-11}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_ROOTS=(
  "$OUTPUT_ROOT/v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep40-multilang-diverse-0911.224x224"
  "$OUTPUT_ROOT/v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep40-multilang-fixed-0911.224x224"
)
PROMPT_MODES=(diverse fixed)
CHECKPOINT_STEPS=(
  "${DIVERSE_CHECKPOINT_STEP:-checkpoint-180000}"
  "${FIXED_CHECKPOINT_STEP:-checkpoint-126000}"
)

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
  echo "LoRA launcher not found: $CHILD_SCRIPT" >&2
  exit 3
fi
if [[ ! "$RANK" =~ ^[0-9]+$ ]] || (( 10#$RANK < 1 )); then
  echo "RANK must be a positive integer, got: $RANK" >&2
  exit 2
fi
if [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || (( 10#$EPOCHS < 1 )); then
  echo "EPOCHS must be a positive integer, got: $EPOCHS" >&2
  exit 2
fi
for checkpoint_step in "${CHECKPOINT_STEPS[@]}"; do
  if [[ ! "$checkpoint_step" =~ ^checkpoint-[0-9]+$ ]]; then
    echo "Checkpoint steps must look like checkpoint-180000, got: $checkpoint_step" >&2
    exit 2
  fi
done
read -r -a CHILD_EXTRA_ARGS <<< "$EXTRA_ARGS"

ALPHA=$((2 * 10#$RANK))
echo "Mode: $MODE"
echo "Jobs: 2 multilingual runs (diverse prompt, fixed prompt)"
echo "Languages: de+en+zh joint training"
echo "LoRA: targets=q_proj,k_proj,v_proj,o_proj; layers=all-40; rank=$RANK; alpha=$ALPHA"
echo "Training: epochs=$EPOCHS; llm_lr=${LLM_LORA_LR:-1e-4}; GPUs=2"
echo "Checkpoint steps: diverse=${CHECKPOINT_STEPS[0]}; fixed=${CHECKPOINT_STEPS[1]}"
echo "Output root: $OUTPUT_ROOT"
echo "Extra child arguments: ${EXTRA_ARGS:-none}"

for index in "${!PROMPT_MODES[@]}"; do
  prompt_mode="${PROMPT_MODES[$index]}"
  checkpoint_dir="${RUN_ROOTS[$index]}/${CHECKPOINT_STEPS[$index]}"
  job_name="slt_14b_llora_multi_${prompt_mode}_r${RANK}a${ALPHA}"
  child_args=("$checkpoint_dir" "$RANK" multi "$prompt_mode")
  child_args+=("$EPOCHS")
  child_args+=("${CHILD_EXTRA_ARGS[@]}")

  if [[ "$MODE" == sbatch ]]; then
    command=(sbatch --partition=gpu-h100 --gres=gpu:2 --time=2-12:00:00 \
      -J "$job_name" --export="ALL,LLM_LORA_OUTPUT_ROOT=$OUTPUT_ROOT" \
      "$CHILD_SCRIPT" "${child_args[@]}")
  else
    command=(env "LLM_LORA_OUTPUT_ROOT=$OUTPUT_ROOT" bash "$CHILD_SCRIPT" "${child_args[@]}")
  fi

  printf '[%s] checkpoint: %s\n' "$prompt_mode" "$checkpoint_dir"
  printf '[%s] command: ' "$prompt_mode"
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$MODE" != dry-run ]]; then
    "${command[@]}"
  fi
done

if [[ "$MODE" == dry-run ]]; then
  echo "Dry run only. Omit --dry-run to execute sequentially, or use --sbatch."
fi
