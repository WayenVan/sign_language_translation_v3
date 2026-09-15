#! /bin/bash
#
# Inject q/k/v/o LoRA into the final single-language Qwen3-14B stage-1 models.
# German is temporarily disabled because it has already been trained.
# The active best checkpoints are en=42000 and zh=48000.
#
# Usage:
#   bash scripts/stage2_14b/run_single_language_lora_injection.sh             # sequential
#   bash scripts/stage2_14b/run_single_language_lora_injection.sh --sbatch    # submit 2 Slurm jobs
#   bash scripts/stage2_14b/run_single_language_lora_injection.sh --dry-run   # print defaults and commands
#
# Environment overrides:
#   RANK=768 EPOCHS=18 EXTRA_ARGS="share"
#   EN_CHECKPOINT_STEP=checkpoint-42000 ZH_CHECKPOINT_STEP=checkpoint-48000
#
# All runs use StableAdamW (`stable` is always passed to the child launcher),
# so outputs land in *-stableadamw dirs next to the earlier AdamW runs.

set -euo pipefail

PROJECT_DIR=/users/2533494w/projects/sign_language_translation_v3
CHILD_SCRIPT="$PROJECT_DIR/scripts/run_llm_lora_inject.sh"
OUTPUT_ROOT="$PROJECT_DIR/outputs/v5.0-14b-final-ckpts"

RANK="${RANK:-768}"
EPOCHS="${EPOCHS:-18}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_ROOTS=(
  # "$OUTPUT_ROOT/v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-de-0911.224x224"
  "$OUTPUT_ROOT/v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-en-0911.224x224"
  "$OUTPUT_ROOT/v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-zh-0911.224x224"
)
LANGUAGES=(
  # de
  en
  zh
)
CHECKPOINT_STEPS=(
  # "${DE_CHECKPOINT_STEP:-checkpoint-132000}"
  "${EN_CHECKPOINT_STEP:-checkpoint-42000}"
  "${ZH_CHECKPOINT_STEP:-checkpoint-48000}"
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
    echo "Checkpoint steps must look like checkpoint-132000, got: $checkpoint_step" >&2
    exit 2
  fi
done
read -r -a CHILD_EXTRA_ARGS <<< "$EXTRA_ARGS"

ALPHA=$((2 * 10#$RANK))
echo "Mode: $MODE"
echo "Jobs: 2 single-language runs (en, zh); de is temporarily disabled"
echo "LoRA: targets=q_proj,k_proj,v_proj,o_proj; layers=all-40; rank=$RANK; alpha=$ALPHA"
echo "Training: epochs=$EPOCHS; llm_lr=${LLM_LORA_LR:-1e-4}; optimizer=stable_adamw; prompt=fixed; GPUs=2"
echo "Checkpoint steps: en=${CHECKPOINT_STEPS[0]}; zh=${CHECKPOINT_STEPS[1]}"
echo "Output root: $OUTPUT_ROOT"
echo "Extra child arguments: ${EXTRA_ARGS:-none}"

for index in "${!LANGUAGES[@]}"; do
  language="${LANGUAGES[$index]}"
  checkpoint_dir="${RUN_ROOTS[$index]}/${CHECKPOINT_STEPS[$index]}"
  job_name="slt_14b_llora_${language}_qkvo_r${RANK}a${ALPHA}_stable"
  child_args=("$checkpoint_dir" "$RANK" "$language" "$EPOCHS")
  child_args+=(stable)
  child_args+=("${CHILD_EXTRA_ARGS[@]}")

  if [[ "$MODE" == sbatch ]]; then
    command=(sbatch --partition=gpu-h100 --gres=gpu:2 --time=2-12:00:00 \
      -J "$job_name" --export="ALL,LLM_LORA_OUTPUT_ROOT=$OUTPUT_ROOT" \
      "$CHILD_SCRIPT" "${child_args[@]}")
  else
    command=(env "LLM_LORA_OUTPUT_ROOT=$OUTPUT_ROOT" bash "$CHILD_SCRIPT" "${child_args[@]}")
  fi

  printf '[%s] checkpoint: %s\n' "$language" "$checkpoint_dir"
  printf '[%s] command: ' "$language"
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$MODE" != dry-run ]]; then
    "${command[@]}"
  fi
done

if [[ "$MODE" == dry-run ]]; then
  echo "Dry run only. Omit --dry-run to execute sequentially, or use --sbatch."
fi
