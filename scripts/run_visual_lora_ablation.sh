#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_visual_lora_ablation
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

# =========================================================================== #
# USAGE
#   sbatch [-J JOB_NAME] scripts/run_visual_lora_ablation.sh [COUNT] [RANK] [EPOCHS] [debug] [share]
#   bash scripts/run_visual_lora_ablation.sh [COUNT] [RANK] [EPOCHS] [debug] [share]
#
# Defaults: COUNT=2, RANK=8, ALPHA=2*RANK, LR=1e-3, EPOCHS=20.
# Example:  sbatch -J slt_vlora_n4_r16 scripts/run_visual_lora_ablation.sh 4 16 30
# Help:     scripts/run_visual_lora_ablation.sh --help
# =========================================================================== #

usage() {
  sed -n '/^# USAGE$/,/^# =\{10,\} #$/p' "$0" \
    | sed 's/^# \{0,1\}//' \
    | sed '$d'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export NCCL_P2P_DISABLE=1 # NOTE: 测试的时候集群通信容易出问题 集群出现了问题

if [[ "$(hostname -f)" == "tubbs.eng.gla.ac.uk" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
else
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
fi

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# Visual-LoRA ablation on the ol-8 best-adapter checkpoint.
#
# Continues from the stage-1 checkpoint below (visual adapter trained against a
# frozen Qwen3-4B + frozen C-RADIO, output_layer=-8, learned visual position
# table) and adds LoRA to the last N C-RADIO ViT blocks. The span ends at the
# block that feeds the adapter -- output_layer=-8 -- which is baked into the
# checkpoint, so there is no per-layer scorer to swap here (unlike
# run_cognition_outputlayer.sh).
#
# Config: train/lora_visual_encoder/base. What it moves:
#   * visual_backbone LoRA  -- lr = 1e-3 by default in this launcher
#   frozen: LLM, visual adapter, CTC head, learned visual position table,
#           visual boundary embeddings, and visual_scale.
#
# The ablation knobs. Run experiments one at a time; nothing is swept here.
#   $1  count  -- how many final C-RADIO ViT blocks get LoRA   (default 2)
#   $2  rank   -- LoRA rank; lora_alpha is set to 2 * rank      (default 8)
#   $3  epochs -- optional positive int; omitted keeps the config default (20)
# plus, anywhere after those: debug (no WandB, outputs/debug), share (use the
# shared dataset path instead of staging to local scratch).
#
# Examples:
#   sbatch -J slt_vlora_n2_r8      scripts/run_visual_lora_ablation.sh 2 8
#   sbatch -J slt_vlora_n4_r16     scripts/run_visual_lora_ablation.sh 4 16
#   sbatch -J slt_vlora_n2_r8_ep50 scripts/run_visual_lora_ablation.sh 2 8 50
#   bash scripts/run_visual_lora_ablation.sh 2 8 debug share
# --------------------------------------------------------------------------- #

# Default stage-1 checkpoint used by the visual-LoRA continuation.
CKPT_ROOT="/mnt/scratch/users/2533494w/slt_outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-0907.224x224"
CHECKPOINT_DIR="${CKPT_ROOT}/checkpoint-96000"

COUNT="${1:-2}"
RANK="${2:-8}"
LEARNING_RATE="1e-3"
if [[ ! "$COUNT" =~ ^[0-9]+$ ]] || (( COUNT < 1 )); then
  echo "count must be a positive integer, got: $COUNT" >&2
  exit 2
fi
if [[ ! "$RANK" =~ ^[0-9]+$ ]] || (( RANK < 1 )); then
  echo "rank must be a positive integer, got: $RANK" >&2
  exit 2
fi
[[ $# -ge 1 ]] && shift
[[ $# -ge 1 ]] && shift
ALPHA=$(( 2 * RANK ))

# Optional trailing flags: debug, share, and a bare positive int for epochs.
DEBUG=false
SHARED_DATASET=false
NUM_TRAIN_EPOCHS=""
for arg in "$@"; do
  case "$arg" in
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  *)
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
      NUM_TRAIN_EPOCHS="$arg"
    else
      echo "Unknown argument: $arg (supported: <int epochs>, debug, share)" >&2
      exit 2
    fi
    ;;
  esac
done

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "stage-1 checkpoint not found: $CHECKPOINT_DIR" >&2
  exit 3
fi

# Fold a non-default epoch count into every run identifier so a longer run
# never overwrites the default-length one's output dir.
EP_SUFFIX=""
if [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  EP_SUFFIX="-ep${NUM_TRAIN_EPOCHS}"
fi
RUN_TAG="vlora-only-ckpt96k-n${COUNT}-r${RANK}a${ALPHA}-lr${LEARNING_RATE}${EP_SUFFIX}"

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="visual-lora,lora-only,ol-8,ckpt96k,lr-${LEARNING_RATE},${RUN_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-ol-8-${RUN_TAG}-0908.224x224"
fi

# 设置 TQDM_DISABLE 和 HG_TQDM_DISABLE 用于 accelerate launch
if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

# share 模式下直接使用共享路径，否则准备数据集到本地 scratch。
if [[ "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"
echo "COUNT=$COUNT  RANK=$RANK  ALPHA=$ALPHA  LEARNING_RATE=$LEARNING_RATE"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/lora_visual_encoder/base
  model.checkpoint_dir="$CHECKPOINT_DIR"
  # --- the ablation knobs ---
  peft.visual_lora_layers.count="$COUNT"
  peft.visual_lora_config.r="$RANK"
  peft.visual_lora_config.lora_alpha="$ALPHA"
  engine.training_args.learning_rate="$LEARNING_RATE"
  # --------------------------
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

# Optional longer run: override only when an epoch count was passed, otherwise
# the config keeps num_train_epochs=20.
if [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  CMD_ARGS+=(engine.training_args.num_train_epochs="$NUM_TRAIN_EPOCHS")
  echo "NUM_TRAIN_EPOCHS override = $NUM_TRAIN_EPOCHS"
fi

accelerate launch "${CMD_ARGS[@]}"
