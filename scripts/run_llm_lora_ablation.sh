#! /bin/bash
#
# =========================================================================== #
# USAGE
#   sbatch [-J JOB_NAME] scripts/run_llm_lora_ablation.sh [TARGETS] [RANK] [EPOCHS] [debug] [share]
#   bash scripts/run_llm_lora_ablation.sh [TARGETS] [RANK] [EPOCHS] [debug] [share]
#
# TARGETS: qv or qkvo. Defaults: qv, RANK=8, ALPHA=2*RANK, all 36 layers,
# LR=1e-4, EPOCHS=config (12).
# Example:  sbatch -J slt_llora_qkvo_r8 scripts/run_llm_lora_ablation.sh qkvo 8
# Env:      LLM_LORA_LR overrides the LoRA learning rate (default 1e-4).
# Help:     scripts/run_llm_lora_ablation.sh --help
# =========================================================================== #

#SBATCH --job-name=slt_qwen3_4b_llm_lora_ablation
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

usage() {
  sed -n '/^# USAGE$/,/^# =\{10,\} #$/p' "$0" \
    | sed 's/^# \{0,1\}//' \
    | sed '$d'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Host allowlist. run_cognition_outputlayer.sh silently falls back to the
# cluster path on any non-tubbs host; here an unrecognized host is a hard stop
# so a stray launch (laptop, wrong login node) fails immediately instead of
# cd-ing into a path that does not exist and half-running.
#
# Per-host, IS_TUBBS also decides two things below:
#   * dataset: tubbs has it extracted in-repo, so skip the stage-to-scratch step;
#   * NCCL P2P: only the cluster fabric needs NCCL_P2P_DISABLE=1 (集群通信容易
#     出问题); tubbs's local GPUs keep P2P enabled.
HOST_FQDN="$(hostname -f)"
if [[ "$HOST_FQDN" == "tubbs.eng.gla.ac.uk" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
  IS_TUBBS=true
elif [[ -d /users/2533494w/projects/sign_language_translation_v3 ]]; then
  # Glasgow HPC cluster.
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
  IS_TUBBS=false
else
  echo "unrecognized host '$HOST_FQDN': no known sign_language_translation_v3 checkout here; refusing to run." >&2
  exit 4
fi

if [[ "$IS_TUBBS" == true ]]; then
  unset NCCL_P2P_DISABLE
else
  export NCCL_P2P_DISABLE=1
fi

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# LLM-backend LoRA ablation on the ol-8 best-adapter checkpoint.
#
# Continues from the stage-1 checkpoint below (visual adapter trained against a
# frozen Qwen3-4B + frozen C-RADIO, output_layer=-8, learned visual position
# table; checkpoint-96000 is that run's best_model_checkpoint, test de_bleu4
# 0.1715) and injects LoRA into all 36 Qwen3-4B decoder blocks.
#
# Config: train/lora_llm/base. What it moves:
#   * llm LoRA (selected attention projections, all blocks) -- lr = 1e-4
#   * visual-side interface (adapter, CTC head, learned positions, boundary
#     embeddings, and visual_scale)              -- lr = 1e-5, from the config
#   frozen: C-RADIO backbone.
# The adapter stays trainable on purpose: every v4.0 stage-2 run that froze it
# lost ground against its own stage-1 checkpoint.
#
# The ablation knobs. Run experiments one at a time; nothing is swept here.
#   $1  targets -- qv => q_proj/v_proj; qkvo => q/k/v/o projections (default qv)
#   $2  rank    -- LoRA rank; lora_alpha is set to 2 * rank        (default 8)
#   $3  epochs -- optional positive int; omitted keeps the config default (12)
# plus, anywhere after those: debug (no WandB, outputs/debug), share (use the
# shared dataset path instead of staging to local scratch).
#
# Examples:
#   sbatch -J slt_llora_qv_r8       scripts/run_llm_lora_ablation.sh qv 8
#   sbatch -J slt_llora_qkvo_r16    scripts/run_llm_lora_ablation.sh qkvo 16
#   sbatch -J slt_llora_qv_r8_ep20  scripts/run_llm_lora_ablation.sh qv 8 20
#   bash scripts/run_llm_lora_ablation.sh qkvo 8 debug share
# --------------------------------------------------------------------------- #

# Default stage-1 checkpoint used by the LLM-LoRA continuation. Prefer the
# scratch copy on the cluster; fall back to the in-repo outputs/ tree.
CKPT_LEAF="v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-0907.224x224/checkpoint-96000"
if [[ -d "/mnt/scratch/users/2533494w/slt_outputs/${CKPT_LEAF}" ]]; then
  CHECKPOINT_DIR="/mnt/scratch/users/2533494w/slt_outputs/${CKPT_LEAF}"
else
  CHECKPOINT_DIR="${SCRIPT_DIR}/outputs/${CKPT_LEAF}"
fi

TARGETS="${1:-qv}"
RANK="${2:-8}"
LEARNING_RATE="${LLM_LORA_LR:-1e-4}"
case "$TARGETS" in
qv) TARGET_MODULES='[q_proj,v_proj]' ;;
qkvo) TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj]' ;;
*) echo "targets must be qv or qkvo, got: $TARGETS" >&2; exit 2 ;;
esac
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
TARGET_LANGUAGE=de
RUN_TAG="llmlora-all-ckpt96k-${TARGET_LANGUAGE}-${TARGETS}-r${RANK}a${ALPHA}-lr${LEARNING_RATE}${EP_SUFFIX}"

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="llm-lora,all-layers,targets-${TARGETS},ol-8,ckpt96k,language-${TARGET_LANGUAGE},lr-${LEARNING_RATE},${RUN_TAG}"
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

# tubbs 上数据集已在仓库内解压好，直接用；share 模式同理走共享路径。
# 只有集群需要把数据集迁移(stage)到本地 scratch。
if [[ "$IS_TUBBS" == true || "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"
echo "TARGET_LANGUAGE=$TARGET_LANGUAGE"
echo "TARGETS=$TARGETS  TARGET_MODULES=$TARGET_MODULES"
echo "LAYERS=all  RANK=$RANK  ALPHA=$ALPHA  LEARNING_RATE=$LEARNING_RATE"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/lora_llm/base
  model.checkpoint_dir="$CHECKPOINT_DIR"
  # --- the ablation knobs ---
  peft.llm_lora_config.target_modules="$TARGET_MODULES"
  peft.llm_lora_config.r="$RANK"
  peft.llm_lora_config.lora_alpha="$ALPHA"
  engine.optimization.llm.learning_rate="$LEARNING_RATE"
  # --------------------------
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
  data.language="$TARGET_LANGUAGE"
)

# Optional longer/shorter run: override only when an epoch count was passed,
# otherwise the config keeps num_train_epochs=12.
if [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  CMD_ARGS+=(engine.training_args.num_train_epochs="$NUM_TRAIN_EPOCHS")
  echo "NUM_TRAIN_EPOCHS override = $NUM_TRAIN_EPOCHS"
fi

accelerate launch "${CMD_ARGS[@]}"
