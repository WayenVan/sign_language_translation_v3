#!/usr/bin/env bash

#SBATCH --job-name=slt_best_adapter_multilang
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

# =========================================================================== #
# USAGE
#   sbatch scripts/run_best_adapter_multilang.sh [PROMPT_CONFIG] [EPOCHS]
#   bash scripts/run_best_adapter_multilang.sh [PROMPT_CONFIG] [EPOCHS]
#
# Defaults: PROMPT_CONFIG=fixed_prompt, EPOCHS=20.
# PROMPT_CONFIG is the basename of a YAML file under configs/prompt/.
#
# Examples:
#   sbatch scripts/run_best_adapter_multilang.sh
#   sbatch scripts/run_best_adapter_multilang.sh diverse_train
#   sbatch scripts/run_best_adapter_multilang.sh fixed_prompt_weather 25
#   scripts/run_best_adapter_multilang.sh --help
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

if (( $# > 2 )); then
  usage >&2
  exit 2
fi

PROMPT_CONFIG="${1:-fixed_prompt}"
NUM_TRAIN_EPOCHS="${2:-20}"

# Keep the prompt argument to a Hydra config-group name, not an arbitrary path.
if [[ ! "$PROMPT_CONFIG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "prompt config must contain only letters, digits, '_' or '-': $PROMPT_CONFIG" >&2
  exit 2
fi
if [[ ! "$NUM_TRAIN_EPOCHS" =~ ^[0-9]+$ ]] || (( NUM_TRAIN_EPOCHS < 1 )); then
  echo "epochs must be a positive integer, got: $NUM_TRAIN_EPOCHS" >&2
  exit 2
fi

# Match the host handling used by the newer experiment launchers. On tubbs the
# dataset is already extracted in the checkout; on the cluster it is staged to
# node-local scratch before training.
HOST_FQDN="$(hostname -f)"
if [[ "$HOST_FQDN" == "tubbs.eng.gla.ac.uk" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
  IS_TUBBS=true
elif [[ -d /users/2533494w/projects/sign_language_translation_v3 ]]; then
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
  IS_TUBBS=false
else
  echo "unrecognized host '$HOST_FQDN': no known sign_language_translation_v3 checkout here; refusing to run." >&2
  exit 4
fi

if [[ ! -f "$SCRIPT_DIR/configs/prompt/${PROMPT_CONFIG}.yaml" ]]; then
  echo "prompt config does not exist: configs/prompt/${PROMPT_CONFIG}.yaml" >&2
  exit 2
fi

if [[ "$IS_TUBBS" == true ]]; then
  unset NCCL_P2P_DISABLE
else
  export NCCL_P2P_DISABLE=1
fi

cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

export WANDB_PROJECT=sign_language_translation_v5.0-dev
export WANDB_TAGS="best-adapter,multilang,${PROMPT_CONFIG},posenc-learned,ep${NUM_TRAIN_EPOCHS}"
OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-best-adapter-multilang-${PROMPT_CONFIG}-ep${NUM_TRAIN_EPOCHS}-0909.224x224"

if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

if [[ "$IS_TUBBS" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi

echo "PROMPT_CONFIG=$PROMPT_CONFIG"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/pretrain_adapter/best_adapter_multilang
  # The only experiment overrides exposed by this launcher.
  prompt="$PROMPT_CONFIG"
  engine.training_args.num_train_epochs="$NUM_TRAIN_EPOCHS"
  # Runtime-specific values; these do not change the experiment definition.
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
