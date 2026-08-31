#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_attnpool20m_tconv
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

export NCCL_P2P_DISABLE=1 # NOTE: 测试的时候集群通信容易出问题 集群出现了问题

SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# 可选参数：debug 关闭 WandB，share 直接使用共享数据集。
DEBUG=false
SHARED_DATASET=false
for arg in "$@"; do
  case "$arg" in
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  *)
    echo "Unknown argument: $arg (supported: debug, share)" >&2
    exit 2
    ;;
  esac
done

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB."
  REPORT_TO=none
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="attnpool,20m,attention-gated-ablation,temporal-conv"
  REPORT_TO=wandb
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
  source "$SCRIPT_DIR/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"

# Array form instead of backslash-continued lines: inside (...) each element
# can live on its own line and be commented out individually with a leading
# "#" without breaking the rest of the command (a "#" on a "\"-continued
# line eats that line's trailing backslash too and splits the command).
CMD_ARGS=(
  # FSDP2: shards the frozen LLM across the job's GPUs. Comment this line out
  # to fall back to plain DDP. Qwen3-4B fits on each L40S for this frozen-base
  # probe, while DDP keeps the execution path simpler than FSDP2.
  # --config_file="$SCRIPT_DIR/configs/accelerate/fsdp2.yaml"
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/cognition/attention_gated_ablation
  # Temporary ablation override: enable the adapter's temporal depthwise Conv1d.
  model.config.visual_adapter_kwargs.use_temporal_conv=true
  # Temporal-scale ablation: average every two frames into one visual token.
  model.config.visual_adapter_kwargs.temporal_scale_factor=2
  model.config.video_token_scale=0.5
  data.processor.video_token_scale=0.5
  engine.training_args.output_dir=outputs/v5.0-qwen3-4b-cradio-l-attnpool20m-tconv-lm8-attnm25-topk48-ts2-0831.224x224-de
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
