#! /bin/bash

#SBATCH --job-name=slt_qwen_eval
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

export NCCL_P2P_DISABLE=1 # NOTE: 集群通信在部分节点上容易出现问题。

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

echo "Evaluating, disable wandb"
REPORT_TO=none

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

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.evaluate
  --config-name eval/base
  model.checkpoint_dir=/mnt/scratch/users/2533494w/slt_outputs/v4.0-qwen3-1.7b-cradio-l-crossshuffle-lite-0828.224x224-ctc-de/checkpoint-42000
  prompt=fixed_prompt
  engine.model_dtype=auto
  engine.training_args.output_dir=outputs/eval/v4.0-qwen3-1.7b-cradio-l-crossshuffle-lite-0828.224x224-ctc-de-checkpoint-42000
  engine.training_args.per_device_eval_batch_size=1
  engine.training_args.dataloader_num_workers=6
  engine.training_args.dataloader_persistent_workers=False
  engine.training_args.logging_steps=15
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  engine.generation_config.max_new_tokens=128
  engine.generation_config.do_sample=false
  experiment.permutation=false
  data=ph14t_*x224x224_qwen_single_language
  data.language=de
  data.processor.video_token_scale=1.0
  data.data_root="$DATASET_PATH"
  data.processor.num_extra_video_tokens=2
  data.processor.video_processor.padding_to_multiple_of=4
  data.processor.video_processor.do_resize=False
  data.processor.video_processor.do_normalize=False # NOTE: 很重要！！！
)

accelerate launch "${CMD_ARGS[@]}"
