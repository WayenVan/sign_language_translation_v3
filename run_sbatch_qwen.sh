#! /bin/bash

#SBATCH --job-name=slt_qwen
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=./src

# 修改为共享存储中的数据集路径
SOURCE_DATASET="/users/$USER/sharedscratch/dataset/PHOENIX-2014-T-release-v3"

export WANDB_PROJECT=sign_language_translation_v3.1

# Local scratch 位置
LOCAL_SCRATCH="$HOME/localscratch"

# 默认使用源目录名作为目标目录名
DATASET_NAME="$(basename "$SOURCE_DATASET")"
TARGET_DATASET="$LOCAL_SCRATCH/$DATASET_NAME"
COMPLETE_MARKER="$TARGET_DATASET/.copy_complete"

if [[ ! -d "$SOURCE_DATASET" ]]; then
  echo "错误：源数据集不存在：$SOURCE_DATASET" >&2
  exit 1
fi

mkdir -p "$LOCAL_SCRATCH"

if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "数据集已经拷贝完成，跳过：$TARGET_DATASET"
else
  echo "开始拷贝数据集："
  echo "  来源：$SOURCE_DATASET"
  echo "  目标：$TARGET_DATASET"

  mkdir -p "$TARGET_DATASET"

  # 推荐 rsync：中断后可以继续同步
  rsync -a --info=progress2 \
    "$SOURCE_DATASET/" \
    "$TARGET_DATASET/"

  touch "$COMPLETE_MARKER"
  echo "数据集拷贝完成：$TARGET_DATASET"
fi

# 后续训练可使用这个变量
export DATASET_PATH="$TARGET_DATASET"

source .venv/bin/activate

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  model=qwen3-8b-dino-b-dinoframe \
  engine.training_args.output_dir=outputs/qwen3-8b-dino-b-dinoframe \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=1 \
  engine.training_args.dataloader_num_workers=8 \
  engine.training_args.dataloader_persistent_workers=False \
  engine.training_args.eval_steps=4000 \
  engine.training_args.save_steps=4000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=True \
  data=ph14t_*x224x224_qwen_multiling \
  data.data_root=$DATASET_PATH \
  model.config.video_token_scale=1.0 \
  data.train.processor.video_token_scale=1.0 \
  data.val.processor.video_token_scale=1.0 \
  data.test.processor.video_token_scale=1.0 \
  data.train.processor.video_padding_to_multiple_of=4 \
  data.val.processor.video_padding_to_multiple_of=4 \
  data.test.processor.video_padding_to_multiple_of=4
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train_ft_peft \
# engine.training_args.dataloader_num_workers=10 # accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.visual_adapter_kwargs.num_layers=4 \
# model.config.video_token_scale=0.25 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
