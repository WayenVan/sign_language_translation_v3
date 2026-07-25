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

set -euo pipefail

SOURCE_ARCHIVE="/shared/scratch/$USER/datasets/ph14t.tar.gz"

LOCAL_SCRATCH="$HOME/localscratch"
DATASET_NAME="ph14t"

TARGET_DATASET="$LOCAL_SCRATCH/$DATASET_NAME"
COMPLETE_MARKER="$TARGET_DATASET/.copy_complete"
LOCAL_ARCHIVE="$LOCAL_SCRATCH/$(basename "$SOURCE_ARCHIVE")"

if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
  echo "错误：压缩包不存在：$SOURCE_ARCHIVE" >&2
  exit 1
fi

mkdir -p "$LOCAL_SCRATCH"

if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "数据集已经准备完成，跳过：$TARGET_DATASET"
else
  echo "复制压缩包到 local scratch..."

  rsync -a --info=progress2 \
    "$SOURCE_ARCHIVE" \
    "$LOCAL_ARCHIVE"

  echo "解压数据集..."

  rm -rf "$TARGET_DATASET"
  mkdir -p "$TARGET_DATASET"

  tar -xzf "$LOCAL_ARCHIVE" \
    -C "$TARGET_DATASET" \
    --strip-components=1

  touch "$COMPLETE_MARKER"
  rm -f "$LOCAL_ARCHIVE"

  echo "数据集准备完成：$TARGET_DATASET"
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
