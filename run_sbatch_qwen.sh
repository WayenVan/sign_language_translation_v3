#! /bin/bash

#SBATCH --job-name=slt_qwen
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

set -euo pipefail

source .venv/bin/activate

# 共享存储中的 tar 文件
SOURCE_ARCHIVE="dataset/phoenix-2014-T.v3.tar.gz"

# Local scratch
LOCAL_SCRATCH="$HOME/localscratch"

# 本地 tar 文件
LOCAL_ARCHIVE="$LOCAL_SCRATCH/$(basename "$SOURCE_ARCHIVE")"

# 最终数据集目录：解压和预处理都在这里进行
DATASET_PATH="$LOCAL_SCRATCH/ph14t"

# 解压完成标志
EXTRACT_MARKER="$DATASET_PATH/.extract_complete"

# 预处理完成标志
COMPLETE_MARKER="$DATASET_PATH/.data_complete"

mkdir -p "$LOCAL_SCRATCH"

if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "数据集已经完成预处理，跳过：$DATASET_PATH"

else
  if [[ -f "$EXTRACT_MARKER" ]]; then
    echo "数据集已经完成解压，直接进行预处理：$DATASET_PATH"
  else
    # 本地已有 tar 就直接使用，否则从 shared scratch 复制
    if [[ -f "$LOCAL_ARCHIVE" ]]; then
      echo "发现本地 tar，跳过复制：$LOCAL_ARCHIVE"
    else
      if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
        echo "错误：源 tar 不存在：$SOURCE_ARCHIVE" >&2
        exit 1
      fi

      echo "复制 tar 到 local scratch..."

      rsync -ah --info=progress2 \
        "$SOURCE_ARCHIVE" \
        "$LOCAL_ARCHIVE"
    fi

    echo "解压数据集到：$DATASET_PATH"

    rm -rf "$DATASET_PATH"
    mkdir -p "$DATASET_PATH"

    tar -xf "$LOCAL_ARCHIVE" \
      -C "$DATASET_PATH" \
      --strip-components=1

    # 只有解压成功后才创建标志
    touch "$EXTRACT_MARKER"
  fi

  # 在解压后的目录上进行预处理
  echo "开始预处理..."

  # 之前的预处理结果可能不完整，先删除
  rm -rf "$DATASET_PATH/ph14t-preprocessed"

  python preprocess/dataset_preprocess-T.py \
    --save_dir "$DATASET_PATH/ph14t-preprocessed" \
    --dataset-root "$DATASET_PATH/PHOENIX-2014-T" \
    -p \
    -m \
    -w "$(nproc)"

  # 只有预处理成功后才创建标志
  touch "$COMPLETE_MARKER"

  echo "数据集准备完成：$DATASET_PATH"
fi

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
