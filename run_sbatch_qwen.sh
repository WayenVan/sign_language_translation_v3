#! /bin/bash

#SBATCH --job-name=slt_qwen
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

set -euo pipefail

SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"
source "$SCRIPT_DIR/prepare_dataset.sh"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# 设置 TQDM_DISABLE 和 HG_TQDM_DISABLE 用于 accelerate launch
if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

# 准备数据集，函数输出 DATASET_PATH
DATASET_PATH=$(prepare_dataset \
  "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
  "$HOME/localscratch/ph14t")
echo "DATASET_PATH=$DATASET_PATH"

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  model=qwen3-8b-dino-b-dinoframecross \
  engine.training_args.output_dir=outputs/qwen3-8b-dino-b-dinoframe-cross-test \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=1 \
  engine.training_args.dataloader_num_workers=4 \
  engine.training_args.eval_steps=4000 \
  engine.training_args.save_steps=4000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE" \
  data=ph14t_*x224x224_qwen_multiling \
  data.data_root="$DATASET_PATH" \
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
