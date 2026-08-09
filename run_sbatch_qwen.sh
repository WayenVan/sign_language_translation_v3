#! /bin/bash

#SBATCH --job-name=slt_qwen
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

set -euo pipefail

export NCCL_P2P_DISABLE=1 # NOTE: 测试的时候集群通信容易出问题 集群出现了问题

SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"
source "$SCRIPT_DIR/prepare_dataset.sh"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# 如果第一个参数是 "debug"，则设置 REPORT_TO=none
if [[ "${1:-}" == "debug" ]]; then
  echo "Debug mode: Disabling reporting to WandB."
  REPORT_TO=none
else
  export WANDB_PROJECT=sign_language_translation_v3.1
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

# 准备数据集，函数输出 DATASET_PATH
DATASET_PATH=$(prepare_dataset \
  "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
  "$HOME/localscratch/ph14t")
echo "DATASET_PATH=$DATASET_PATH"

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  model=qwen3-1.7b-cradio-l-dinoframecrossv2global \
  engine.training_args.output_dir=outputs/qwen3-1.7b-cradio-l-dinoframecrossv2global-0808.224x224 \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=1 \
  engine.training_args.dataloader_num_workers=8 \
  engine.training_args.dataloader_persistent_workers=True \
  engine.training_args.eval_steps=6000 \
  engine.training_args.save_steps=6000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE" \
  engine.training_args.report_to="$REPORT_TO" \
  engine.training_args.ddp_find_unused_parameters=True \
  data=ph14t_*x224x224_qwen_multiling \
  data.processor.video_token_scale=2.0 \
  data.data_root="$DATASET_PATH" \
  data.processor.num_extra_video_tokens=3 \
  data.processor.video_processor.padding_to_multiple_of=4 \
  data.processor.video_processor.do_resize=False \
  data.processor.video_processor.do_normalize=False # NOTE: 很重要！！！

# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train_ft_peft \
# engine.training_args.dataloader_num_workers=10 # accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.visual_adapter_kwargs.num_layers=4 \
# model.config.video_token_scale=0.25 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
