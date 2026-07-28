#! /bin/bash

#SBATCH --job-name=slt_qwen
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export OPENCV_FOR_THREADS_NUM=1

export WANDB_PROJECT=sign_language_translation_v3.1

source .venv/bin/activate

# 如果第一个参数是 "debug"，则设置 REPORT_TO=none
if [[ "${1:-}" == "debug" ]]; then
  echo "Debug mode: Disabling reporting to WandB."
  REPORT_TO=none
else
  export WANDB_PROJECT=sign_language_translation_v3.1
  REPORT_TO=wandb
fi

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  model=qwen3-1.7b-dino-b-dinoframecrossv2shuffle \
  engine.training_args.output_dir=outputs/qwen3-1.7b-dinoframev2-shuffle-cross-test \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=1 \
  engine.training_args.dataloader_num_workers=12 \
  engine.training_args.eval_steps=4000 \
  engine.training_args.save_steps=4000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=False \
  engine.training_args.report_to="$REPORT_TO" \
  data=ph14t_*x224x224_qwen_multiling \
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
