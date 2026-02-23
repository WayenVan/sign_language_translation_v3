#! /bin/bash

#SBATCH --job-name=slt
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

export WANDB_PROJECT=sign_language_translation_v3

export PYTHONPATH=./src:$PYTHONPATH

source .venv/bin/activate

accelerate launch --num_processes=3 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  model=qwen3-1.7b-dino-base-tsamplerv2 \
  engine.training_args.output_dir=outputs/qwen3-1.7b-dino-base-tsamplerv2-shuffle-3 \
  model.config.visual_adapter_kwargs.num_layers_connector=3 \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=2 \
  engine.training_args.dataloader_num_workers=10 \
  engine.training_args.eval_steps=4000 \
  engine.training_args.save_steps=4000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=True \
  data.train.processor.video_token_scale=0.125 \
  data.val.processor.video_token_scale=0.125 \
  data.test.processor.video_token_scale=0.125 \
  data.train.processor.video_padding_to_multiple_of=8 \
  data.val.processor.video_padding_to_multiple_of=8 \
  data.test.processor.video_padding_to_multiple_of=8 \
  \
  # model.config.visual_adapter_kwargs.num_layers=4 \
  # -m csi_slt.commands.train_ft_peft # engine.training_args.auto_output_root=./outputs/peft_ft \
# accelerate launch --num_processes=2 --mixed_precision=bf16 # engine.training_args.dataloader_num_workers=10 \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# model.config.video_token_scale=1.0 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
