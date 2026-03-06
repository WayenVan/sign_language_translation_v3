#! /bin/bash

#SBATCH --job-name=slt_vl
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
  model=qwen3vl-2.5b-dino-base-tsamplerv2 \
  engine.training_args.output_dir=outputs/qwen3vl-2.5b-tsamplerv2-shuffle-1 \
  model.config.visual_adapter_kwargs.num_layers_connector=1 \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=2 \
  engine.training_args.dataloader_num_workers=10 \
  engine.training_args.eval_steps=4000 \
  engine.training_args.save_steps=4000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=True \
  model.config.video_token_scale=0.25 \
  data.train.processor.video_token_scale=0.25 \
  data.val.processor.video_token_scale=0.25 \
  data.test.processor.video_token_scale=0.25 \
  data.train.processor.video_padding_to_multiple_of=4 \
  data.val.processor.video_padding_to_multiple_of=4 \
  data.test.processor.video_padding_to_multiple_of=4
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train_ft_peft \
# engine.training_args.dataloader_num_workers=10 # accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.visual_adapter_kwargs.num_layers=4 \
# model.config.video_token_scale=0.25 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
