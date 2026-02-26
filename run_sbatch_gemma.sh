#! /bin/bash

#SBATCH --job-name=slt_gemma_shuffle4
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

export WANDB_PROJECT=sign_language_translation_v3

export PYTHONPATH=./src:$PYTHONPATH

source .venv/bin/activate

accelerate launch --num_processes=3 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  model=gemma3-4b-dino-base-tsamplerv2 \
  data=ph14t_*x224x224_gemma_multiling \
  engine.training_args.output_dir=outputs/gemma3-4b-dino2-b-tsamplev2-shuffle-4 \
  model.config.visual_adapter_kwargs.num_layers_connector=4 \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=2 \
  engine.training_args.dataloader_num_workers=10 \
  engine.training_args.eval_steps=4000 \
  engine.training_args.save_steps=4000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=True \
  model.config.video_token_scale=0.0625 \
  data.train.processor.video_token_scale=0.0625 \
  data.val.processor.video_token_scale=0.0625 \
  data.test.processor.video_token_scale=0.0625 \
  data.train.processor.video_padding_to_multiple_of=16 \
  data.val.processor.video_padding_to_multiple_of=16 \
  data.test.processor.video_padding_to_multiple_of=16
# data.train.processor.video_token_scale=1.0 \
# data.val.processor.video_token_scale=1.0 \
# data.test.processor.video_token_scale=1.0 \
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# model.config.video_token_scale=1.0
# model.config.visual_adapter_kwargs.num_layers=4 \
# accelerate launch --num_processes=2 --mixed_precision=bf16 \
# 	-m csi_slt.commands.train_ft_peft \
# 	engine.training_args.dataloader_num_workers=10 \
# 	engine.training_args.auto_output_root=./outputs/peft_ft # accelerate launch --num_processes=2 --mixed_precision=fp16 \
