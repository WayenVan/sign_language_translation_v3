#! /bin/bash

#SBATCH --job-name=slt_qwen14b
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g

export WANDB_PROJECT=sign_language_translation_v3

export PYTHONPATH=./src:$PYTHONPATH

source .venv/bin/activate

accelerate launch --num_processes=3 --mixed_precision=bf16 --debug -m csi_slt.commands.evaluate \
  model.checkpoint_dir=outputs/qwen3-1.7b-dino-base-tsamplerv2-shuffle-4/checkpoint-88000 \
  engine.training_args.output_dir=outputs/eval/qwen3-14b-dino-base-tsamplerv2-shuffle-4 \
  engine.training_args.per_device_eval_batch_size=2 \
  engine.training_args.dataloader_num_workers=10 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=False \
  data.train.processor.video_token_scale=0.0625 \
  data.val.processor.video_token_scale=0.0625 \
  data.test.processor.video_token_scale=0.0625 \
  data.train.processor.video_padding_to_multiple_of=16 \
  data.val.processor.video_padding_to_multiple_of=16 \
  data.test.processor.video_padding_to_multiple_of=16
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train_ft_peft \
# engine.training_args.dataloader_num_workers=10 # accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.visual_adapter_kwargs.num_layers=4 \
# model.config.video_token_scale=0.0625 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
