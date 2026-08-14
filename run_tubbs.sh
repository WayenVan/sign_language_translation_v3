#! /bin/bash

export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

export WANDB_PROJECT=sign_language_translation_v3.0-dev

source .venv/bin/activate

# 如果第一个参数是 "debug"，则设置 REPORT_TO=none
if [[ "${1:-}" == "debug" ]]; then
  echo "Debug mode: Disabling reporting to WandB."
  REPORT_TO=none
else
  export WANDB_PROJECT=sign_language_translation_v3.0-dev
  REPORT_TO=wandb
fi

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  model=qwen3-1.7b-siglip2-g-dinoframecrossv3 \
  engine.training_args.output_dir=outputs/v3.0-qwen3-1.7b-siglip2-g-dinoframecrossv3-0814-256x256 \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=1 \
  engine.training_args.dataloader_num_workers=12 \
  engine.training_args.eval_steps=6000 \
  engine.training_args.save_steps=6000 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=False \
  engine.training_args.report_to="$REPORT_TO" \
  engine.llm_dtype=bfloat16 \
  engine.visual_backbone_dtype=float32 \
  data=ph14t_*x256x256_qwen_multiling \
  data.processor.video_token_scale=2.0 \
  data.processor.num_extra_video_tokens=2 \
  data.processor.video_processor.padding_to_multiple_of=4 \
  data.processor.video_processor.do_resize=False \
  data.processor.video_processor.do_normalize=False
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train_ft_peft \
# engine.training_args.dataloader_num_workers=10 # accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.visual_adapter_kwargs.num_layers=4 \
# model.config.video_token_scale=0.25 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# model=qwen3-8b-dino-b-dinoframecrossv2shuffle \
#   engine.training_args.output_dir=outputs/qwen3-8b-dinoframev2-shuffle-cross-0730
