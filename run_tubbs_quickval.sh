#! /bin/bash

# 跑一些valid数据实验，使用小数据，混合数据或者验证数据来进行 最小测试

export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

export WANDB_PROJECT=sign_language_translation_v3.1_quickval

source .venv/bin/activate

# 如果第一个参数是 "debug"，则设置 REPORT_TO=none
if [[ "${1:-}" == "debug" ]]; then
  echo "Debug mode: Disabling reporting to WandB."
  REPORT_TO=none
else
  REPORT_TO=wandb
fi

accelerate launch --num_processes=1 --mixed_precision=bf16 --debug -m csi_slt.commands.train \
  --config-name train/quick_validation \
  model=qwen3-1.7b-cradio-l-dinoframecrossv2shuffle \
  engine.training_args.output_dir=outputs/quick_val-qwen3-1.7b-cradio-l-dinoframecrossv2shuffle-0807.224x224 \
  engine.training_args.per_device_train_batch_size=2 \
  engine.training_args.per_device_eval_batch_size=1 \
  engine.training_args.dataloader_num_workers=12 \
  engine.training_args.logging_steps=10 \
  engine.training_args.disable_tqdm=False \
  engine.training_args.report_to="$REPORT_TO" \
  data=ph14t_*x224x224_qwen_multiling \
  data.processor.video_processor.do_resize=False \
  data.processor.video_token_scale=1.0 \
  data.processor.video_processor.padding_to_multiple_of=4 \
  data.processor.video_processor.do_normalize=False
# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train_ft_peft \
# engine.training_args.dataloader_num_workers=10 # accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.visual_adapter_kwargs.num_layers=4 \
# model.config.video_token_scale=0.25 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# model=qwen3-8b-dino-b-dinoframecrossv2shuffle \
#   engine.training_args.output_dir=outputs/qwen3-8b-dinoframev2-shuffle-cross-0730
