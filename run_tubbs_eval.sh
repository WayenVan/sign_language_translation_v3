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

source .venv/bin/activate

# 如果第一个参数是 "debug"，则设置 REPORT_TO=none
if [[ "${1:-}" == "debug" ]]; then
  echo "Debug mode"
else
fi

accelerate launch --num_processes=2 --mixed_precision=bf16 --debug -m csi_slt.commands.evaluate \
  model.checkpoint_dir=outputs/v3.0-qwen3-1.7b-cradio-l-dinoframecrossv2shuffle-0815-224x224//checkpoint-54000 \
  engine.training_args.output_dir=outputs/eval_base \
  engine.training_args.per_device_eval_batch_size=1 \
  engine.training_args.dataloader_num_workers=12 \
  engine.training_args.logging_steps=15 \
  engine.training_args.disable_tqdm=False \
  engine.training_args.report_to=none \
  data=ph14t_*x224x224_qwen_multiling \
  data.processor.video_token_scale=1.0 \
  data.processor.video_processor.padding_to_multiple_of=4 \
  data.processor.video_processor.do_resize=False \
  data.processor.video_processor.do_normalize=False
# accelerate launch --num_processes=2 --mixed_precision=fp16 \

# engine.training_args.dataloader_num_workers=10 \
# model.config.visual_adapter_kwargs.num_layers=4 \

# model=qwen3-8b-dino-b-dinoframecrossv2shuffle # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train \
# accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.video_token_scale=0.25 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
#   engine.training_args.output_dir=outputs/qwen3-8b-dinoframev2-shuffle-cross-0730
