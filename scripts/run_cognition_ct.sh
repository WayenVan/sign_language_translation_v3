#! /bin/bash

#SBATCH --job-name=slt_qwen
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=1 # NOTE: 测试的时候集群通信容易出问题 集群出现了问题

SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# 可选参数：debug 关闭 WandB，share 直接使用共享数据集。
DEBUG=false
SHARED_DATASET=false
for arg in "$@"; do
  case "$arg" in
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  *)
    echo "Unknown argument: $arg (supported: debug, share)" >&2
    exit 2
    ;;
  esac
done

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB."
  REPORT_TO=none
else
  export WANDB_PROJECT=sign_language_translation_v4.0-dev
  export WANDB_TAGS=lite-lora-experiment
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

# share 模式下直接使用共享路径，否则准备数据集到本地 scratch。
if [[ "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"

# Array form instead of backslash-continued lines: inside (...) each element
# can live on its own line and be commented out individually with a leading
# "#" without breaking the rest of the command (a "#" on a "\"-continued
# line eats that line's trailing backslash too and splits the command).
CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name train/ft_peft
  # peft=qwen3-1.7b-llm-all
  peft=qwen3-1.7b-cradio-v4-so400m-visual-last4
  peft.visual_lora_config.r=4
  peft.visual_lora_config.lora_alpha=4
  # C-RADIO stays in deterministic eval mode while LoRA receives gradients;
  # make the disabled-dropout intent agree with the PEFT configuration.
  peft.visual_lora_config.lora_dropout=0.0
  model.checkpoint_dir=/mnt/scratch/users/2533494w/slt_outputs/v4.0-qwen3-1.7b-cradio-l-crossshuffle-lite-0828.224x224-ctc-de//checkpoint-42000
  engine.trainability.llm.parameter_mode=frozen
  engine.trainability.llm.runtime_mode=eval
  engine.trainability.visual_backbone.parameter_mode=lora
  engine.trainability.visual_backbone.runtime_mode=eval
  engine.trainability.visual_adapter.parameter_mode=frozen
  engine.trainability.visual_adapter.runtime_mode=eval
  prompt=fixed_prompt
  engine.training_args.num_train_epochs=50
  engine.training_args.learning_rate=1e-4
  # The visual backbone is LoRA-only here, so this targets exactly the visual
  # LoRA parameters the old visual_lora_* fields did.
  ++engine.optimization.visual_backbone.learning_rate=2e-3
  ++engine.optimization.visual_backbone.weight_decay=0.01
  # ++engine.optimization.visual_adapter.learning_rate=1e-5
  # ++engine.optimization.visual_adapter.weight_decay=0.05
  engine.evaluate_before_training=true
  engine.training_args.eval_steps=0.08
  # datamodule=train_with_val_and_test
  # engine.training_args.output_dir=outputs/v4.0-qwen3-1.7b-cradio-l-dinoframecrossv28shuffle-0827.224x224-ctc-de-cpt-vlora-last8-qkv-r8-adapter-frozen-ep40
  # engine.training_args.output_dir=outputs/v4.0-qwen3-1.7b-cradio-l-dinoframecrossv28shuffle-0827.224x224-ctc-de-cpt-vlora-last8-qkv-r8-adapter-ft-from33672-ep25
  # engine.training_args.output_dir=outputs/v4.0-qwen3-1.7b-cradio-l-dinoframecrossv28shuffle-0825.224x224-ctc-de-cpt-vlora-last-half-qkv-r8-adapter-frozen
  engine.training_args.output_dir=outputs/v4.0-qwen3-1.7b-cradio-l-crossshuffle-lite-0828.224x224-ctc-de-cpt-vlora-last4-r4a4-adapter-frozen-ep50
  # engine.training_args.output_dir=outputs/v4.0-qwen3-1.7b-cradio-l-dinoframecrossv28shuffle-0825.224x224-ctc-en-cpt-lora-r8
  # engine.training_args.output_dir=outputs/v4.0-qwen3-1.7b-cradio-l-dinoframecrossv28shuffle-0825.224x224-ctc-zh-cpt-lora-r8
  # engine.training_args.output_dir=outputs/v4.0-qwen3-1.7b-cradio-l-dinoframecrossv28shuffle-0825.224x224-ctc-fixed-cpt-lora-r8
  engine.training_args.per_device_train_batch_size=2
  engine.training_args.per_device_eval_batch_size=1
  engine.training_args.dataloader_num_workers=6
  engine.training_args.logging_steps=15
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  engine.training_args.ddp_find_unused_parameters=False
  engine.llm_dtype=bfloat16
  engine.visual_backbone_dtype=auto
  # data=ph14t_*x224x224_qwen_multiling
  data=ph14t_*x224x224_qwen_single_language
  data.language=de
  data.processor.video_token_scale=1.0
  data.data_root="$DATASET_PATH"
  data.processor.num_extra_video_tokens=2
  data.processor.video_processor.padding_to_multiple_of=4
  data.processor.video_processor.do_resize=False
  data.processor.video_processor.do_normalize=False # NOTE: 很重要！！！
)

accelerate launch "${CMD_ARGS[@]}"

# model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# accelerate launch --num_processes=2 --mixed_precision=fp16 \
# engine.training_args.auto_output_root=./outputs/peft_ft # -m csi_slt.commands.train \
# engine.training_args.dataloader_num_workers=10 # accelerate launch --num_processes=2 --mixed_precision=bf16 \
# model.config.visual_adapter_kwargs.num_layers=4 \
# model.config.video_token_scale=0.25 # model.config.visual_adapter_kwargs.use_temporal_shuffle=False \
# model=qwen3-1.7b-cradio-l-dinoframecrossv2global \
