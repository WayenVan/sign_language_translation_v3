#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_stpool_baseline
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

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
  export WANDB_TAGS="spatiotemporal-pooled-baseline,backbone-layer-probe"
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
  source "$SCRIPT_DIR/prepare_dataset.sh"
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
  # FSDP2: shards the frozen LLM across the job's GPUs. Comment this line out
  # to fall back to plain DDP. Qwen3-4B fits on each L40S for this frozen-base
  # probe, while DDP keeps the execution path simpler than FSDP2.
  # --config_file="$SCRIPT_DIR/configs/accelerate/fsdp2.yaml"
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  model=qwen3-4b-cradio-l-spatiotemporal-pooled-linear
  # Keep these two ablation choices explicit in the launch command even though
  # the model config already sets them, so a run log records them unambiguously.
  # Plain causal attention: visual tokens attend only to the past in Qwen.
  model.config.video_bidirectional_attention=false
  # Construct no adapter-side visual-position module. Temporal order between
  # pooled windows is represented only by Qwen's own RoPE.
  model.config.visual_position_embedding_type=none
  # Layer-probe policy: no LoRA is injected; freeze both pretrained towers and
  # train only the new baseline adapter plus the randomly initialized CTC head.
  peft=none
  engine.trainability.llm.mode=frozen
  engine.trainability.llm.runtime_mode=eval
  engine.trainability.visual_backbone.mode=frozen
  engine.trainability.visual_backbone.runtime_mode=eval
  engine.trainability.visual_adapter.mode=full
  engine.trainability.visual_adapter.runtime_mode=train
  engine.trainability.ctc_head.mode=full
  # These components are frozen to prevent them from absorbing layer-specific
  # differences. visual_position_embedding is absent in "none" mode, but its
  # explicit frozen plan keeps all seven required plan entries unambiguous.
  engine.trainability.visual_position_embedding.mode=frozen
  engine.trainability.visual_boundary_embeddings.mode=frozen
  engine.trainability.visual_scale.mode=frozen
  # The adapter receives an explicit LR; the CTC head uses the default LR.
  engine.training_args.learning_rate=1e-4
  +engine.training_args.visual_adapter_learning_rate=1e-4
  # prompt=diverse_train
  prompt=fixed_prompt
  engine.training_args.output_dir=outputs/v4.0-qwen3-4b-cradio-l-stpool-baseline-lm1-0830.224x224-ctc-de-causal-nopos
  engine.training_args.per_device_train_batch_size=2
  engine.training_args.per_device_eval_batch_size=1
  engine.training_args.dataloader_num_workers=6
  engine.training_args.eval_steps=6000
  engine.training_args.logging_steps=15
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  engine.training_args.ddp_find_unused_parameters=False
  # data=ph14t_*x224x224_gemma_multiling
  # data=ph14t_*x224x224_qwen_multiling
  data=ph14t_*x224x224_qwen_single_language
  data.language=de
  # The adapter averages every two frames and therefore emits T/2 tokens. This
  # must match model.config.video_token_scale=0.5 exactly.
  data.processor.video_token_scale=0.5
  data.data_root="$DATASET_PATH"
  data.processor.num_extra_video_tokens=2
  data.processor.video_processor.padding_to_multiple_of=4
  data.processor.video_processor.do_resize=False
  data.processor.video_processor.do_normalize=False # NOTE: 很重要！！！
)

accelerate launch "${CMD_ARGS[@]}"
