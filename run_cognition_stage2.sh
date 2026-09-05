#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe20m_stage2_llora_all
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
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="stage2,llm-lora,all-layers,r8,adapter-ft,next-frame-20m"
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

# Stage-1 checkpoint this continues: dev/test weighted BLEU-4 = 0.1075 at
# 42,000 steps (23.7 epochs), LLM fully frozen throughout.
STAGE1_CKPT=/mnt/scratch/users/2533494w/slt_outputs/v5.0-qwen3-4b-cradio-l-spatiotemporal-next-frame20m-lm1-hardmatch-wr3-ts2-0901.224x224-de/checkpoint-42000

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/cognition/stage2_llm_lora
  model.checkpoint_dir="$STAGE1_CKPT"
  engine.training_args.output_dir=outputs/v5.0-qwen3-4b-nextframe20m-stage2-llora-all-qv-r8a16-dp0.1-adapter1e-5-ep12-0903.224x224-de
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"

# --- Ablation ladder, one variable each, run only after A's step-0 gate passes.
#
# B  adapter frozen, everything else identical -- the direct test of question 1
#    engine.trainability.visual_adapter.parameter_mode=frozen
#    engine.trainability.visual_adapter.runtime_mode=eval
#
# C  last-4 layers only at matched total rank -- the direct test of question 2
#    peft.llm_lora_config.layers_to_transform=[32,33,34,35]
#    peft.llm_lora_config.r=32 peft.llm_lora_config.lora_alpha=64
#
# D  wider all-layer coverage if A is under-fitting rather than over-fitting
#    peft.llm_lora_config.target_modules=[q_proj,k_proj,v_proj,o_proj]
#
# E  NEFTune on the text embeddings only (visual soft tokens untouched)
#    engine.training_args.neftune_noise_alpha=5.0
