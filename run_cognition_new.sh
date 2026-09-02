#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe20m_gate1_lm1
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
  export WANDB_TAGS="next-frame-fusion,20m,lm1,hardmatch,wr3,gate-init-1.0"
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
  # Next-frame fusion, repeating the wr3/hardmatch run with a more open gate.
  #
  # Overridden here rather than in the config because gate_init is being swept:
  # the gate is optimized against the training loss, and both improved adapters
  # already overfit (train-dev BLEU-4 gap +0.043 by step 30k against the
  # baseline's +0.004), so which value generalizes has to be picked on dev
  # rather than handed to a faster learning rate. The config keeps 0.0, the run
  # that produced test BLEU-4 0.1075.
  #
  # sigmoid(1.0) = 0.73 against the previous 0.50. Every other adapter kwarg is
  # unchanged from that checkpoint, verified field by field.
  #
  # CAVEAT: this is not a single-variable change. NextFramePatchFusion's
  # displacement projection now uses fan-in init instead of zeros, because zeros
  # left it at 0.9% of the fusion hidden vector after 42k steps against
  # content's 69%. That is a deliberate confound, accepted for this run.
  --config-name=train/cognition/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-20m
  model.config.visual_adapter_kwargs.patch_fusion_gate_init=1.0
  engine.training_args.output_dir=outputs/v5.0-qwen3-4b-cradio-l-nextframe20m-lm1-hardmatch-wr3-gate1-0902.224x224
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
