#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe20m_dispkaiming_lm1
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
  export WANDB_TAGS="next-frame-fusion,20m,lm1,hardmatch,wr3,gate-init-1.0,displacement-kaiming"
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
  # Ablation of one thing: NextFramePatchFusion's displacement stream now uses
  # fan-in init instead of zeros. Every other setting matches the reference run
  # outputs/...-nextframe20m-lm1-hardmatch-wr3-gate1-0902..., field by field
  # from its own hydra_config -- same adapter, gate_init 1.0, hidden 1024,
  # window radius 3, hard matching, rank 4429, temporal factor 2.
  #
  # Why: zeros left displacement at 0.9% of the fusion hidden vector after 42k
  # steps, against content's 69% and delta's 74%, with its weight norm still
  # climbing rather than settling. The handicap is structural -- 2 dimensions of
  # unit scale against 1152 LayerNormed ones -- so starting at zero penalises it
  # twice for the same thing. Fan-in starts it at 15.2%.
  #
  # Read the run against the reference's dev curve (0.1066 @18k, 0.1135 @24k)
  # and against visual_adapter/mean_displacement in the logs.
  --config-name=train/cognition/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-20m
  engine.training_args.output_dir=outputs/v5.0-qwen3-4b-cradio-l-nextframe20m-lm1-hardmatch-wr3-gate1-dispkaiming-0902.224x224
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
