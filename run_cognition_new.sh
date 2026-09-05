#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe20m_spatialdrop05
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
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,20m,fixed-prompt,spatial-dropout"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe20m-lm1-hardmatch-wr3-gate1-dispkaiming-spatialdrop0.5-0905.224x224"
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
  # Spatial dropout on top of the nextframe20m dispkaiming run, one variable
  # changed. Everything else -- adapter, gate1, hardmatch, wr3, the fan-in
  # displacement init, and the fixed canonical prompt (baseline_ablation's
  # default, not diverse_train) -- is what
  # outputs/v5.0-qwen3-4b-cradio-l-nextframe20m-lm1-hardmatch-wr3-gate1-dispkaiming-0902.224x224
  # ran, so that run is the reference curve:
  #
  #            dev      train probe   gap
  #   18k    0.0948       0.0913    -0.0035
  #   24k    0.0990       0.1126    +0.0136
  #   30k    0.1114       0.1456    +0.0342   (best, test BLEU-4 0.1114)
  #
  # The gap only opens after ~24k, so this needs >=30k before it says anything
  # about generalization; before that the two curves are expected to sit on top
  # of each other, or this one slightly lower for the noise dropout adds.
  #
  # spatial_dropout only, projection_dropout left at 0.
  # .ai/overfitting_component_attribution.md ranks the projection hidden layer
  # higher and recommends both together, but running both at once cannot say
  # which one paid; this measures the spatial half alone first.
  #
  # p=0.5: SpatialDropoutMean draws the mask per frame and renormalizes by the
  # survivors, so at 196 patches the mean stays a good estimate while no single
  # patch -- backdrop, logo, the signer's face -- is reliable enough to key on,
  # which is the shortcut it is aimed at. The drop happens after the fusion,
  # never before it: the fusion matches patches against a spatial neighbourhood
  # in the next frame, so dropping first would break correspondences instead of
  # a shortcut.
  #
  # Requires engine.trainability.visual_adapter.runtime_mode = train, which
  # baseline_ablation inherits, or the module silently stays in eval and this
  # is a plain mean.
  --config-name=train/cognition/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-20m
  model.config.visual_adapter_kwargs.spatial_dropout=0.5
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
