#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe_handroi_cls_20m_projdrop05
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
  export WANDB_TAGS="next-frame,hand-roi,cls,20m,fixed-prompt,proj-dropout"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-0905.224x224"
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
  # FSDP2: shards the frozen LLM across the job's GPUs. Comment this line out
  # to fall back to plain DDP. Qwen3-4B fits on each L40S for this frozen-base
  # probe, while DDP keeps the execution path simpler than FSDP2.
  # --config_file="$SCRIPT_DIR/configs/accelerate/fsdp2.yaml"
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  # Validation run for 表 B's recommended final structure
  # (.ai/visual_adapter_component_ablation_summary.md):
  #
  #   行 3  next-frame patch fusion, fusion_gate init +1.0   (strongest module)
  #   行 10 + gated hand-ROI residual                        (table's best eval)
  #   行 2  + gated CLS residual                             (+0.8, near free)
  #   行 9  + projection dropout 0.5 on every branch         (only op that cut
  #         the train-dev gap and raised eval together)
  #
  # This is the spatiotemporal_next_frame_hand_roi_cls adapter (行 10's global +
  # ROI branches plus the CLS residual), sized to 20M in
  # qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-cls-20m. Everything else
  # -- gate1, hardmatch, wr3, the adapter's default zero displacement init, the
  # fixed canonical prompt and single-language de -- is baseline_ablation's, so
  # this run sits on the same 表 B curve as 行 1-12.
  #
  # Reference (行 10 alone, no CLS, no dropout): eval 12.3, train-dev gap 49.7.
  # The three dropouts test whether 行 9's regularizer brings that gap down while
  # holding eval. The gap only opens after ~24k, so this needs >=30k before it
  # says anything about generalization.
  #
  # projection_dropout / roi_projection_dropout / cls_projection_dropout all sit
  # after their branch projection's GELU (Linear -> GELU -> Dropout -> Linear).
  # On the two residual branches the trailing non-affine LayerNorm restores the
  # norm, so 0.5 there is strong direction noise on the ~24-patch hand pool and
  # the CLS vector, not magnitude damping -- watch the roi_gate / cls_gate logs;
  # if either collapses toward zero, back that branch's dropout down first.
  # spatial_dropout is left at 0 (表 B 行 11: it did not move the gap).
  #
  # Requires engine.trainability.visual_adapter.runtime_mode = train, which
  # baseline_ablation inherits, or dropout silently stays off in eval.
  --config-name=train/pretrain_adapter/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-cls-20m
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  # Temporary: run the three scalar gates (patch-fusion / roi / cls, exposed by
  # the adapter's optimization_parameter_groups() as the "gates" group) 10x
  # above the adapter's own rate so they can move to their equilibrium before
  # the projections lock in. "+" because engine.optimization is an empty {} in
  # baseline_ablation and struct mode rejects a plain deep-key add.
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
