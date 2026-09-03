#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe_handroi20m_gated_lm1
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
  export WANDB_TAGS="next-frame-fusion,hand-roi,20m,lm1,hardmatch,wr3,gate-init-1.0,displacement-kaiming,roi-gated,roi-gate-init--2.0,no-dropout"
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
  # The two branches that moved dev BLEU-4 put in one adapter: patches are
  # fused with their next-frame match, then pooled both globally and over the
  # frozen scorer's hand ROI, with the ROI half riding in as a gated residual.
  # Still parameter-matched to the pooled-linear baseline (19,999,694 against
  # 19,999,369, +0.0016%), so the comparison stays a comparison of information
  # paths rather than of capacity.
  #
  # This is not a one-variable ablation -- it combines two -- so it is read
  # against both of its parents rather than against one reference:
  #
  #   next-frame dispkaiming  dev 0.0948 @18k  0.0990 @24k  0.1114 @30k
  #   next-frame gate1        dev 0.1066 @18k  0.1135 @24k
  #   hand-ROI gated          dev 0.0728 @18k  0.0865 @24k
  #
  # dispkaiming is the code-matched one: the fan-in displacement init landed in
  # NextFramePatchFusion itself, so this run inherits it with no flag. Note it
  # was behind gate1 at both 18k and 24k, i.e. that init has not paid off so
  # far, which is a confound this run carries and cannot separate.
  #
  # gated, not concat, for the ROI half: on the standalone hand-ROI pair gated
  # reached 0.0865 dev with a zero train-dev gap against concat's 0.0824 at
  # +0.007, and the point of combining is to add a path without adding a way to
  # memorize. Both gates are learnable and both are logged
  # (visual_adapter/motion_gate, visual_adapter/roi_gate), so what each branch
  # ends up carrying is readable from the run.
  #
  # Both dropouts stay 0. .ai/overfitting_component_attribution.md argues the
  # projection hidden layer is where the memorizing happens, but turning it on
  # here would confound the combination with the regularizer; that is the next
  # run, once this one has a curve.
  #
  # The gap only opens after ~30k steps in these runs, so this needs >=30k
  # before it says anything about generalization.
  --config-name=train/cognition/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-20m
  engine.training_args.output_dir=outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi20m-gated-lm1-hardmatch-wr3-gate1-roigate-2.0-0903.224x224
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
