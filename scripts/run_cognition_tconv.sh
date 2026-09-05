#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe_handroi20m_tconv_r1
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
  export WANDB_TAGS="next-frame-handroi,20m,fixed-prompt,temporal-conv"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi20m-gated-lm1-hardmatch-wr3-gate1-roigate-2.0-tconv-r1-0905.224x224"
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
  # Does a learnable temporal filter beat the fixed window mean? The only
  # change against
  # outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi20m-gated-lm1-hardmatch-wr3-gate1-roigate-2.0-0903.224x224
  # is the step that turns two frames into one token: that run's exact
  # non-overlapping mean becomes TemporalConvDownsample, a depthwise
  # per-channel filter over the pooled [global; roi] vector. Diffed field by
  # field against that run's saved hydra_config: model config and every
  # training hyperparameter (lr 1e-4, cosine, wd 0.05, bs 2, fixed prompt)
  # are identical, so it is the reference curve:
  #
  #            dev      train probe   gap
  #   18k    0.1053       0.1164    +0.0111
  #   24k    0.1108       0.1510    +0.0403
  #   30k    0.1238       0.2037    +0.0799   (best)
  #
  # Capacity is not the variable: 20,011,214 trainable against that run's
  # 19,999,694, +11,520 (+0.058%), exactly the depthwise conv's 2304*4 weights
  # plus 2304 biases. Measured by instantiating both adapters, not read off
  # the config comment.
  #
  # radius=1 is two changes at once, deliberately. kernel_size =
  # temporal_scale_factor + 2*radius = 4, so the filter is both learnable and
  # wider than the 2-frame window it replaces, and at step 0 it is a uniform
  # 4-frame mean rather than the reference's exact 2-frame mean. That is the
  # version with a real chance of helping; if it wins, temporal_conv_radius=0
  # separates "learnable" from "wider" (radius=0 reproduces the reference mean
  # numerically at step 0), and if it loses there is nothing to separate.
  #
  # What tempers the expectation: the filter is depthwise on the pooled
  # 2304-dim vector, so it can only reweight each channel across 4 frames --
  # no cross-channel temporal mixing. 11,520 parameters is the whole
  # hypothesis, and this run is the first evidence on this branch about
  # whether they buy anything.
  #
  # conv.weight is ndim=3, so unlike the reference's (nonexistent) pooling
  # parameters it receives weight_decay 0.05. AdamW's decoupled decay shrinks
  # all four taps equally and the following LayerNorm absorbs a uniform scale,
  # so this should not change what the filter learns; it is noted because it is
  # the one parameter in this run with no counterpart in the reference.
  #
  # Needs >=30k before it says anything: the reference's gap only opens after
  # 24k, and its 0.1238 is a 30k number.
  --config-name=train/cognition/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-conv-20m
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
